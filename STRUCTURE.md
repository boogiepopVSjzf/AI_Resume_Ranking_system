# 项目结构说明

## 根目录
- main.py：程序入口，负责初始化应用并串联上传、转换、抽取流程
- requirements.txt：Python 依赖清单

## routes
- 作用：接口层，接收外部请求并返回结果
- __init__.py：包初始化

## services
- 作用：核心业务逻辑与编排
- __init__.py：包初始化
- pdf_to_txt.py：PDF 转 TXT 的解析与文本抽取
- text_clean_service.py：PDF 文本清洗与格式规范化
- llm_service.py：LLM API 调用与响应解析
- extract_service.py：从 TXT 到结构化数据的流程编排

## storage
- 作用：存储层，保存 PDF、TXT、结构化结果
- __init__.py：包初始化
- file_store.py：本地文件读写与路径管理
- pdfs/：原始 PDF 文件
- txts/：解析后的 TXT 文件
- results/：结构化结果（JSON）

## schemas
- 作用：结构化数据模型定义与校验
- __init__.py：包初始化
- models.py：简历结构化字段模型

## config
- 作用：配置管理
- __init__.py：包初始化
- settings.py：配置加载与默认值管理
  - LLM_BASE_URL / LLM_MODEL：Dashscope 兼容接口配置
  - LLM_API_KEY：从环境变量 LLM_API_KEY 读取（Dashscope 使用）
  - DEFAULT_LLM_PROVIDER / DEFAULT_LLM_MODEL：默认路由与默认模型
  - OPENAI_API_URL / OPENAI_MODEL / OPENAI_API_KEY：OpenAI 配置
  - GEMINI_API_URL_TEMPLATE / GEMINI_MODEL / GEMINI_API_KEY：Gemini 配置
  - OLLAMA_API_URL / OLLAMA_MODEL：Ollama 配置
  - LLM_TIMEOUT_SECONDS：请求超时（秒，OpenAI/Gemini/Dashscope 共用）
  - MAX_UPLOAD_MB / ALLOWED_EXTENSIONS：上传限制与允许类型

## utils
- 作用：通用工具能力
- __init__.py：包初始化
- logger.py：日志封装
- errors.py：自定义错误与异常

## tests
- 作用：测试用例
- __init__.py：包初始化
- test_text_clean_service.py：文本清洗规则测试
- test_extract_service.py：结构化抽取与 JSON 解析测试

# 上传与 PDF 解析阶段的异常防护（设计说明）

本项目第一步链路为：上传 PDF（routes）→ 保存 PDF（storage）→ PDF 转 TXT（services）→ 保存 TXT（storage）。

## 1) 文件类型与内容校验（拒绝非 PDF、伪装 PDF）

### 目标
- 只允许“真正的 PDF 文件”进入解析与抽取流程
- 拒绝扩展名为 `.pdf` 但内容不是 PDF 的文件

### 方案与落点
- `routes/api.py`
  - 通过 `Path(file.filename).suffix.lower()` + `ALLOWED_EXTENSIONS` 限制只接受 `.pdf`
  - 使用 `read_upload_with_limit` + `MAX_UPLOAD_BYTES` / `content-length` 进行 **流式体积限制**，超出返回 `413`
  - 读取到的二进制内容头部，检查是否以 `%PDF-` 开头（“PDF 魔数”校验），不符合直接抛出 `InvalidFileType`，返回 `400`
- `services/pdf_to_txt.py`
  - 调用前再次通过 `_validate_size` 对落盘后的文件做 **文件级体积校验**（避免中途被截断 / 损坏的伪装文件进入解析）
  - 发生解析异常时统一抛出 `PDFParseError` / 子类异常，供 `routes/api.py` 统一转换为 HTTP 错误码

Trouble‑shooting 提示：
- 如果接口返回 `400` 且消息为“仅支持 PDF 文件”，优先检查：是否真的是 PDF，或是否通过在线工具错误导出。
- 如果日志中看到“Failed to parse PDF”之类信息，通常说明文件内容结构异常或已损坏。

## 2) 文件大小与边界控制（避免过小 / 过大 / 截断文件）

### 目标
- 拦截明显不合理的超大文件，避免占用过多 IO / 内存
- 拦截过小、疑似空文件或上传过程被截断的 PDF

### 方案与落点
- `config/settings.py`
  - `MAX_UPLOAD_MB` / `MAX_UPLOAD_BYTES`：上传体积上限
  - `MIN_UPLOAD_BYTES`：PDF 最小体积阈值（过小视为无效文件）
- `routes/api.py`
  - `read_upload_with_limit` 在读取流时，超过 `MAX_UPLOAD_BYTES` 直接中断并返回 `413`
- `services/pdf_to_txt.py`
  - `_validate_size` 使用 `MIN_UPLOAD_BYTES` / `MAX_UPLOAD_BYTES` 对已保存的 PDF 再做一层校验  
    - 太小或太大时抛出 `FileSizeError`，由 `routes/api.py` 转为 `400`

Trouble‑shooting 提示：
- 返回 `413`：说明上传单个文件已经超过配置上限，需调整 `MAX_UPLOAD_MB` 或让用户压缩 / 精简 PDF。
- 返回 `400` 且包含 “File too small” 等字样：大概率是空文件、传输中断或服务端保存异常。

## 3) 文本层检测与图片 PDF（扫描件）拒绝

### 目标
- 对只有图片、没有文字层的 PDF（扫描件）进行识别并拒绝，避免“空文本”误进后续 LLM 链路

### 方案与落点
- `services/pdf_to_txt.py`
  - 正常使用 `PdfReader` + `page.extract_text()` 提取每页文本
  - 利用 `MIN_EXTRACTED_TEXT_CHARS` 对清洗后的文本进行最小长度校验：  
    - 去掉空白字符后字符数低于阈值，认为“无有效文字层”（疑似图片 PDF）
    - 抛出 `PDFParseError`，提示“PDF 不包含可提取的文本（疑似图片 PDF）”
- `routes/api.py`
  - 捕获 `PDFParseError` 后：
    - 删除对应的 `storage/pdfs/{resume_id}.pdf`，避免残留无效 PDF
    - 将错误映射为 `422`，表示“输入格式正确但内容不可解析”

Trouble‑shooting 提示：
- 用户上传扫描件时会收到 `422` 错误，可在前端提示“当前只支持带文字层的 PDF，请导出为可复制文本的 PDF 再试”。

## 4) 加密 / 损坏 PDF 的处理（区分可恢复 vs 不可恢复）

### 目标
- 明确区分“正常 PDF 但被加密”与“结构损坏的 PDF”
- 为前端提供不同的错误提示建议

### 方案与落点
- `services/pdf_to_txt.py`
  - 创建 `PdfReader` 后：
    - 如果 `reader.is_encrypted` 或抛出 `FileNotDecryptedError`，包装为 `EncryptedPDFError`
    - 如果抛出 `PdfReadError`，包装为 `CorruptedPDFError`
    - 其它解析异常统一包装为 `PDFParseError`
- `routes/api.py`
  - `upload` 接口中：
    - `EncryptedPDFError` / `CorruptedPDFError`：返回 `422`，表示“文件格式正确但无法解析”
    - 其它 `PDFParseError`：同样视为 `422`，并清理已保存的 PDF 文件

Trouble‑shooting 提示：
- 日志中出现 `EncryptedPDFError`：提醒用户导出“未加密版本”或去掉打开密码。
- 出现 `CorruptedPDFError`：说明文件结构已损坏，通常需要用户重新导出或重新获取原件。

## 5) 文件名、批量上传等边缘场景

### 目标
- 避免异常长文件名或批量上传中的单个问题文件影响整体流程

### 方案与落点
- `config/settings.py`
  - `MAX_FILENAME_LENGTH`：文件名长度上限（默认 128）
- `routes/api.py`
  - 单文件上传：
    - `filename` 为空：返回 `400`
    - 超过 `MAX_FILENAME_LENGTH`：返回 `400`
  - 批量上传（`/api/upload/batch`）：
    - 对每个文件分别应用上述校验与解析逻辑
    - 将成功与失败项分别记录到 `succeeded` / `failed` 列表中返回，方便排查是哪一个文件出问题

Trouble‑shooting 提示：
- 批量上传时，如只部分失败，可直接查看返回 JSON 中 `failed` 内的 `reason` 字段定位具体问题（类型、体积、加密、无文本等）。

# 可配置与个性化修改入口


## 可配置与个性化修改入口
- 更换大模型接口与模型名：config/settings.py（LLM_API_URL、LLM_MODEL）
- 更换 API Key：通过环境变量 LLM_API_KEY
- 调整 Prompt 规则：services/extract_service.py（build_prompt）
- 调整结构化输出格式：schemas/models.py（字段与类型定义）

# 本次改动明细（2026-03-13）

## 目标
- 选择“第二种策略”：当上传已落盘但后续解析失败时，清理无效 PDF，避免残留
- 同时修复若干会导致运行/测试失败的不一致点，保证程序可跑通

## 代码改动点
- 上传与解析失败清理：routes/api.py 的 /api/upload 与 /api/parse
  - 新增：文件名长度限制（MAX_FILENAME_LENGTH）
  - 新增：流式读取并限制上传大小（MAX_UPLOAD_BYTES，超限返回 413）
  - 新增：PDF 文件头（%PDF-）校验，拒绝“伪装 PDF”
  - 新增：PDFParseError 时删除已保存的 storage/pdfs/{resume_id}.pdf，并以 422 返回
- PDF 转 TXT：services/pdf_to_txt.py
  - 新增：加密 PDF 检测，抛出 PDFParseError
  - 新增：最小可提取文本阈值（MIN_EXTRACTED_TEXT_CHARS），拒绝疑似图片 PDF
- 路由/调用一致性修复
  - routes/api.py：/api/parse 调用 extract_structured_resume 改为传入 ExtractionInput（避免类型不匹配）
  - routes/upload.py：导入修正为 services/pdf_to_txt.py 的 pdf_to_txt（避免引用不存在的 services.pdf_service）
  - routes/extract.py：修正缩进，避免语法/逻辑错误
  - tests：修正导入与配置字段，确保 pytest 可运行
