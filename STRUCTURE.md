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

下面仅覆盖当前要考虑的三类场景，并说明“应该在哪个文件加什么样的校验/分支处理”，不涉及具体代码实现。

## 1) 必须是文本 PDF（拒绝非 PDF、伪装 PDF、图片 PDF）

### 目标
- 只允许“真正的 PDF 文件”进入解析流程
- 拒绝扩展名是 .pdf 但内容不是 PDF 的文件
- 拒绝只有图片、没有文字层的 PDF（即扫描件/图片 PDF）

### 方案与落点
- routes/api.py（上传入口）
  - 现有：根据文件扩展名限制只接受 .pdf，并做上传大小限制
  - 补充：在读取完整内容后，增加“PDF 魔数/文件头”校验
    - 检查前若干字节是否包含 PDF 标识（例如以 %PDF- 开头）
    - 不符合则直接返回 400（InvalidFileType / HTTPException）并中断流程
  - 补充：若后续解析阶段判定 PDF 不符合要求（例如加密/图片 PDF），应清理已落盘的 PDF
    - 解析失败时删除 storage/pdfs/{resume_id}.pdf，避免残留无效文件
- services/pdf_to_txt.py（解析入口）
  - 补充：对“图片 PDF”的判定与拒绝
    - 解析后如果提取出的文本为空或仅包含极少可见字符（可用阈值判断，例如去掉空白后长度很小），视为“无文字层”
    - 这种情况抛出 PDFParseError（例如“疑似扫描件 PDF，不支持”），由 routes/api.py 捕获后返回 4xx/5xx（建议 422 或 400）

## 2) 文件名限制（拒绝超长文件名）

### 目标
- 防止超长文件名导致存储/日志/文件系统异常
- 降低路径穿越和奇怪字符带来的风险（即便当前存储不使用原始文件名，也建议做基础校验）

### 方案与落点
- config/settings.py
  - 增加可配置项：MAX_FILENAME_LENGTH（例如 128 或 255）
- routes/api.py
  - 在读取文件内容前，校验 UploadFile.filename
    - filename 为空：返回 400
    - filename 长度超过 MAX_FILENAME_LENGTH：返回 400
    - 可选：对不可见字符、路径分隔符进行拒绝或规范化（本阶段不强制）

## 3) 加密 PDF（受保护导致无法读取）

### 目标
- 明确区分“PDF 损坏/解析失败”与“PDF 加密导致无法读取”
- 对加密 PDF 给出可理解的错误提示（例如提示用户导出未加密版本）

### 方案与落点
- services/pdf_to_txt.py
  - 在创建/读取 PdfReader 后，优先检测 is_encrypted
    - 如果加密且无法解密，抛出 PDFParseError（例如“PDF 已加密，无法解析”）
    - 若未来要支持密码，可扩展为在此处接入密码参数（当前不做）
- routes/api.py
  - 复用现有的 PDFParseError 捕获逻辑，将错误转换为 HTTP 返回
    - 建议把“加密 PDF”作为 422（输入不可处理）或 400（输入不合规）返回，而不是 500

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
