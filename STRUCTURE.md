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
- pdf_service.py：PDF 转 TXT 的解析与文本抽取
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
  - LLM_API_URL / LLM_MODEL：大模型调用配置
  - LLM_API_KEY：从环境变量 LLM_API_KEY 读取

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

## frontend
- 作用：前端页面与静态资源，用于上传简历与展示解析结果
- index.html：页面入口与基础布局
- app.js：前端交互逻辑（上传文件、调用后端接口、渲染结果）
- styles.css：页面样式
