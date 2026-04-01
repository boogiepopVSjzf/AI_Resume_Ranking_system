# 项目结构说明

## 根目录
- main.py：程序入口，负责初始化应用并串联上传、转换、抽取流程
- requirements.txt：Python 依赖清单

## routes
- 作用：接口层，接收外部请求并返回结果
- __init__.py：包初始化
- api.py: 核心业务 API
- llm.py: LLM 测试 API

## services
- 作用：核心业务逻辑与编排
- __init__.py：包初始化
- document_to_txt.py：文档转 TXT 的解析与文本抽取
- document_validate.py: 文档校验
- text_clean_service.py：PDF 文本清洗与格式规范化
- llm_service.py：LLM API 调用与响应解析
- extract_service.py：从 TXT 到结构化数据的流程编排
- upload_service.py: 上传服务

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
- api_models.py: API 请求/响应模型

## config
- 作用：配置管理
- __init__.py：包初始化
- settings.py：配置加载与默认值管理，集中定义 API Keys、文件路径、模型名称等所有可配置项。

## utils
- 作用：通用工具能力
- __init__.py：包初始化
- logger.py：日志封装
- errors.py：自定义错误与异常
- resume_validity_checker.py: 简历有效性检查

## tests
- 作用：测试用例
