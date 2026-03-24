import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from config import settings
from services.extract_service import extract_structured_resume
from services.pdf_to_txt import pdf_to_txt
from storage.file_store import new_resume_id, save_pdf_bytes, save_result_json, save_txt
from utils.errors import (
    AppError,
    CorruptedPDFError,
    EncryptedPDFError,
    FileSizeError,
    InvalidFileType,
    LLMParseError,
    NotResumeError,
    PDFParseError,
)
from utils.logger import get_logger

router = APIRouter()  #创建一个路由容器，后续会把不同接口注册到这个router里面
logger = get_logger("api") # 日志记录器，用于记录接口调用日志


async def read_upload_with_limit(  
    file: UploadFile,
    max_bytes: int,
    content_length: Optional[int],
    chunk_size: int = 1024 * 1024,
) -> bytes:  #定义一个异步函数：从 FastAPI 的 UploadFile 里把上传内容读出来，读的时候限制最大字节数，最后返回原始二进制 bytes 。用 async 的原因： UploadFile.read() 是异步 IO，避免阻塞服务线程。
    if content_length is not None and content_length > max_bytes: #文件过大直接拒绝
        raise HTTPException(status_code=413, detail="文件过大")

    data = bytearray() #准备可变字节数组容器，用来累计每个chunk
    total = 0  #当前已经读取的字节数
    while True:
        chunk = await file.read(chunk_size) #这里 await 表示“等 IO 读完再继续”，期间事件循环可以去处理别的请求。
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail="文件过大")
        data.extend(chunk)
    return bytes(data)


@router.get("/")  #打开服务后第一眼看到的导航页 + 运行状态确认
def index():
    return JSONResponse(
        {
            "message": "ok",
            "docs": "/docs",
            "endpoints": ["/api/upload", "/api/extract", "/api/parse"],
        }
    )


@router.post("/api/upload") #和 GET 最大区别：GET 通常是“读取/查询”，POST 通常是“提交/产生变化
async def upload_resume(request: Request, file: UploadFile = File(...)):
    if not file.filename: #如果没有文件名，就拒绝
        raise HTTPException(status_code=400, detail="未选择文件")

    if len(file.filename) > settings.MAX_FILENAME_LENGTH: #文件名过长，就拒绝
        raise HTTPException(status_code=400, detail="文件名过长")

    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS: #文件类型不支持，就拒绝
        raise InvalidFileType("仅支持 PDF 文件")

    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            content_length = int(content_length)
        except ValueError:
            content_length = None

    content = await read_upload_with_limit(
        file,
        settings.MAX_UPLOAD_BYTES,
        content_length,
    )
    if not content:
        raise HTTPException(status_code=400, detail="文件内容为空")

    if not content[:1024].lstrip().startswith(b"%PDF-"): #防止伪装pdf文件
        raise InvalidFileType("仅支持 PDF 文件")

    resume_id = new_resume_id() #生成一个新的resume_id
    logger.info("Uploaded resume %s", resume_id)
    pdf_path = save_pdf_bytes(resume_id, content) #保存pdf文件到指定路径

    try:
        text = pdf_to_txt(pdf_path)
    except FileSizeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (EncryptedPDFError, CorruptedPDFError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PDFParseError as exc:
        try:
            pdf_path.unlink()
        except Exception:
            pass
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    txt_path = save_txt(resume_id, text) #保存解析后的文本到指定路径
    logger.info("Parsed resume %s to %s", resume_id, txt_path.name)

    return JSONResponse(
        {
            "resume_id": resume_id,
            "text": text,
            "txt_path": f"storage/txts/{txt_path.name}",
        }
    )


@router.post("/api/upload/batch")
async def upload_resume_batch(request: Request, files: list[UploadFile] = File(...)):
    succeeded = []
    failed = []

    for file in files:
        filename = file.filename or "<unknown>"

        if not filename:
            failed.append({"filename": filename, "reason": "No filename provided"})
            continue

        ext = Path(filename).suffix.lower()
        if ext not in settings.ALLOWED_EXTENSIONS:
            failed.append({"filename": filename, "reason": f"Unsupported file type: {ext}"})
            continue

        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                content_length = int(content_length)
            except ValueError:
                content_length = None

        try:
            content = await read_upload_with_limit(
                file,
                settings.MAX_UPLOAD_BYTES,
                content_length,
            )
        except HTTPException as exc:
            logger.error("[BATCH] Skipping %s: HTTP %s — %s", filename, exc.status_code, exc.detail)
            failed.append({"filename": filename, "reason": exc.detail})
            continue

        if not content:
            logger.error("[BATCH] Skipping %s: file content is empty", filename)
            failed.append({"filename": filename, "reason": "File content is empty"})
            continue

        resume_id = new_resume_id()
        pdf_path = save_pdf_bytes(resume_id, content)

        try:
            text = pdf_to_txt(pdf_path)
        except (FileSizeError, EncryptedPDFError, CorruptedPDFError, PDFParseError) as exc:
            logger.error(
                "[BATCH] Skipping %s: %s: %s",
                filename,
                type(exc).__name__,
                exc,
            )
            failed.append({"filename": filename, "reason": str(exc)})
            continue
        except Exception as exc:
            logger.error("[BATCH] Unexpected error for %s: %s", filename, exc)
            failed.append({"filename": filename, "reason": f"Unexpected error: {exc}"})
            continue

        txt_path = save_txt(resume_id, text)
        logger.info("[BATCH] Parsed %s -> resume_id=%s txt=%s", filename, resume_id, txt_path.name)
        succeeded.append(
            {
                "resume_id": resume_id,
                "filename": filename,
                "txt_path": f"storage/txts/{txt_path.name}",
            }
        )

    return JSONResponse(
        {
            "total": len(files),
            "succeeded_count": len(succeeded),
            "failed_count": len(failed),
            "succeeded": succeeded,
            "failed": failed,
        }
    )


@router.post("/api/parse")
async def parse_resume(request: Request, file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="未选择文件")

    if len(file.filename) > settings.MAX_FILENAME_LENGTH:
        raise HTTPException(status_code=400, detail="文件名过长")

    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise InvalidFileType("仅支持 PDF 文件")

    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            content_length = int(content_length)
        except ValueError:
            content_length = None

    content = await read_upload_with_limit(
        file,
        settings.MAX_UPLOAD_BYTES,
        content_length,
    )
    if not content:
        raise HTTPException(status_code=400, detail="文件内容为空")

    if not content[:1024].lstrip().startswith(b"%PDF-"):
        raise InvalidFileType("仅支持 PDF 文件")

    resume_id = new_resume_id()
    pdf_path = save_pdf_bytes(resume_id, content)

    try:
        text = pdf_to_txt(pdf_path)
    except PDFParseError as exc:
        try:
            pdf_path.unlink()
        except Exception:
            pass
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        from schemas.models import ExtractionInput

        extraction_input = ExtractionInput(text=text, resume_id=resume_id)
        structured = extract_structured_resume(extraction_input)
    except NotResumeError as exc:
        try:
            pdf_path.unlink()
        except Exception:
            pass
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (LLMParseError, AppError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    txt_path = save_txt(resume_id, text)
    logger.info("Parsed resume %s to %s", resume_id, txt_path.name)

    json_text = structured.model_dump_json(ensure_ascii=False)
    save_result_json(resume_id, json_text)

    return JSONResponse(
        {
            "resume_id": resume_id,
            "result": json.loads(json_text),
        }
    )


@router.post("/api/extract")
async def extract_resume(payload: dict): #输入是一个字典，包含 resume_id 和 text 两个键值对，输入是一个字典，包含 resume_id 和 text 两个键值对，输出是一个 JSON 字符串，包含解析后的简历信息
    resume_id = payload.get("resume_id")
    text = payload.get("text")

    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=400, detail="text 不能为空")

    from schemas.models import ExtractionInput

    extraction_input = ExtractionInput(text=text, resume_id=resume_id if isinstance(resume_id, str) else None)
#抽取输入
    try:
        structured = extract_structured_resume(extraction_input)
    except NotResumeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (LLMParseError, AppError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
#llm调用失败返回502，含义为网关，上游服务出错
    json_text = structured.model_dump_json(ensure_ascii=False)  
    if extraction_input.resume_id:  #落盘保存（方便后续复用）
        save_result_json(extraction_input.resume_id, json_text)

    return JSONResponse(json.loads(json_text))
