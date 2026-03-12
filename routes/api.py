import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from config import settings
from services.extract_service import extract_structured_resume
# use pdf_service in this repo
from services.pdf_service import pdf_to_text as pdf_to_txt
from storage.file_store import new_resume_id, save_pdf_bytes, save_result_json, save_txt
from utils.errors import AppError, InvalidFileType, LLMParseError, PDFParseError
from utils.logger import get_logger

router = APIRouter()
logger = get_logger("api")


async def read_upload_with_limit(
    file: UploadFile,
    max_bytes: int,
    content_length: Optional[int],
    chunk_size: int = 1024 * 1024,
) -> bytes:
    if content_length is not None and content_length > max_bytes:
        raise HTTPException(status_code=413, detail="文件过大")

    data = bytearray()
    total = 0
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail="文件过大")
        data.extend(chunk)
    return bytes(data)


@router.get("/")
def index():
    return JSONResponse(
        {
            "message": "ok",
            "docs": "/docs",
            "endpoints": ["/api/upload", "/api/extract", "/api/parse"],
        }
    )


@router.post("/api/upload")
async def upload_resume(request: Request, file: UploadFile = File(...)):
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

    txt_path = save_txt(resume_id, text)
    logger.info("Parsed resume %s to %s", resume_id, txt_path.name)

    return JSONResponse(
        {
            "resume_id": resume_id,
            "text": text,
            "txt_path": f"storage/txts/{txt_path.name}",
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

    txt_path = save_txt(resume_id, text)
    logger.info("Parsed resume %s to %s", resume_id, txt_path.name)

    try:
        structured = extract_structured_resume(text)
    except (LLMParseError, AppError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    json_text = structured.model_dump_json(ensure_ascii=False)
    save_result_json(resume_id, json_text)

    return JSONResponse(
        {
            "resume_id": resume_id,
            "result": json.loads(json_text),
        }
    )


@router.post("/api/extract")
async def extract_resume(payload: dict):
    resume_id = payload.get("resume_id")
    text = payload.get("text")

    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=400, detail="text 不能为空")

    from schemas.models import ExtractionInput

    extraction_input = ExtractionInput(text=text, resume_id=resume_id if isinstance(resume_id, str) else None)

    try:
        structured = extract_structured_resume(extraction_input)
    except (LLMParseError, AppError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    json_text = structured.model_dump_json(ensure_ascii=False)
    if extraction_input.resume_id:
        save_result_json(extraction_input.resume_id, json_text)

    return JSONResponse(json.loads(json_text))
