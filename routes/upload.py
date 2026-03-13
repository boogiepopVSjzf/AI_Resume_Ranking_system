from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from config import settings
from services.pdf_to_txt import pdf_to_txt
from storage.file_store import new_resume_id, save_pdf_bytes, save_txt
from utils.errors import InvalidFileType, PDFParseError
from utils.logger import get_logger

router = APIRouter(prefix="/api", tags=["upload"])
logger = get_logger("upload")


@router.post("/upload")
async def upload_resume(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="未选择文件")

    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise InvalidFileType("仅支持 PDF 文件")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="文件内容为空")

    if len(content) > settings.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="文件过大")

    resume_id = new_resume_id()
    pdf_path = save_pdf_bytes(resume_id, content)

    try:
        text = pdf_to_txt(pdf_path)
    except PDFParseError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    txt_path = save_txt(resume_id, text)
    logger.info("Parsed resume %s to %s", resume_id, txt_path.name)

    return JSONResponse(
        {
            "resume_id": resume_id,
            "text": text,
            "txt_path": f"storage/txts/{txt_path.name}",
        }
    )
