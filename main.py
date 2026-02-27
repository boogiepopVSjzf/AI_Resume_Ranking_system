import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import settings
from services.extract_service import extract_structured_resume
from services.pdf_service import pdf_to_text
from storage.file_store import new_resume_id, save_pdf_bytes, save_result_json, save_txt
from storage.db_store import init_db, save_parsed_resume
from utils.errors import AppError, InvalidFileType, LLMParseError, PDFParseError
from utils.logger import get_logger

app = FastAPI()
logger = get_logger("app")

# Initialize the SQLite database schema on app startup
init_db()

app.mount("/static", StaticFiles(directory=settings.BASE_DIR / "frontend"), name="static")


@app.get("/")
def index():
    return FileResponse(settings.BASE_DIR / "frontend" / "index.html")


@app.post("/api/upload")
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
        text = pdf_to_text(pdf_path)
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


@app.post("/api/extract")
async def extract_resume(payload: dict):
    """
    Extract structured resume information from plain text.

    This endpoint:
      1) accepts resume_id and text from the frontend,
      2) calls the extraction service to get a ResumeStructured object,
      3) saves the result to both JSON file storage and SQLite database,
      4) returns the structured data as JSON to the client.
    """
    resume_id = payload.get("resume_id")
    text = payload.get("text")

    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=400, detail="text 不能为空")

    try:
        # Call the extraction service to get a structured resume
        structured = extract_structured_resume(text)
    except (LLMParseError, AppError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # Serialize the structured resume to JSON for file storage and response
    json_text = structured.model_dump_json(ensure_ascii=False)

    if isinstance(resume_id, str) and resume_id:
        # 1) Keep the existing behavior: save result as JSON file
        save_result_json(resume_id, json_text)
        # 2) New behavior: also persist the structured resume into SQLite DB
        save_parsed_resume(resume_id, structured)

    # Return the structured data back to the frontend
    return JSONResponse(json.loads(json_text))


@app.exception_handler(InvalidFileType)
def invalid_file_handler(request, exc: InvalidFileType):
    return JSONResponse(status_code=400, content={"detail": str(exc)})
