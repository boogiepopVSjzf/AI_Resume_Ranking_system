from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import settings
from routes.extract import routes as extract_router
from routes.llm import routes as llm_router
from routes.upload import router as upload_router
from utils.errors import InvalidFileType

app = FastAPI()

app.mount("/static", StaticFiles(directory=settings.BASE_DIR / "frontend"), name="static")


@app.get("/")
def index():
    return FileResponse(settings.BASE_DIR / "frontend" / "index.html")


app.include_router(upload_router)
app.include_router(extract_router)
app.include_router(llm_router)


@app.exception_handler(InvalidFileType)
def invalid_file_handler(request, exc: InvalidFileType):
    return JSONResponse(status_code=400, content={"detail": str(exc)})