from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import settings
from routes.api import router as api_router
from routes.extract import routes as extract_router
from routes.llm import routes as llm_router
from routes.upload import router as upload_router
from utils.errors import InvalidFileType

app = FastAPI()

# Mount frontend static files with error handling
try:
    frontend_dir = settings.BASE_DIR / "frontend"
    if frontend_dir.exists():
        app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
        
        @app.get("/")
        def index():
            return FileResponse(frontend_dir / "index.html")
except Exception:
    pass

# Include all routers
app.include_router(api_router)  # feature_jzf main router
app.include_router(upload_router)
app.include_router(extract_router)
app.include_router(llm_router)


@app.exception_handler(InvalidFileType)
def invalid_file_handler(request, exc: InvalidFileType):
    return JSONResponse(status_code=400, content={"detail": str(exc)})