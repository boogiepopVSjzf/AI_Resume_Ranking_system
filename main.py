from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from routes.auth import router as auth_router

from config import settings
from routes.api import router as api_router
from routes.llm import routes as llm_router
from services.embedding_service import preload_embedding_model
from utils.errors import InvalidFileType
from utils.logger import get_logger

app = FastAPI()  #主接口，用于把后续接口集合挂载上去
logger = get_logger("main")

# 后续写前端接口就在这里挂载上去，目前没有就直接pass
try:   
    frontend_dir = settings.BASE_DIR / "frontend"
    if frontend_dir.exists():
        app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
        
        @app.get("/")
        def index():
            return FileResponse(frontend_dir / "index.html")
except Exception:
    pass

# 把各个接口封装到总服务接口上，目前提供四个接口，也就是说在前端fastapi上可以调用4个接口
app.include_router(api_router)  # 
app.include_router(llm_router)
app.include_router(auth_router)


@app.on_event("startup")
def startup_tasks() -> None:
    if not settings.PRELOAD_EMBEDDING_MODEL:
        return
    try:
        preload_embedding_model()
        logger.info("Embedding model preloaded at startup")
    except Exception as exc:
        logger.warning("Embedding model preload skipped: %s", exc)


@app.exception_handler(InvalidFileType) #如果整个服务运行过程中出现 InvalidFileType 这个异常，就交给下面那个函数处理 ，而不是让程序崩掉或返回默认的 500。
def invalid_file_handler(request, exc: InvalidFileType):
    return JSONResponse(status_code=400, content={"detail": str(exc)})
