from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import settings
from routes.api import router as api_router
from routes.llm import routes as llm_router
from utils.errors import InvalidFileType

app = FastAPI()  #主接口，用于把后续接口集合挂载上去

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


@app.exception_handler(InvalidFileType) #如果整个服务运行过程中出现 InvalidFileType 这个异常，就交给下面那个函数处理 ，而不是让程序崩掉或返回默认的 500。
def invalid_file_handler(request, exc: InvalidFileType):
    return JSONResponse(status_code=400, content={"detail": str(exc)})
