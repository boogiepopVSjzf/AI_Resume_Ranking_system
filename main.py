from fastapi import FastAPI
from fastapi.responses import JSONResponse

from routes.api import router as api_router
from utils.errors import InvalidFileType

app = FastAPI()
app.include_router(api_router)


@app.exception_handler(InvalidFileType)
def invalid_file_handler(request, exc: InvalidFileType):
    return JSONResponse(status_code=400, content={"detail": str(exc)})
