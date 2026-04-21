from datetime import datetime, timedelta, timezone
import os
import secrets

from jose import JWTError, jwt
from fastapi import HTTPException, status

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-secret-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "480"))

# 今天先用 demo 账户，够做真正的 auth/authz 演示
_HR_USERNAME = os.getenv("HR_DEMO_USERNAME", "hr_demo")
_HR_PASSWORD = os.getenv("HR_DEMO_PASSWORD", "hr123456")
_INTERNAL_USERNAME = os.getenv("INTERNAL_DEMO_USERNAME", "internal_demo")
_INTERNAL_PASSWORD = os.getenv("INTERNAL_DEMO_PASSWORD", "internal123456")

DEMO_USERS = {
    _HR_USERNAME: {
        "username": _HR_USERNAME,
        "password": _HR_PASSWORD,
        "role": "hr",
    },
    _INTERNAL_USERNAME: {
        "username": _INTERNAL_USERNAME,
        "password": _INTERNAL_PASSWORD,
        "role": "internal",
    },
}


def authenticate_user(username: str, password: str):
    user = DEMO_USERS.get(username)
    if not user:
        return None

    # 用 constant-time compare，今天先别碰 bcrypt/passlib
    if not secrets.compare_digest(password, user["password"]):
        return None

    return {
        "username": user["username"],
        "role": user["role"],
    }


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        role = payload.get("role")

        if not username or not role:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )

        return {"username": username, "role": role}

    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        ) from exc