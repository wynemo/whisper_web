import hashlib
import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from jose import JWTError, jwt
from sqlmodel import Session, select
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse

from config import settings
from db import get_session
from models import User

ALGORITHM = "HS256"
COOKIE_NAME = "access_token"


class AuthGuardMiddleware(BaseHTTPMiddleware):
    """页面认证守卫：未登录用户访问受保护页面时 302 重定向到 /login"""

    PUBLIC_PATHS = {"/login", "/login/"}
    SKIP_PREFIXES = (
        "/auth/", "/api/", "/ws", "/upload/", "/files/",
        "/docs", "/redoc", "/openapi.json",
    )

    async def dispatch(self, request, call_next):
        path = request.url.path

        # 公开路径放行
        if path in self.PUBLIC_PATHS:
            return await call_next(request)

        # API / WebSocket 等路径放行（由端点自行认证）
        for prefix in self.SKIP_PREFIXES:
            if path.startswith(prefix):
                return await call_next(request)

        # 非 GET 请求放行（POST 等由端点自行返回 401）
        if request.method != "GET":
            return await call_next(request)

        # 校验 JWT cookie
        token = request.cookies.get(COOKIE_NAME)
        if token:
            try:
                payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
                if payload.get("sub"):
                    return await call_next(request)
            except JWTError:
                pass

        # 未认证，重定向到登录页
        return RedirectResponse(url="/login", status_code=302)


def generate_salt() -> str:
    return os.urandom(16).hex()


def hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((password + salt).encode()).hexdigest()


def verify_password(plain_password: str, hashed_password: str, salt: str) -> bool:
    return hash_password(plain_password, salt) == hashed_password


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(
    request: Request,
    session: Session = Depends(get_session),
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无法验证凭据",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # 优先从 httponly cookie 获取 token
    token = request.cookies.get(COOKIE_NAME)

    # 回退到 Authorization header（方便 API 调用）
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[len("Bearer "):]

    if not token:
        raise credentials_exception

    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = session.exec(select(User).where(User.username == username)).first()
    if user is None or not user.is_active:
        raise credentials_exception

    return user


def get_token_from_query(token: str, session: Session = Depends(get_session)) -> User:
    """WebSocket 认证：从 query parameter 获取 token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无法验证凭据",
    )

    if not token:
        raise credentials_exception

    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = session.exec(select(User).where(User.username == username)).first()
    if user is None or not user.is_active:
        raise credentials_exception

    return user
