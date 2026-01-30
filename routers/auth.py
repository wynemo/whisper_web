from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlmodel import Session, select

from auth import (
    COOKIE_NAME,
    create_access_token,
    generate_salt,
    get_current_user,
    hash_password,
    verify_password,
)
from config import settings
from db import get_session
from models import User

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    username: str
    password: str


@router.post("/register")
async def register(request: RegisterRequest, session: Session = Depends(get_session)):
    """用户注册"""
    existing = session.exec(
        select(User).where(User.username == request.username)
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="用户名已存在")

    salt = generate_salt()
    hashed = hash_password(request.password, salt)
    user = User(
        username=request.username,
        hashed_password=hashed,
        salt=salt,
    )
    session.add(user)
    session.commit()
    session.refresh(user)

    return {"id": user.id, "username": user.username}


@router.post("/login")
async def login(
    username: str = Form(...),
    password: str = Form(...),
    session: Session = Depends(get_session),
):
    """用户登录（OAuth2 表单格式），返回 httponly cookie"""
    user = session.exec(select(User).where(User.username == username)).first()
    if not user or not verify_password(password, user.hashed_password, user.salt):
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    token = create_access_token(data={"sub": user.username})
    response = JSONResponse(content={"message": "登录成功", "username": user.username, "token": token})
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
    )
    return response


@router.post("/logout")
async def logout():
    """用户登出，清除 httponly cookie"""
    response = JSONResponse(content={"message": "已登出"})
    response.delete_cookie(key=COOKIE_NAME)
    return response


@router.get("/me")
async def get_me(user: User = Depends(get_current_user)):
    """获取当前登录用户信息"""
    return {"id": user.id, "username": user.username}
