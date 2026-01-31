from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi.responses import JSONResponse
from sqlmodel import Session, select

from auth import (
    COOKIE_NAME,
    create_access_token,
    get_current_user,
    verify_password,
)
from config import settings
from db import get_session
from models import User

router = APIRouter(prefix="/auth", tags=["auth"])


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
