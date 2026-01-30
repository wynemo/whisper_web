import os

from fastapi import APIRouter, Depends, File, UploadFile

from auth import get_current_user
from models import User

router = APIRouter(tags=["files"])

UPLOAD_DIR = "upload"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload/")
async def upload_file(file: UploadFile = File(...), _user: User = Depends(get_current_user)):
    """保存上传的文件到指定目录"""
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    return {"filename": file.filename, "filepath": file_path}


@router.get("/files/")
async def list_files(_user: User = Depends(get_current_user)):
    """获取 upload 目录下的文件列表"""
    files = os.listdir(UPLOAD_DIR)
    return {"files": files}
