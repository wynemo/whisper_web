import argparse
import os
import logging

from fastapi import FastAPI
from fastapi_toolbox import run_server, NextJSRouteMiddleware, StaticFilesCache
from fastapi.middleware.cors import CORSMiddleware

from auth import AuthGuardMiddleware
from db import create_db_and_tables
from routers import auth, files, whisper, tts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 挂载静态页面
PAGES_DIR = os.path.join(os.path.dirname(__file__), "pages")
app.add_middleware(
    NextJSRouteMiddleware,
    static_dir=PAGES_DIR,
    skip_prefixes=["/api", "/auth", "/ws", "/upload", "/files", "/docs", "/redoc", "/openapi.json"],
)

# 页面认证守卫（在 NextJSRouteMiddleware 之外，拦截未登录的页面请求）
app.add_middleware(AuthGuardMiddleware)

# 可选：允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    try:
        create_db_and_tables()
    except Exception:
        logger.exception("数据库初始化失败")

# 注册路由
app.include_router(auth.router)
app.include_router(files.router)
app.include_router(whisper.router)
app.include_router(tts.router)

# 挂载静态页面目录（放在所有路由之后，作为 fallback）
try:
    app.mount("/", StaticFilesCache(directory=PAGES_DIR, html=True), name="pages")
except Exception as e:
    logging.error(f"静态文件目录挂载失败：{e}")

# uv run uvicorn main:app 开发模式
# uv run main.py --workers 2 部署模式
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    args = parser.parse_args()
    workers = args.workers
    port = args.port
    host = args.host

    def filter_logs(record):
        if record.name.startswith("sqlalchemy"):
            if record.levelno < logging.ERROR:
                return True
        return False

    run_server(
        "main:app",
        host=host,
        port=port,
        workers=workers,
        log_file="logs/app.log",
        filter_callbacks=[filter_logs]
    )
