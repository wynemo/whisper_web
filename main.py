from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
# from fastapi.staticfiles import StaticFiles
from subprocess import Popen, PIPE, STDOUT
import os
import asyncio
import uvicorn

app = FastAPI()

# 可选：允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 更严格时可改为 ["http://localhost:8000"] 或其他特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建保存文件的目录
UPLOAD_DIR = "upload"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 挂载静态文件目录
# app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_index():
    """提供 index.html 文件"""
    return FileResponse("index.html")

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """保存上传的文件到指定目录"""
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    return {"filename": file.filename, "filepath": file_path}

@app.get("/files/")
async def list_files():
    """获取 upload 目录下的文件列表"""
    files = os.listdir(UPLOAD_DIR)
    return {"files": files}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    process = None

    try:
        # 接收前端发送的文件名
        data = await websocket.receive_text()
        filename = os.path.join(UPLOAD_DIR, data)
        if not os.path.exists(filename):
            await websocket.send_text(f"Error: File '{data}' does not exist.")
            return

        cmd = ["uv", "run", "whisper", filename, "--model", "turbo", "--language", "Chinese",
            "--task", "transcribe",
            "--max_line_count", "1", "--max_words_per_line", "24", "--word_timestamps", "True",
            "--output_format", "srt", "--initial_prompt", "以下是普通话的句子。"]
        # 复制当前环境变量并添加 PYTHONUNBUFFERED=1
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        process = Popen(cmd, stdout=PIPE, stderr=STDOUT, text=True, env=env)

        while True:
            line = process.stdout.readline()
            if line == "" and process.poll() is not None:
                break
            if line:
                await websocket.send_text(line.strip())
            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    finally:
        if process:
            process.terminate()
            process.wait()
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
