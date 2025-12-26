from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from subprocess import Popen, PIPE, STDOUT
import os
import asyncio
import logging
import uvicorn

from config import (
    DOUBAO_APPID,
    DOUBAO_ACCESS_TOKEN,
    DOUBAO_RESOURCE_ID,
    DOUBAO_DEFAULT_VOICE_TYPE,
)
from utils.srt_parser import parse_srt
from utils.bidirection import text_to_speech
from utils.audio_aligner import (
    align_audio_to_subtitle,
    merge_aligned_audios,
    export_audio_bytes,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

@app.post("/srt-to-speech/")
async def srt_to_speech(file: UploadFile = File(...)):
    """
    将 SRT 字幕转换为语音

    1. 解析 SRT 字幕
    2. 逐条调用火山 TTS 生成语音
    3. 使用 whisperx 对齐音频到字幕时间轴
    4. 合并所有音频段
    5. 返回 mp3 文件
    """
    # 保存上传的 SRT 文件
    srt_path = os.path.join(UPLOAD_DIR, file.filename)
    content = await file.read()
    with open(srt_path, "wb") as f:
        f.write(content)

    # 解析字幕
    srt_content = content.decode("utf-8")
    subtitles = parse_srt(srt_content)

    if not subtitles:
        return {"error": "无法解析字幕文件"}

    logger.info(f"解析到 {len(subtitles)} 条字幕")

    # 逐条生成语音并对齐
    aligned_segments = []
    for i, subtitle in enumerate(subtitles):
        logger.info(f"处理字幕 {i+1}/{len(subtitles)}: {subtitle.text[:20]}...")

        try:
            # 调用 TTS 生成语音
            audio_data, _, _ = await text_to_speech(
                text=subtitle.text,
                appid=DOUBAO_APPID,
                access_token=DOUBAO_ACCESS_TOKEN,
                voice_type=DOUBAO_DEFAULT_VOICE_TYPE,
                resource_id=DOUBAO_RESOURCE_ID,
            )

            # 对齐到字幕时间轴
            aligned_audio = align_audio_to_subtitle(audio_data, subtitle)

            aligned_segments.append((subtitle, aligned_audio))

        except Exception as e:
            logger.error(f"处理字幕 {i+1} 失败: {e}")
            continue

    if not aligned_segments:
        return {"error": "所有字幕处理失败"}

    # 合并所有音频段
    merged_audio = merge_aligned_audios(aligned_segments)
    audio_bytes = export_audio_bytes(merged_audio, format="mp3")

    # 返回音频文件
    output_filename = os.path.splitext(file.filename)[0] + ".mp3"
    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={"Content-Disposition": f"attachment; filename={output_filename}"},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
