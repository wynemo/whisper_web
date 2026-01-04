import argparse
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi_toolbox import run_server
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from subprocess import Popen, PIPE, STDOUT
import os
import asyncio
import logging
import platform

from config import settings

# 自动设置 NVIDIA 库路径 (解决 cuDNN 加载问题)
def _setup_nvidia_lib_path():
    """设置 LD_LIBRARY_PATH 以包含 nvidia 库路径"""
    try:
        import nvidia.cublas.lib
        import nvidia.cudnn.lib
        cublas_path = nvidia.cublas.lib.__path__[0]
        cudnn_path = nvidia.cudnn.lib.__path__[0]
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if cudnn_path not in ld_path:
            os.environ["LD_LIBRARY_PATH"] = f"{cudnn_path}:{cublas_path}:{ld_path}"
            logging.getLogger(__name__).info(f"已添加 NVIDIA 库路径: {cudnn_path}, {cublas_path}")
    except ImportError:
        pass  # nvidia 库未安装，跳过

# faster-whisper 条件导入 (仅 Linux)
_faster_whisper_available = False
_whisper_model = None

if settings.WHISPER_ENGINE == "faster" and platform.system() == "Linux":
    _setup_nvidia_lib_path()  # 先设置库路径
    try:
        from faster_whisper import WhisperModel
        _faster_whisper_available = True
        logger_init = logging.getLogger(__name__)
        logger_init.info("faster-whisper 已加载")
    except ImportError:
        logger_init = logging.getLogger(__name__)
        logger_init.warning(
            "faster-whisper 未安装，将回退到 openai-whisper。"
            "请运行: uv pip install faster-whisper"
        )
from utils.srt_parser import parse_srt
from utils.bidirection import text_to_speech
from utils.audio_aligner import (
    align_audio_to_subtitle_with_retry,
    merge_aligned_audios,
    export_audio_bytes,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _format_timestamp_srt(seconds: float) -> str:
    """将秒转换为 SRT 时间戳格式 (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _get_faster_whisper_model():
    """获取或初始化 faster-whisper 模型 (懒加载)"""
    global _whisper_model
    if _whisper_model is None and _faster_whisper_available:
        device = "cuda" if settings.USE_CUDA else "cpu"
        compute_type = settings.FASTER_WHISPER_COMPUTE_TYPE
        # CPU 模式下只能使用 int8 或 float32
        if device == "cpu" and compute_type == "float16":
            compute_type = "int8"
        logger.info(
            f"初始化 faster-whisper 模型: {settings.WHISPER_MODEL}, "
            f"device={device}, compute_type={compute_type}"
        )
        _whisper_model = WhisperModel(
            settings.WHISPER_MODEL,
            device=device,
            compute_type=compute_type,
        )
        logger.info("faster-whisper 模型加载完成")
    return _whisper_model


async def _transcribe_with_faster_whisper(filename: str, websocket: WebSocket):
    """使用 faster-whisper 进行转录"""
    model = _get_faster_whisper_model()
    if model is None:
        await websocket.send_text("Error: faster-whisper 模型未加载")
        return

    await websocket.send_text("使用 faster-whisper 引擎进行转录...")

    # 执行转录
    segments, info = model.transcribe(
        filename,
        language="zh",
        task="transcribe",
        beam_size=5,
        word_timestamps=True,
        initial_prompt="以下是普通话的句子。",
    )

    await websocket.send_text(f"检测到语言: {info.language} (概率: {info.language_probability:.2f})")

    # 生成 SRT 内容并流式输出
    srt_lines = []
    segment_count = 0

    for segment in segments:
        segment_count += 1
        start_ts = _format_timestamp_srt(segment.start)
        end_ts = _format_timestamp_srt(segment.end)
        text = segment.text.strip()

        # SRT 格式
        srt_entry = f"{segment_count}\n{start_ts} --> {end_ts}\n{text}\n"
        srt_lines.append(srt_entry)

        # 发送进度
        await websocket.send_text(f"[{start_ts} --> {end_ts}] {text}")
        await asyncio.sleep(0.01)  # 让出控制权

    # 保存 SRT 文件
    srt_filename = os.path.splitext(filename)[0] + ".srt"
    with open(srt_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))

    await websocket.send_text(f"\n转录完成! 共 {segment_count} 个片段")
    await websocket.send_text(f"SRT 文件已保存: {os.path.basename(srt_filename)}")


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


@app.get("/srt-to-speech")
async def serve_srt_to_speech():
    """提供 SRT 转语音页面"""
    return FileResponse("srt_to_speech.html")

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

        # 根据配置选择转录引擎
        if _faster_whisper_available:
            # 使用 faster-whisper (仅 Linux)
            await _transcribe_with_faster_whisper(filename, websocket)
        else:
            # 使用 openai-whisper 命令行
            await websocket.send_text("使用 openai-whisper 引擎进行转录...")
            cmd = [
                "uv", "run", "whisper", filename,
                "--model", settings.WHISPER_MODEL,
                "--language", "Chinese",
                "--task", "transcribe",
                "--max_line_count", "1",
                "--max_words_per_line", "24",
                "--word_timestamps", "True",
                "--output_format", "srt",
                "--initial_prompt", "以下是普通话的句子。",
                "--device", "cuda" if settings.USE_CUDA else "cpu"
            ]
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
            tts_kwargs = {
                "text": subtitle.text,
                "appid": settings.DOUBAO_APPID,
                "access_token": settings.DOUBAO_ACCESS_TOKEN,
                "voice_type": settings.DOUBAO_DEFAULT_VOICE_TYPE,
                "resource_id": settings.DOUBAO_RESOURCE_ID,
            }
            result = await align_audio_to_subtitle_with_retry(
                subtitle, text_to_speech, tts_kwargs
            )

            aligned_segments.append((subtitle, result["audio_segment"]))

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
    # URL encode the filename for non-ASCII characters (RFC 5987)
    from urllib.parse import quote
    encoded_filename = quote(output_filename)
    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}"},
    )


# uv run uvicorn main:app 开发模式
# uv run main.py --workers 2 部署模式
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加 workers 参数，默认为 1 个 worker
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    # 添加 port 参数，默认端口为 8000
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    # 添加 host 参数，默认 host 为 0.0.0.0
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    # 解析命令行参数
    args = parser.parse_args()
    workers = args.workers
    port = args.port
    host = args.host


    def filter_logs(record):
        # 过滤 SQLAlchemy 低级别日志
        if record.name.startswith("sqlalchemy"):
            if record.levelno < logging.ERROR:
                return True
        return False

    run_server(
        "main:app",
        host=host,
        port=port,
        workers=workers,
        log_file="logs/app.log", # 日志轮转
        filter_callbacks=[filter_logs]
    )

