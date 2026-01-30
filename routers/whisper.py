import os
import asyncio
import logging
import platform
from subprocess import Popen, PIPE, STDOUT

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["whisper"])

UPLOAD_DIR = "upload"

# faster-whisper 条件导入 (仅 Linux)
_faster_whisper_available = False
_whisper_model = None

if settings.WHISPER_ENGINE == "faster" and platform.system() == "Linux":
    try:
        from faster_whisper import WhisperModel
        _faster_whisper_available = True
        logger.info("faster-whisper 已加载")
    except ImportError:
        logger.warning(
            "faster-whisper 未安装，将回退到 openai-whisper。"
            "请运行: uv pip install faster-whisper"
        )


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

    segments, info = model.transcribe(
        filename,
        language="zh",
        task="transcribe",
        beam_size=5,
        word_timestamps=True,
        initial_prompt="以下是普通话的句子。",
    )

    await websocket.send_text(f"检测到语言: {info.language} (概率: {info.language_probability:.2f})")

    srt_lines = []
    segment_count = 0

    for segment in segments:
        segment_count += 1
        start_ts = _format_timestamp_srt(segment.start)
        end_ts = _format_timestamp_srt(segment.end)
        text = segment.text.strip()

        srt_entry = f"{segment_count}\n{start_ts} --> {end_ts}\n{text}\n"
        srt_lines.append(srt_entry)

        await websocket.send_text(f"[{start_ts} --> {end_ts}] {text}")
        await asyncio.sleep(0.01)

    srt_filename = os.path.splitext(filename)[0] + ".srt"
    with open(srt_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))

    await websocket.send_text(f"\n转录完成! 共 {segment_count} 个片段")
    await websocket.send_text(f"SRT 文件已保存: {os.path.basename(srt_filename)}")


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = Query(...)):
    from jose import JWTError, jwt as jose_jwt
    from auth import ALGORITHM
    try:
        payload = jose_jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            await websocket.close(code=4001, reason="无效的认证凭据")
            return
    except JWTError:
        await websocket.close(code=4001, reason="无效的认证凭据")
        return

    await websocket.accept()
    process = None

    try:
        data = await websocket.receive_text()
        filename = os.path.join(UPLOAD_DIR, data)
        if not os.path.exists(filename):
            await websocket.send_text(f"Error: File '{data}' does not exist.")
            return

        if _faster_whisper_available:
            await _transcribe_with_faster_whisper(filename, websocket)
        else:
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
