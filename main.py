import argparse
import base64
import io
import json
import re
from typing import List, Optional

import httpx
from pydantic import BaseModel

from pydub import AudioSegment

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi_toolbox import run_server
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from subprocess import Popen, PIPE, STDOUT
import os
import asyncio
import logging
import platform

from config import settings

# faster-whisper 条件导入 (仅 Linux)
_faster_whisper_available = False
_whisper_model = None

if settings.WHISPER_ENGINE == "faster" and platform.system() == "Linux":
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
    adjust_audio_duration,
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


def split_sentences(text: str) -> list[str]:
    """将文本按标点符号分割成句子"""
    # 按中英文标点分句，保留标点符号
    sentences = re.split(r'(?<=[。？?！!；;\n])', text)
    # 过滤空白句子
    return [s.strip() for s in sentences if s.strip()]

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


@app.get("/tts-timeline")
async def serve_tts_timeline():
    """提供 TTS 时间轴编辑页面"""
    return FileResponse("tts_timeline.html")

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


@app.post("/text-to-speech/")
async def text_to_speech_api(
    text: str = Form(...),
    duration_seconds: Optional[float] = Form(None),
):
    """
    将纯文本转换为语音

    1. 自动分句
    2. 逐句生成语音
    3. 如果指定了目标时长，按比例调整每句音频时长
    4. 返回 JSON，包含每句的文本和 Base64 编码的音频
    """
    # 分句处理
    sentences = split_sentences(text)
    if not sentences:
        return {"error": "无法分割文本，请确保文本包含有效内容"}

    logger.info(f"分割成 {len(sentences)} 个句子")

    # 逐句生成语音
    segments = []
    for i, sentence in enumerate(sentences):
        logger.info(f"处理句子 {i+1}/{len(sentences)}: {sentence[:20]}...")

        try:
            audio_data, _, _ = await text_to_speech(
                text=sentence,
                appid=settings.DOUBAO_APPID,
                access_token=settings.DOUBAO_ACCESS_TOKEN,
                voice_type=settings.DOUBAO_DEFAULT_VOICE_TYPE,
                resource_id=settings.DOUBAO_RESOURCE_ID,
            )
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))

            segments.append({
                "text": sentence,
                "audio_segment": audio_segment,
                "duration_ms": len(audio_segment),
            })

        except Exception as e:
            logger.error(f"处理句子 {i+1} 失败: {e}")
            continue

    if not segments:
        return {"error": "所有句子处理失败"}

    # 计算总时长
    total_duration_ms = sum(seg["duration_ms"] for seg in segments)

    # 如果指定了目标时长，按比例调整每句音频
    if duration_seconds is not None:
        target_total_ms = int(duration_seconds * 1000)
        if target_total_ms > 0 and total_duration_ms > 0:
            ratio = target_total_ms / total_duration_ms

            for seg in segments:
                target_duration_ms = int(seg["duration_ms"] * ratio)
                adjusted_audio, _ = adjust_audio_duration(
                    seg["audio_segment"], target_duration_ms
                )
                seg["audio_segment"] = adjusted_audio
                seg["duration_ms"] = len(adjusted_audio)

            # 重新计算总时长
            total_duration_ms = sum(seg["duration_ms"] for seg in segments)

    # 构建返回结果
    result_segments = []
    for seg in segments:
        # 导出为 MP3 并编码为 Base64
        audio_bytes = export_audio_bytes(seg["audio_segment"], format="mp3")
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        result_segments.append({
            "text": seg["text"],
            "audio": audio_base64,
            "duration_ms": seg["duration_ms"],
        })

    return {
        "segments": result_segments,
        "total_duration_ms": total_duration_ms,
    }


class AudioSegmentInput(BaseModel):
    audio: str  # Base64 encoded
    start_offset_ms: int
    duration_ms: int


class MergeRequest(BaseModel):
    segments: List[AudioSegmentInput]


class SrtSegmentInput(BaseModel):
    text: str
    start_offset_ms: int
    duration_ms: int


class GenerateSrtRequest(BaseModel):
    segments: List[SrtSegmentInput]


@app.post("/merge-audio/")
async def merge_audio(request: MergeRequest):
    """
    合并多个音频片段，按时间轴位置排列

    1. 接收 Base64 编码的音频片段和时间偏移
    2. 按时间位置拼接音频（保留空白间隙）
    3. 返回合并后的 MP3 文件
    """
    if not request.segments:
        return {"error": "没有音频片段"}

    # 按 start_offset_ms 排序
    sorted_segments = sorted(request.segments, key=lambda x: x.start_offset_ms)

    # 先解码所有音频，计算实际总时长
    decoded_segments = []
    for seg in sorted_segments:
        try:
            audio_bytes = base64.b64decode(seg.audio)
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            decoded_segments.append({
                "audio": audio_segment,
                "start_offset_ms": seg.start_offset_ms,
                "actual_duration_ms": len(audio_segment),
            })
        except Exception as e:
            logger.error(f"解码音频片段失败: {e}")
            continue

    if not decoded_segments:
        return {"error": "所有音频片段解码失败"}

    # 计算总时长（使用实际音频长度）
    total_duration_ms = max(
        seg["start_offset_ms"] + seg["actual_duration_ms"]
        for seg in decoded_segments
    )

    # 创建静音的基底音频
    merged_audio = AudioSegment.silent(duration=total_duration_ms)

    # 逐个叠加音频
    for seg in decoded_segments:
        merged_audio = merged_audio.overlay(
            seg["audio"], position=seg["start_offset_ms"]
        )

    # 导出为 MP3
    audio_bytes = export_audio_bytes(merged_audio, format="mp3")

    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={"Content-Disposition": "attachment; filename=merged_audio.mp3"},
    )


@app.post("/generate-srt/")
async def generate_srt(request: GenerateSrtRequest):
    """
    根据时间轴片段生成 SRT 字幕文件

    1. 接收片段列表（文本、起始时间、时长）
    2. 按起始时间排序
    3. 生成 SRT 格式字幕
    4. 返回 SRT 文件
    """
    if not request.segments:
        return {"error": "没有片段数据"}

    # 按 start_offset_ms 排序
    sorted_segments = sorted(request.segments, key=lambda x: x.start_offset_ms)

    # 生成 SRT 内容
    srt_lines = []
    for i, seg in enumerate(sorted_segments, start=1):
        start_seconds = seg.start_offset_ms / 1000.0
        end_seconds = (seg.start_offset_ms + seg.duration_ms) / 1000.0

        start_ts = _format_timestamp_srt(start_seconds)
        end_ts = _format_timestamp_srt(end_seconds)

        srt_entry = f"{i}\n{start_ts} --> {end_ts}\n{seg.text}\n"
        srt_lines.append(srt_entry)

    srt_content = "\n".join(srt_lines)

    return Response(
        content=srt_content.encode("utf-8"),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=timeline.srt"},
    )


class CorrectSubtitlesRequest(BaseModel):
    srt_content: str
    reference_text: str
    api_base_url: str = ""
    api_key: str = ""
    model: str = ""


@app.post("/correct-subtitles/")
async def correct_subtitles(request: CorrectSubtitlesRequest):
    """
    使用 LLM 纠正 ASR 生成的字幕

    1. 接收 SRT 字幕内容和参考文本
    2. 发送给 OpenAI 兼容的大模型 API
    3. 返回纠正后的 SRT 字幕
    """
    # 优先使用请求中的配置，否则回退到环境变量配置
    api_base_url = request.api_base_url or settings.LLM_API_BASE_URL
    api_key = request.api_key or settings.LLM_API_KEY
    model = request.model or settings.LLM_MODEL

    if not api_base_url or not api_key or not model:
        return {"error": "请配置 LLM API 地址、API Key 和模型名称"}

    # 构建 prompt
    system_prompt = (
        "你是一个专业的字幕校对助手。你的任务是根据参考文本纠正 ASR（语音识别）生成的 SRT 字幕。\n"
        "规则：\n"
        "1. 保持 SRT 格式不变（序号、时间轴、空行分隔）\n"
        "2. 只修正文字内容，不要改动时间轴\n"
        "3. 根据参考文本修正错别字、漏字、多字等识别错误\n"
        "4. 如果某段字幕在参考文本中找不到对应内容，保持原样\n"
        "5. 只输出纠正后的完整 SRT 内容，不要输出任何解释说明"
    )

    user_prompt = (
        f"以下是 ASR 生成的 SRT 字幕：\n\n{request.srt_content}\n\n"
        f"以下是参考文本：\n\n{request.reference_text}\n\n"
        "请根据参考文本纠正字幕中的识别错误，只输出纠正后的完整 SRT 内容。"
    )

    # 调用 OpenAI 兼容 API
    # 确保 base_url 以 /v1 结尾
    base_url = api_base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"
    url = f"{base_url}/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()

        result = response.json()
        corrected_srt = result["choices"][0]["message"]["content"].strip()

        # 去除可能的 markdown 代码块标记
        if corrected_srt.startswith("```"):
            lines = corrected_srt.split("\n")
            # 去掉首行 ``` 和末行 ```
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            corrected_srt = "\n".join(lines).strip()

        return {"corrected_srt": corrected_srt}

    except httpx.HTTPStatusError as e:
        error_detail = e.response.text if e.response else str(e)
        logger.error(f"LLM API 请求失败: {e.response.status_code} - {error_detail}")
        return {"error": f"LLM API 请求失败: {e.response.status_code} - {error_detail}"}
    except httpx.RequestError as e:
        logger.error(f"LLM API 连接失败: {e}")
        return {"error": f"LLM API 连接失败: {str(e)}"}
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        logger.error(f"LLM API 响应解析失败: {e}")
        return {"error": f"LLM API 响应解析失败: {str(e)}"}


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

