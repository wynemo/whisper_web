import base64
import io
import json
import logging
import os
import re
from typing import List, Optional
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from pydub import AudioSegment

from auth import get_current_user
from config import settings
from models import User
from utils.srt_parser import parse_srt
from utils.bidirection import text_to_speech
from utils.audio_aligner import (
    align_audio_to_subtitle_with_retry,
    merge_aligned_audios,
    export_audio_bytes,
    adjust_audio_duration,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["tts"])

UPLOAD_DIR = "upload"


def split_sentences(text: str) -> list[str]:
    """将文本按标点符号分割成句子"""
    sentences = re.split(r'(?<=[。？?！!；;\n])', text)
    return [s.strip() for s in sentences if s.strip()]


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


class CorrectSubtitlesRequest(BaseModel):
    srt_content: str
    reference_text: str
    api_base_url: str = ""
    api_key: str = ""
    model: str = ""


def _format_timestamp_srt(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


@router.post("/srt-to-speech/")
async def srt_to_speech_api(file: UploadFile = File(...), _user: User = Depends(get_current_user)):
    """将 SRT 字幕转换为语音"""
    srt_path = os.path.join(UPLOAD_DIR, file.filename)
    content = await file.read()
    with open(srt_path, "wb") as f:
        f.write(content)

    srt_content = content.decode("utf-8")
    subtitles = parse_srt(srt_content)

    if not subtitles:
        return {"error": "无法解析字幕文件"}

    logger.info(f"解析到 {len(subtitles)} 条字幕")

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

    merged_audio = merge_aligned_audios(aligned_segments)
    audio_bytes = export_audio_bytes(merged_audio, format="mp3")

    output_filename = os.path.splitext(file.filename)[0] + ".mp3"
    encoded_filename = quote(output_filename)
    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}"},
    )


@router.post("/text-to-speech/")
async def text_to_speech_api(
    text: str = Form(...),
    duration_seconds: Optional[float] = Form(None),
    _user: User = Depends(get_current_user),
):
    """将纯文本转换为语音"""
    sentences = split_sentences(text)
    if not sentences:
        return {"error": "无法分割文本，请确保文本包含有效内容"}

    logger.info(f"分割成 {len(sentences)} 个句子")

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

    total_duration_ms = sum(seg["duration_ms"] for seg in segments)

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

            total_duration_ms = sum(seg["duration_ms"] for seg in segments)

    result_segments = []
    for seg in segments:
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


@router.post("/merge-audio/")
async def merge_audio(request: MergeRequest, _user: User = Depends(get_current_user)):
    """合并多个音频片段，按时间轴位置排列"""
    if not request.segments:
        return {"error": "没有音频片段"}

    sorted_segments = sorted(request.segments, key=lambda x: x.start_offset_ms)

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

    total_duration_ms = max(
        seg["start_offset_ms"] + seg["actual_duration_ms"]
        for seg in decoded_segments
    )

    merged_audio = AudioSegment.silent(duration=total_duration_ms)

    for seg in decoded_segments:
        merged_audio = merged_audio.overlay(
            seg["audio"], position=seg["start_offset_ms"]
        )

    audio_bytes = export_audio_bytes(merged_audio, format="mp3")

    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={"Content-Disposition": "attachment; filename=merged_audio.mp3"},
    )


@router.post("/generate-srt/")
async def generate_srt(request: GenerateSrtRequest, _user: User = Depends(get_current_user)):
    """根据时间轴片段生成 SRT 字幕文件"""
    if not request.segments:
        return {"error": "没有片段数据"}

    sorted_segments = sorted(request.segments, key=lambda x: x.start_offset_ms)

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


@router.post("/correct-subtitles/")
async def correct_subtitles(request: CorrectSubtitlesRequest, _user: User = Depends(get_current_user)):
    """使用 LLM 纠正 ASR 生成的字幕"""
    api_base_url = request.api_base_url or settings.LLM_API_BASE_URL
    api_key = request.api_key or settings.LLM_API_KEY
    model = request.model or settings.LLM_MODEL

    if not api_base_url or not api_key or not model:
        return {"error": "请配置 LLM API 地址、API Key 和模型名称"}

    system_prompt = (
        "你是一个专业的字幕校对助手。你的任务是根据参考文本纠正 ASR（语音识别）生成的 SRT 字幕。\n"
        "规则：\n"
        "1. 保持 SRT 格式不变（序号、时间轴、空行分隔）\n"
        "2. 只修正文字内容，不要改动时间轴\n"
        "3. 根据参考文本修正错别字、漏字、多字等识别错误\n"
        "4. 如果某段字幕在参考文本中找不到对应内容，保持原样\n"
        "5. 在不改变原有意思的情况下，去掉口水话，尽量简短\n"
        "6. 只输出纠正后的完整 SRT 内容，不要输出任何解释说明"
    )

    user_prompt = (
        f"以下是 ASR 生成的 SRT 字幕：\n\n{request.srt_content}\n\n"
        f"以下是参考文本：\n\n{request.reference_text}\n\n"
        "请根据参考文本纠正字幕中的识别错误，只输出纠正后的完整 SRT 内容。"
    )

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

        if corrected_srt.startswith("```"):
            lines = corrected_srt.split("\n")
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
