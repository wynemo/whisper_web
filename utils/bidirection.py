#!/usr/bin/env python3
import asyncio
import copy
import io
import json
import logging
import uuid

import websockets
from mutagen.mp3 import MP3

from utils.volcengine_protocols import (
    EventType,
    MsgType,
    finish_connection,
    finish_session,
    receive_message,
    start_connection,
    start_session,
    task_request,
    wait_for_event,
)

logger = logging.getLogger(__name__)


def get_resource_id(voice: str) -> str:
    if voice.startswith("S_"):
        return "volc.megatts.default"
    return "volc.service_type.10029"


async def text_to_speech(
    text: str,
    appid: str,
    access_token: str,
    voice_type: str = "zh_male_jieshuonansheng_mars_bigtts",
    resource_id: str = "seed-tts-1.0",
    encoding: str = "mp3",
) -> tuple[bytes, float, list]:
    """
    使用双向流 WebSocket 将文本转换为语音

    Args:
        text: 需要转换的文本内容
        appid: 火山引擎 appid
        access_token: 火山引擎 access token
        voice_type: 音色类型
        resource_id: 资源 ID
        encoding: 音频编码格式

    Returns:
        tuple[bytes, float, list]: 音频字节数据、音频时长（秒）和语音字幕
    """
    endpoint = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"

    headers = {
        "X-Api-App-Key": appid,
        "X-Api-Access-Key": access_token,
        "X-Api-Resource-Id": resource_id,
        "X-Api-Connect-Id": str(uuid.uuid4()),
    }

    websocket = await websockets.connect(
        endpoint, additional_headers=headers, max_size=10 * 1024 * 1024
    )

    try:
        # Start connection
        await start_connection(websocket)
        await wait_for_event(
            websocket, MsgType.FullServerResponse, EventType.ConnectionStarted
        )

        # Build base request
        base_request = {
            "user": {"uid": str(uuid.uuid4())},
            "namespace": "BidirectionalTTS",
            "req_params": {
                "speaker": voice_type,
                "audio_params": {
                    "format": encoding,
                    "sample_rate": 24000,
                    "enable_timestamp": True,
                },
                "additions": json.dumps({
                    "disable_markdown_filter": True,
                    "enable_latex_tn": True,
                    "disable_emoji_filter": False,
                }),
            },
        }

        # Start session
        start_session_request = copy.deepcopy(base_request)
        start_session_request["event"] = EventType.StartSession
        session_id = str(uuid.uuid4())
        await start_session(
            websocket, json.dumps(start_session_request).encode(), session_id
        )
        await wait_for_event(
            websocket, MsgType.FullServerResponse, EventType.SessionStarted
        )

        # Send text
        async def send_text():
            synthesis_request = copy.deepcopy(base_request)
            synthesis_request["event"] = EventType.TaskRequest
            synthesis_request["req_params"]["text"] = text
            await task_request(
                websocket, json.dumps(synthesis_request).encode(), session_id
            )
            await finish_session(websocket, session_id)

        send_task = asyncio.create_task(send_text())

        # Receive audio data
        audio_data = bytearray()
        words = []
        while True:
            msg = await receive_message(websocket)
            if msg.type == MsgType.FullServerResponse:
                if msg.event == EventType.SessionFinished:
                    break
                # 解析时间戳数据 (TTSSentenceEnd 事件包含 words)
                try:
                    payload = json.loads(msg.payload.decode("utf-8"))
                    if "words" in payload and payload["words"]:
                        for item in payload["words"]:
                            words.append({
                                "word": item.get("word", ""),
                                "start_time": item.get("startTime", 0),
                                "end_time": item.get("endTime", 0),
                                "unit_type": "text",
                            })
                except Exception as e:
                    logger.warning(f"解析时间戳失败: {e}")
            elif msg.type == MsgType.AudioOnlyServer:
                audio_data.extend(msg.payload)
            else:
                raise RuntimeError(f"TTS conversion failed: {msg}")

        await send_task

        if not audio_data:
            raise RuntimeError("No audio data received")

        # Calculate duration
        duration = 0.0
        try:
            audio_file = io.BytesIO(bytes(audio_data))
            if encoding == "mp3":
                audio = MP3(audio_file)
                duration = audio.info.length
        except Exception as e:
            logger.warning(f"音频时长计算失败: {str(e)}，使用默认值0.0")

        logger.info(
            f"语音生成成功, 大小: {len(audio_data)}字节, 时长: {duration}秒, words: {len(words)}个"
        )
        return bytes(audio_data), duration, words

    except Exception as e:
        error_msg = f"双向流语音合成失败: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    finally:
        await finish_connection(websocket)
        try:
            await wait_for_event(
                websocket, MsgType.FullServerResponse, EventType.ConnectionFinished
            )
        except Exception:
            pass
        await websocket.close()
