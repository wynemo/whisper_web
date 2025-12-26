import io
import os
from pydub import AudioSegment

from utils.srt_parser import Subtitle


def adjust_audio_duration(
    audio_segment: AudioSegment,
    target_duration_ms: int,
    words: list[dict] | None = None,
) -> tuple[AudioSegment, list[dict]]:
    """
    调整音频时长以匹配目标时长

    Args:
        audio_segment: 原始音频
        target_duration_ms: 目标时长（毫秒）
        words: word-level 时间戳列表

    Returns:
        调整后的音频和更新后的 words 时间戳
    """
    current_duration_ms = len(audio_segment)
    words = words or []

    if current_duration_ms == target_duration_ms:
        return audio_segment, words

    if current_duration_ms < target_duration_ms:
        # 音频比字幕短，在末尾添加静音
        silence_duration = target_duration_ms - current_duration_ms
        silence = AudioSegment.silent(duration=silence_duration)
        adjusted_audio = audio_segment + silence
        # words 时间戳不变
        return adjusted_audio, words
    else:
        # 音频比字幕长，通过变速拉伸
        # 计算速度比例（音频需要加快播放）
        speed_ratio = current_duration_ms / target_duration_ms

        # 使用 pydub 的 speedup 方法（通过改变 frame_rate 实现）
        # 先改变采样率，再转换回原采样率，实现变速不变调的效果
        original_frame_rate = audio_segment.frame_rate
        new_frame_rate = int(original_frame_rate * speed_ratio)

        # 改变采样率实现变速
        adjusted_audio = audio_segment._spawn(
            audio_segment.raw_data,
            overrides={"frame_rate": new_frame_rate}
        ).set_frame_rate(original_frame_rate)

        # 更新 words 时间戳
        adjusted_words = []
        for word in words:
            adjusted_word = word.copy()
            adjusted_word["start_time"] = word["start_time"] / speed_ratio
            adjusted_word["end_time"] = word["end_time"] / speed_ratio
            adjusted_words.append(adjusted_word)

        return adjusted_audio, adjusted_words


def align_audio_to_subtitle(
    audio_data: bytes,
    subtitle: Subtitle,
    words: list[dict] | None = None,
) -> dict:
    """
    将 TTS 生成的音频与字幕对齐，调整音频时长匹配字幕时长

    Args:
        audio_data: TTS 生成的音频数据 (bytes)
        subtitle: 字幕信息
        words: TTS 返回的 word-level 时间戳列表

    Returns:
        包含字幕、音频和时间戳的结果
    """
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))

    # 计算字幕时长
    subtitle_duration_ms = subtitle.end_time_ms - subtitle.start_time_ms

    # 调整音频时长以匹配字幕
    adjusted_audio, adjusted_words = adjust_audio_duration(
        audio_segment, subtitle_duration_ms, words
    )

    return {
        "subtitle": subtitle,
        "audio_segment": adjusted_audio,
        "words": adjusted_words,
    }


def merge_aligned_audios(aligned_segments: list[tuple[Subtitle, AudioSegment]]) -> AudioSegment:
    """
    合并所有音频段，按字幕时间戳顺序拼接

    Args:
        aligned_segments: 音频段列表，每项为 (字幕, 音频) 元组

    Returns:
        合并后的完整音频
    """
    if not aligned_segments:
        return AudioSegment.empty()

    # 按字幕开始时间排序
    aligned_segments = sorted(aligned_segments, key=lambda x: x[0].start_time_ms)

    merged = AudioSegment.empty()
    current_pos_ms = 0

    for subtitle, audio in aligned_segments:
        start_ms = subtitle.start_time_ms

        # 如果字幕开始时间晚于当前位置，插入静音填充
        if start_ms > current_pos_ms:
            silence = AudioSegment.silent(duration=start_ms - current_pos_ms)
            merged += silence
            current_pos_ms = start_ms

        # 拼接音频
        merged += audio
        current_pos_ms += len(audio)

    return merged


def export_audio_bytes(audio: AudioSegment, format: str = "mp3") -> bytes:
    """
    将 AudioSegment 导出为字节数据

    Args:
        audio: AudioSegment 对象
        format: 输出格式 (mp3, wav, ogg 等)

    Returns:
        音频字节数据
    """
    buffer = io.BytesIO()
    audio.export(buffer, format=format)
    buffer.seek(0)
    return buffer.read()


if __name__ == "__main__":
    import asyncio
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from config import (
        DOUBAO_APPID,
        DOUBAO_ACCESS_TOKEN,
        DOUBAO_RESOURCE_ID,
        DOUBAO_DEFAULT_VOICE_TYPE,
    )
    from utils.srt_parser import parse_srt
    from utils.bidirection import text_to_speech, get_resource_id

    TEST_SRT = """1
00:00:01,033 --> 00:00:06,066
今天这期节目的话是那个莱卡云赞助的

2
00:00:06,466 --> 00:00:13,100
莱卡云是一个做云服务器的厂商

3
00:00:13,633 --> 00:00:17,166
然后他的话就是说有国内的服务器

4
00:00:17,500 --> 00:00:20,200
也有那个海外的服务器

5
00:00:20,733 --> 00:00:21,966
看他们的主页

6
00:00:24,900 --> 00:00:28,400
有那个香港的CN2路线

7
00:00:28,933 --> 00:00:33,366
有那个美国的CN2路线
"""

    async def main():
        if not DOUBAO_APPID or not DOUBAO_ACCESS_TOKEN:
            print("请设置 DOUBAO_APPID 和 DOUBAO_ACCESS_TOKEN 环境变量")
            sys.exit(1)

        subtitles = parse_srt(TEST_SRT)
        print(f"解析 {len(subtitles)} 条字幕")

        resource_id = DOUBAO_RESOURCE_ID or get_resource_id(DOUBAO_DEFAULT_VOICE_TYPE)

        aligned_segments = []
        for subtitle in subtitles:
            audio_data, duration, words = await text_to_speech(
                text=subtitle.text,
                appid=DOUBAO_APPID,
                access_token=DOUBAO_ACCESS_TOKEN,
            )
            print(f"字幕{subtitle.index} TTS完成, 时长{duration}s")

            result = align_audio_to_subtitle(audio_data, subtitle, words=words)
            aligned_segments.append((subtitle, result["audio_segment"]))

        merged = merge_aligned_audios(aligned_segments)
        output_bytes = export_audio_bytes(merged, format="mp3")
        with open("test_output.mp3", "wb") as f:
            f.write(output_bytes)
        print(f"合并完成，输出: test_output.mp3, 时长: {len(merged)}ms")

    asyncio.run(main())
