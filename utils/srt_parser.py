import re
from dataclasses import dataclass


@dataclass
class Subtitle:
    index: int
    start_time_ms: int
    end_time_ms: int
    text: str


def parse_time(time_str: str) -> int:
    """将 SRT 时间格式转换为毫秒"""
    # 格式: 00:00:00,000
    match = re.match(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})", time_str)
    if not match:
        raise ValueError(f"Invalid time format: {time_str}")
    hours, minutes, seconds, milliseconds = map(int, match.groups())
    return hours * 3600000 + minutes * 60000 + seconds * 1000 + milliseconds


def parse_srt(content: str) -> list[Subtitle]:
    """解析 SRT 字幕内容"""
    subtitles = []
    # 按空行分割字幕块
    blocks = re.split(r"\n\s*\n", content.strip())

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        # 第一行: 序号
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue

        # 第二行: 时间轴
        time_match = re.match(
            r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})",
            lines[1].strip()
        )
        if not time_match:
            continue

        start_time_ms = parse_time(time_match.group(1))
        end_time_ms = parse_time(time_match.group(2))

        # 剩余行: 字幕文本
        text = "\n".join(lines[2:]).strip()

        subtitles.append(Subtitle(
            index=index,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            text=text,
        ))

    return subtitles


def parse_srt_file(file_path: str) -> list[Subtitle]:
    """从文件解析 SRT 字幕"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return parse_srt(content)
