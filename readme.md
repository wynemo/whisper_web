调用whsiper转换录音为字幕的 用一台mac mini弄了一个web服务

```bash

brew install uv ffmpeg #install uv and ffmpeg, use the newest uv

uv sync
uv run --env-file settings.env main.py

```


# todo

1. 用大模型整理笔记
2. 转换为srt
3. 对接youtube下载音频




### 字幕生成语音

添加一个路由

1. 上传 srt 字幕
2. 调用字节跳动的语音合成服务
3. ~~调用 whisperx 根据字幕对齐语音~~
4. 合并生成的音频
