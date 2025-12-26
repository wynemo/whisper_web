调用whsiper转换录音为字幕的 用一台mac mini弄了一个web服务

```bash

brew install uv ffmpeg #install uv and ffmpeg

uv venv --python 3.9 # install python, but virtual env         （它会用系统的3.9）
uv pip install git+https://github.com/openai/whisper.git # install whisper
uv pip install fastapi uvicorn[standard] python-multipart
uv run main.py

```


# todo

1. 用大模型整理笔记
2. 转换为srt
3. 对接youtube下载音频




### 字幕生成语音

添加一个路由

1. 上传 srt 字幕
2. 调用字节跳动的语音合成服务
3. 调用 whisperx 根据字幕对齐语音
4. 合并生成的音频
