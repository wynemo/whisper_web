调用whsiper转换录音为字幕的 用一台mac mini弄了一个web服务

```bash
uv pip install fastapi uvicorn[standard] python-multipart

brew install uv ffmpeg #install uv and ffmpeg

uv venv --python 3.9 # install python, but virtual env         （它会用系统的3.9）

uv pip install git+https://github.com/openai/whisper.git # install whisper

uv run whisper /Users/tommygreen/Movies/mihomo\ party/mihomo\ party.mov --model turbo \
      --language Chinese --task transcribe --output_format srt --initial_prompt "以下是普通话的句子。"
```


# todo

1. 整理笔记
2. 转换为srt
3. 对接youtube下载音频
