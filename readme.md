```bash
uv pip install fastapi uvicorn[standard] python-multipart

brew install uv ffmpeg #install uv and ffmpeg

uv venv --python 3.9 # install python, but virtual env         （它会用系统的3.9）

uv pip install git+https://github.com/openai/whisper.git # install whisper

uv run whisper /Users/tommygreen/Movies/mihomo\ party/mihomo\ party.mov --model turbo \
      --language Chinese --task transcribe --output_format srt --initial_prompt "以下是普通话的句子。"
```
