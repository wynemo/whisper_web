# Whisper Web

基于 Whisper 的语音转字幕 Web 服务，支持字幕纠错和语音合成。

## 功能

### 语音识别 (ASR)

- 上传音频文件，使用 Whisper 自动转录为 SRT 字幕
- 支持两种引擎：OpenAI Whisper（跨平台）和 Faster-Whisper（仅 Linux，性能更优）
- 可选模型：tiny / base / small / medium / large-v2 / large-v3 / turbo
- WebSocket 实时推送转录进度

### 字幕纠错 (LLM)

- 使用大语言模型自动纠正 ASR 识别错误
- 支持上传参考文本进行对照纠错
- 兼容任意 OpenAI 格式的 API（GPT、Claude、本地模型等）
- 纠错结果预览，支持 SRT / 纯文本下载及剪贴板复制

### SRT 字幕转语音

- 上传 SRT 字幕文件，调用字节跳动火山引擎 TTS 生成语音
- 自动对齐语音时长与字幕时间轴（语速调整 + 静音填充）
- 合并输出完整 MP3 音频

### 时间轴编辑器

- 输入文本自动分句并生成语音片段
- 可视化时间轴，支持拖拽调整片段位置
- 编辑片段文本、起始时间、目标时长，可单独重新生成
- 片段逐个播放或顺序播放全部片段
- 合并所有片段为单个 MP3 文件，同时导出 SRT 字幕
- 本地视频预览：加载本地视频文件，同步时间轴与视频播放，实时显示字幕

## 安装与运行

```bash
# 安装系统依赖
brew install uv ffmpeg        # macOS
# apt install uv ffmpeg       # Linux

# 安装 Python 依赖
uv sync

# 运行（开发模式）
uv run main.py

# 运行（指定环境变量文件）
uv run --env-file settings.env main.py

# 运行（生产模式，多 worker）
uv run main.py --workers 2 --port 8000 --host 0.0.0.0
```

启动后访问 `http://localhost:8000`。

## 配置

在项目根目录创建 `settings.env` 文件：

```env
# Whisper 配置
USE_CUDA=False                                    # 是否启用 GPU 加速
WHISPER_ENGINE=openai                             # openai 或 faster（仅 Linux）
WHISPER_MODEL=turbo                               # 模型选择

# 火山引擎 TTS 配置
DOUBAO_APPID=                                     # 火山引擎 App ID
DOUBAO_ACCESS_TOKEN=                              # 火山引擎 Access Token
DOUBAO_RESOURCE_ID=volc.megatts.default           # TTS 资源 ID
DOUBAO_DEFAULT_VOICE_TYPE=zh_female_vv_uranus_bigtts  # 语音类型

# LLM 配置（字幕纠错用，OpenAI 兼容 API）
LLM_API_BASE_URL=                                 # API 地址
LLM_API_KEY=                                      # API Key
LLM_MODEL=                                        # 模型名称
```

## 技术栈

- **后端**：FastAPI + Uvicorn
- **语音识别**：OpenAI Whisper / Faster-Whisper
- **语音合成**：字节跳动火山引擎 TTS（WebSocket 双向流式协议）
- **音频处理**：pydub + mutagen
- **前端**：原生 HTML / JavaScript / CSS
