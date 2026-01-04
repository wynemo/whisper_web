# Faster-Whisper 调研报告

## 1. 概述

[faster-whisper](https://github.com/SYSTRAN/faster-whisper) 是 OpenAI Whisper 模型的 CTranslate2 重新实现，提供显著的性能提升。

## 2. 性能对比

| 指标 | openai-whisper | faster-whisper |
|------|---------------|----------------|
| 速度 | 基准 | **快 4 倍** |
| GPU 内存 (FP16) | 基准 | 相当 |
| GPU 内存 (INT8) | - | **减少 50%** |
| 13分钟音频 (RTX 3070 Ti) | 2m23s | **1m3s (FP16) / 59s (INT8)** |

## 3. 主要优势

1. **速度快**: 比 openai-whisper 快约 4 倍
2. **内存效率**: INT8 量化可减少一半 GPU 内存
3. **无 FFmpeg 依赖**: 使用 PyAV 解码音频
4. **量化支持**: 原生支持 int8, float16 计算类型
5. **批处理推理**: 支持 BatchedInferencePipeline 提高吞吐量

## 4. 系统要求

### Linux (推荐)
- Python 3.9+
- NVIDIA GPU + CUDA 12 + cuBLAS + cuDNN 9
- 或 CPU 模式运行

### macOS
- **兼容性问题**: CTranslate2 在 macOS 上可能有问题
- 仅建议在 Linux 上使用 faster-whisper

## 5. 安装

### 5.1 安装 faster-whisper

```bash
pip install faster-whisper
# 或使用 uv
uv add faster-whisper
```

### 5.2 安装 cuDNN 9 (Ubuntu 22.04 GPU 加速必需)

```bash
# 1. 添加 NVIDIA 仓库
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# 2. 安装 cuDNN 9 for CUDA 12
sudo apt install libcudnn9-cuda-12

# 3. 验证安装
ldconfig -p | grep cudnn
```

### 5.3 验证 GPU 支持

```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 应显示 CUDA Version: 12.x
```

## 6. API 使用

### 6.1 基本用法

```python
from faster_whisper import WhisperModel

# 初始化模型
model = WhisperModel(
    "large-v3",           # 模型大小: tiny, base, small, medium, large-v2, large-v3, turbo
    device="cuda",        # "cuda", "cpu", "auto"
    compute_type="float16"  # "float16", "int8", "int8_float16", "default"
)

# 转录
segments, info = model.transcribe(
    "audio.mp3",
    language="zh",        # 语言代码
    task="transcribe",    # "transcribe" 或 "translate"
    beam_size=5,
    word_timestamps=True  # 启用词级时间戳
)

# 遍历结果
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

### 6.2 返回值结构

**Segment 对象**:
```python
@dataclass
class Segment:
    id: int
    start: float         # 开始时间（秒）
    end: float           # 结束时间（秒）
    text: str            # 转录文本
    words: List[Word]    # 词级信息（需 word_timestamps=True）
```

**Word 对象**:
```python
@dataclass
class Word:
    start: float         # 词开始时间
    end: float           # 词结束时间
    word: str            # 词文本
    probability: float   # 置信度
```

### 6.3 生成 SRT 格式

faster-whisper 不直接输出 SRT，需要手动转换：

```python
def segments_to_srt(segments):
    srt_content = []
    for i, segment in enumerate(segments, 1):
        start = format_timestamp(segment.start)
        end = format_timestamp(segment.end)
        srt_content.append(f"{i}\n{start} --> {end}\n{segment.text.strip()}\n")
    return "\n".join(srt_content)

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
```

## 7. 与现有代码的对比

### 当前实现 (openai-whisper)
- 通过 subprocess 调用 `whisper` 命令行
- 直接输出 SRT 文件
- 流式输出进度信息

### faster-whisper 实现
- Python API 直接调用
- 返回 segments 生成器（可流式处理）
- 需要手动转换为 SRT 格式

## 8. 集成方案

### 8.1 配置选项

在 `config.py` 添加：
```python
# Whisper 引擎选择
WHISPER_ENGINE: str = "openai"  # "openai" 或 "faster"
```

### 8.2 依赖管理

在 `pyproject.toml` 添加可选依赖：
```toml
[project.optional-dependencies]
faster = ["faster-whisper"]
```

### 8.3 条件导入

```python
import platform

if settings.WHISPER_ENGINE == "faster" and platform.system() == "Linux":
    from faster_whisper import WhisperModel
else:
    # 使用 openai-whisper 命令行
    pass
```

## 9. 注意事项

1. **仅 Linux**: macOS 可能有兼容性问题，只在 Linux 上启用
2. **模型缓存**: 首次使用会下载模型，建议预下载
3. **GPU 内存**: large-v3 模型需要约 10GB GPU 内存
4. **流式输出**: segments 是生成器，需要迭代处理

## 10. 推荐配置

| 场景 | 推荐配置 |
|------|---------|
| 高性能 Linux | `device="cuda", compute_type="float16"` |
| 低显存 Linux | `device="cuda", compute_type="int8"` |
| CPU 模式 | `device="cpu", compute_type="int8"` |

## 参考链接

- [GitHub: faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [PyPI: faster-whisper](https://pypi.org/project/faster-whisper/)
- [CTranslate2](https://github.com/OpenNMT/CTranslate2)
