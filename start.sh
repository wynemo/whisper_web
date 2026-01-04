#!/bin/bash
# 启动 whisper_web 服务的脚本
# 自动设置 NVIDIA 库路径以支持 faster-whisper GPU 加速

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 加载环境变量文件 (先加载，以便 uv 能正确工作)
if [ -f "settings.env" ]; then
    echo "加载 settings.env..."
    export $(grep -v '^#' settings.env | xargs)
fi

# 尝试设置 NVIDIA 库路径 (使用 uv run python 来获取正确的包路径)
echo "检测 NVIDIA 库路径..."
CUDNN_PATH=$(uv run python -c "import nvidia.cudnn.lib; print(nvidia.cudnn.lib.__path__[0])" 2>/dev/null)
CUBLAS_PATH=$(uv run python -c "import nvidia.cublas.lib; print(nvidia.cublas.lib.__path__[0])" 2>/dev/null)

if [ -n "$CUDNN_PATH" ] && [ -n "$CUBLAS_PATH" ]; then
    export LD_LIBRARY_PATH="$CUDNN_PATH:$CUBLAS_PATH:$LD_LIBRARY_PATH"
    echo "已设置 NVIDIA 库路径:"
    echo "  cuDNN: $CUDNN_PATH"
    echo "  cuBLAS: $CUBLAS_PATH"
else
    echo "警告: 未找到 NVIDIA 库，GPU 加速可能不可用"
fi

# 启动服务
echo "启动 whisper_web 服务..."
exec uv run main.py "$@"
