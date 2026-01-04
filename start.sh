#!/bin/bash
# 启动 whisper_web 服务的脚本
# 自动设置 NVIDIA 库路径以支持 faster-whisper GPU 加速

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 尝试设置 NVIDIA 库路径
if command -v python3 &> /dev/null; then
    CUDNN_PATH=$(python3 -c "import nvidia.cudnn.lib; print(nvidia.cudnn.lib.__path__[0])" 2>/dev/null)
    CUBLAS_PATH=$(python3 -c "import nvidia.cublas.lib; print(nvidia.cublas.lib.__path__[0])" 2>/dev/null)

    if [ -n "$CUDNN_PATH" ] && [ -n "$CUBLAS_PATH" ]; then
        export LD_LIBRARY_PATH="$CUDNN_PATH:$CUBLAS_PATH:$LD_LIBRARY_PATH"
        echo "已设置 NVIDIA 库路径:"
        echo "  cuDNN: $CUDNN_PATH"
        echo "  cuBLAS: $CUBLAS_PATH"
    fi
fi

# 加载环境变量文件
if [ -f "settings.env" ]; then
    echo "加载 settings.env..."
    export $(grep -v '^#' settings.env | xargs)
fi

# 启动服务
echo "启动 whisper_web 服务..."
exec uv run main.py "$@"
