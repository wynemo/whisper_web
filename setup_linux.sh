#!/bin/bash
set -e

echo "=== 同步基础依赖 ==="
uv sync

echo "=== 安装 faster-whisper (Linux) ==="
uv sync --extra faster

echo "=== 完成 ==="
uv pip list | grep -i -E "faster-whisper|whisper"
