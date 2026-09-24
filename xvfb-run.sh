#!/bin/bash
Xvfb :99 -screen 0 256x256x24 &
XVFB_PID=$!
sleep 1

DISPLAY=:99 ffmpeg -f x11grab -video_size 256x256 -framerate 25 -i :99 \
  -c:v libx264 -preset ultrafast -pix_fmt yuv420p -y train.mp4 &
FFMPEG_PID=$!

cleanup() {
    kill -INT $FFMPEG_PID 2>/dev/null
    wait $FFMPEG_PID 2>/dev/null
    kill $XVFB_PID 2>/dev/null
    echo "저장 완료: train.mp4"
    exit 0
}
trap cleanup INT TERM

DISPLAY=:99 python train.py
cleanup
EOF
