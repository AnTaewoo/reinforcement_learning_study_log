#!/bin/bash
Xvfb :99 -screen 0 256x256x24 &
XVFB_PID=$!
sleep 1
DISPLAY=:99 ffmpeg -f x11grab -video_size 256x256 -framerate 25 -i :99 \
  -c:v libx264 -preset ultrafast -pix_fmt yuv420p -y train.mkv &
FFMPEG_PID=$!
cleanup() {
  kill -INT $FFMPEG_PID 2>/dev/null
  while kill -0 $FFMPEG_PID 2>/dev/null; do sleep 0.2; done
  kill $XVFB_PID 2>/dev/null
  ffmpeg -v error -i train.mkv \
    -vf "fps=10,split[a][b];[a]palettegen[p];[b][p]paletteuse" \
    -loop 0 -y train.gif
  echo "저장 완료: train.gif"
  exit 0
}
trap cleanup INT TERM
DISPLAY=:99 python train.py
cleanup