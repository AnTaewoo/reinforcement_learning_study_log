#!/bin/bash
Xvfb :99 -screen 0 256x256x24 &
XVFB_PID=$!
sleep 1
DISPLAY=:99 ffmpeg -f x11grab -video_size 256x256 -framerate 25 -i :99 \
  -c:v libx264 -preset ultrafast -pix_fmt yuv420p -y test.mkv &
FFMPEG_PID=$!
cleanup() {
  kill -INT $FFMPEG_PID 2>/dev/null
  while kill -0 $FFMPEG_PID 2>/dev/null; do sleep 0.2; done
  kill $XVFB_PID 2>/dev/null
  ffmpeg -v error -i test.mkv \
    -vf "fps=10,split[a][b];[a]palettegen[p];[b][p]paletteuse" \
    -loop 0 -y test.gif && rm -f test.mkv
  echo "저장 완료: test.gif"
  exit 0
}
trap cleanup INT TERM
DISPLAY=:99 python test.py
cleanup