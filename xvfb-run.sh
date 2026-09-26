#!/bin/bash
DISP=:99
OUT=train

# Xvfb·ffmpeg는 별도 세션(setsid) → 터미널 Ctrl+C는 python에게만 감
setsid Xvfb $DISP -screen 0 720x480x24 -nolisten tcp &
XVFB_PID=$!
sleep 1
kill -0 $XVFB_PID 2>/dev/null || { echo "Xvfb 시작 실패 ($DISP 이미 사용 중?)" >&2; exit 1; }

DISPLAY=$DISP setsid ffmpeg -nostdin -f x11grab -draw_mouse 0 -video_size 720x480 -framerate 25 -i $DISP \
  -c:v libx264 -preset ultrafast -pix_fmt yuv420p -flush_packets 1 -y $OUT.mkv &
FFMPEG_PID=$!

# 녹화가 실제로 시작될 때까지 대기 (ffmpeg 시그널 핸들러 설치 전 SIGINT 유실 방지)
until [ -s $OUT.mkv ]; do
  kill -0 $FFMPEG_PID 2>/dev/null || { echo "ffmpeg 시작 실패" >&2; kill $XVFB_PID; exit 1; }
  sleep 0.1
done

cleaned=0
cleanup() {
  (( cleaned )) && return
  cleaned=1
  trap "" INT TERM HUP                        # 정리 중 재진입 방지 (변환도 setsid라 Ctrl+C 연타에 안 끊김)
  kill -INT $FFMPEG_PID 2>/dev/null           # ffmpeg에 보내는 SIGINT는 딱 1번
  for _ in $(seq 100); do                     # 최대 10초 대기
    kill -0 $FFMPEG_PID 2>/dev/null || break
    sleep 0.1
  done
  kill -0 $FFMPEG_PID 2>/dev/null && kill -TERM $FFMPEG_PID   # 최후수단 (flush_packets로 데이터는 이미 디스크에 있음)
  wait $FFMPEG_PID 2>/dev/null
  kill $XVFB_PID 2>/dev/null                  # 녹화가 끝난 뒤에 X 종료
  wait $XVFB_PID 2>/dev/null

  if setsid -w ffmpeg -v error -nostdin -i $OUT.mkv \
       -vf "fps=10,split[a][b];[a]palettegen[p];[b][p]paletteuse" \
       -loop 0 -y $OUT.gif; then
    rm -f $OUT.mkv
    echo "저장 완료: $OUT.gif"
  else
    echo "GIF 변환 실패 — $OUT.mkv 남겨둠" >&2
  fi
}
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM HUP

DISPLAY=$DISP python train.py
cleanup