#!/bin/bash
# Optional: stop the headless Agnes video server started automatically by
# video_generator.py. You normally don't need this -- it'll just get
# restarted the next time someone runs !video.
cd "$(dirname "$0")/agnes-video-generator"
if [ -f server.pid ]; then
    PID=$(cat server.pid)
    if kill "$PID" 2>/dev/null; then
        echo "✅ Stopped Agnes server (PID $PID)"
    else
        echo "⚠️ No running process with PID $PID (already stopped?)"
    fi
    rm -f server.pid
else
    echo "⚠️ No server.pid found -- Agnes server may not be running."
fi
