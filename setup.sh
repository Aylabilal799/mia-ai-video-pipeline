#!/bin/bash
set -e

echo "🚀 Setting up YouTube Shorts Bot (CPU + Free APIs)"

# 1. Install system dependencies
sudo apt update
sudo apt install -y python3 python3-pip python3-venv redis-server ffmpeg git curl

# 2. Clone Agnes Video Generator inside this project folder, with ITS OWN venv.
#    We never call agnes's own start.sh -- that script opens a browser window,
#    which is exactly the behavior we don't want on a headless server.
if [ ! -d "agnes-video-generator" ]; then
    echo "📦 Cloning Agnes Video Generator..."
    git clone https://github.com/lcy362/agnes-video-generator.git
fi
cd agnes-video-generator
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
cd ..

# 3. Set up this bot's own Python virtual environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate

# 4. Create .env from example if not present
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "⚙️ Please edit .env and set your DISCORD_TOKEN and AGNES_API_KEY."
fi

# 5. Start Redis
sudo systemctl enable redis-server
sudo systemctl start redis-server

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env with your tokens:"
echo "   nano .env"
echo ""
echo "2. In Terminal 1, start the Celery worker:"
echo "   ./start_worker.sh"
echo ""
echo "3. In Terminal 2, start the Discord bot:"
echo "   ./start_bot.sh"
echo ""
echo "The Agnes AI video server now starts itself automatically and headlessly"
echo "the first time someone runs !video -- no browser window will open, and"
echo "you won't need to paste the script into a web form."
echo ""
echo "If something goes wrong with video generation, check:"
echo "   agnes-video-generator/server.log"
echo ""
echo "Then invite your bot to Discord and use: !video <your script>"
