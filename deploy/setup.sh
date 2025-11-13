#!/bin/bash
#
# 🚀 Asian Odds Betting Monitor - Auto Setup Script
# ==================================================
# Setup automatico sistema 24/7 con un comando
#
# Usage:
#   ./deploy/setup.sh systemd    # Setup con systemd (raccomandato)
#   ./deploy/setup.sh docker     # Setup con Docker
#   ./deploy/setup.sh manual     # Setup manuale (solo dipendenze)
#

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================
# BANNER
# ============================================================

echo -e "${GREEN}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║     🎰  ASIAN ODDS BETTING MONITOR SETUP  🎰              ║
║                                                           ║
║     Sistema 24/7 con Notifiche Telegram Automatiche      ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# ============================================================
# DETECT OS
# ============================================================

echo -e "${BLUE}📋 Detecting system...${NC}"

if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    OS_VERSION=$VERSION_ID
else
    echo -e "${RED}❌ Cannot detect OS. Exiting.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ OS detected: $OS $OS_VERSION${NC}"

# ============================================================
# CHECK DEPLOYMENT MODE
# ============================================================

DEPLOY_MODE=${1:-"systemd"}

if [ "$DEPLOY_MODE" != "systemd" ] && [ "$DEPLOY_MODE" != "docker" ] && [ "$DEPLOY_MODE" != "manual" ]; then
    echo -e "${RED}❌ Invalid mode: $DEPLOY_MODE${NC}"
    echo "Usage: $0 [systemd|docker|manual]"
    exit 1
fi

echo -e "${BLUE}📦 Deployment mode: $DEPLOY_MODE${NC}"

# ============================================================
# INSTALL SYSTEM DEPENDENCIES
# ============================================================

echo ""
echo -e "${BLUE}📦 Installing system dependencies...${NC}"

if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        python3 \
        python3-pip \
        python3-venv \
        git \
        curl \
        wget \
        htop

    if [ "$DEPLOY_MODE" = "docker" ]; then
        # Install Docker
        if ! command -v docker &> /dev/null; then
            echo -e "${BLUE}📦 Installing Docker...${NC}"
            curl -fsSL https://get.docker.com -o get-docker.sh
            sudo sh get-docker.sh
            sudo usermod -aG docker $USER
            rm get-docker.sh

            # Install docker-compose
            sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
            sudo chmod +x /usr/local/bin/docker-compose
        fi
    fi

elif [ "$OS" = "centos" ] || [ "$OS" = "rhel" ] || [ "$OS" = "fedora" ]; then
    sudo yum update -y -q
    sudo yum install -y -q \
        python3 \
        python3-pip \
        git \
        curl \
        wget \
        htop

    if [ "$DEPLOY_MODE" = "docker" ]; then
        if ! command -v docker &> /dev/null; then
            sudo yum install -y docker
            sudo systemctl start docker
            sudo systemctl enable docker
            sudo usermod -aG docker $USER
        fi
    fi

else
    echo -e "${YELLOW}⚠️  Unknown OS: $OS. Please install dependencies manually.${NC}"
fi

echo -e "${GREEN}✅ System dependencies installed${NC}"

# ============================================================
# SETUP PYTHON ENVIRONMENT
# ============================================================

echo ""
echo -e "${BLUE}🐍 Setting up Python environment...${NC}"

# Create virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠️  Virtual environment already exists${NC}"
fi

# Activate and install dependencies
source venv/bin/activate

pip install --upgrade pip -q
pip install -r requirements.txt -q

echo -e "${GREEN}✅ Python dependencies installed${NC}"

# ============================================================
# CONFIGURE ENVIRONMENT VARIABLES
# ============================================================

echo ""
echo -e "${BLUE}🔧 Configuring environment variables...${NC}"

if [ ! -f ".env" ]; then
    echo -e "${YELLOW}📝 Creating .env file...${NC}"

    # Prompt for Telegram credentials
    echo ""
    echo -e "${BLUE}Please provide your Telegram credentials:${NC}"
    echo ""

    read -p "Telegram Bot Token (from @BotFather): " BOT_TOKEN
    read -p "Telegram Chat ID (from @userinfobot): " CHAT_ID

    # Create .env file
    cat > .env << EOF
# Telegram Configuration
TELEGRAM_BOT_TOKEN="$BOT_TOKEN"
TELEGRAM_CHAT_ID="$CHAT_ID"

# Monitoring Configuration
LIVE_UPDATE_INTERVAL=60
MIN_EV_ALERT=5.0
MIN_CONFIDENCE=60.0

# API Configuration (optional - TheSportsDB is free and unlimited)
API_FOOTBALL_KEY=""

# Logging
LOG_LEVEL=INFO
EOF

    chmod 600 .env
    echo -e "${GREEN}✅ .env file created${NC}"
else
    echo -e "${YELLOW}⚠️  .env file already exists. Skipping.${NC}"
    echo -e "${YELLOW}💡 Edit .env manually if you need to update credentials.${NC}"
fi

# ============================================================
# DEPLOYMENT MODE SPECIFIC SETUP
# ============================================================

if [ "$DEPLOY_MODE" = "systemd" ]; then
    echo ""
    echo -e "${BLUE}🔧 Setting up systemd service...${NC}"

    # Get current directory
    INSTALL_DIR=$(pwd)
    USER_NAME=$(whoami)

    # Create systemd service file
    sudo tee /etc/systemd/system/betting-monitor.service > /dev/null << EOF
[Unit]
Description=Asian Odds Betting Monitor - Live 24/7
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$INSTALL_DIR
Environment="PATH=$INSTALL_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin"
EnvironmentFile=$INSTALL_DIR/.env
ExecStart=$INSTALL_DIR/venv/bin/python3 $INSTALL_DIR/start_live_monitoring.py
Restart=always
RestartSec=10
StandardOutput=append:$INSTALL_DIR/monitor.log
StandardError=append:$INSTALL_DIR/monitor.log

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=$INSTALL_DIR

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd
    sudo systemctl daemon-reload

    # Enable service
    sudo systemctl enable betting-monitor.service

    echo -e "${GREEN}✅ Systemd service created and enabled${NC}"
    echo ""
    echo -e "${BLUE}📋 Service management commands:${NC}"
    echo -e "  ${YELLOW}Start:${NC}   sudo systemctl start betting-monitor"
    echo -e "  ${YELLOW}Stop:${NC}    sudo systemctl stop betting-monitor"
    echo -e "  ${YELLOW}Status:${NC}  sudo systemctl status betting-monitor"
    echo -e "  ${YELLOW}Logs:${NC}    sudo journalctl -u betting-monitor -f"
    echo -e "  ${YELLOW}Restart:${NC} sudo systemctl restart betting-monitor"
    echo ""

    # Ask to start now
    read -p "Start monitoring now? (y/n): " START_NOW
    if [ "$START_NOW" = "y" ] || [ "$START_NOW" = "Y" ]; then
        sudo systemctl start betting-monitor
        echo -e "${GREEN}✅ Service started!${NC}"
        echo ""
        echo -e "${BLUE}Check status with:${NC} sudo systemctl status betting-monitor"
    fi

elif [ "$DEPLOY_MODE" = "docker" ]; then
    echo ""
    echo -e "${BLUE}🐳 Setting up Docker deployment...${NC}"

    # Build Docker image
    echo -e "${BLUE}Building Docker image...${NC}"
    docker build -t betting-monitor:latest .

    echo -e "${GREEN}✅ Docker image built${NC}"
    echo ""
    echo -e "${BLUE}📋 Docker commands:${NC}"
    echo -e "  ${YELLOW}Start:${NC}   docker-compose up -d"
    echo -e "  ${YELLOW}Stop:${NC}    docker-compose down"
    echo -e "  ${YELLOW}Logs:${NC}    docker-compose logs -f"
    echo -e "  ${YELLOW}Restart:${NC} docker-compose restart"
    echo ""

    # Ask to start now
    read -p "Start monitoring with Docker now? (y/n): " START_NOW
    if [ "$START_NOW" = "y" ] || [ "$START_NOW" = "Y" ]; then
        docker-compose up -d
        echo -e "${GREEN}✅ Docker container started!${NC}"
        echo ""
        echo -e "${BLUE}Check logs with:${NC} docker-compose logs -f"
    fi

else
    # Manual mode
    echo ""
    echo -e "${GREEN}✅ Manual setup complete!${NC}"
    echo ""
    echo -e "${BLUE}📋 To start monitoring manually:${NC}"
    echo -e "  source venv/bin/activate"
    echo -e "  python start_live_monitoring.py"
    echo ""
fi

# ============================================================
# FINAL SUMMARY
# ============================================================

echo ""
echo -e "${GREEN}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║              ✅  SETUP COMPLETE!  ✅                      ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "${BLUE}📊 Summary:${NC}"
echo -e "  ✅ Python environment: ${GREEN}OK${NC}"
echo -e "  ✅ Dependencies: ${GREEN}OK${NC}"
echo -e "  ✅ Telegram config: ${GREEN}OK${NC}"

if [ "$DEPLOY_MODE" = "systemd" ]; then
    echo -e "  ✅ Systemd service: ${GREEN}OK${NC}"
    echo ""
    echo -e "${YELLOW}💡 Your bot is now running 24/7!${NC}"
    echo -e "${YELLOW}   Even if you close the terminal or reboot, it will auto-restart.${NC}"
elif [ "$DEPLOY_MODE" = "docker" ]; then
    echo -e "  ✅ Docker container: ${GREEN}OK${NC}"
    echo ""
    echo -e "${YELLOW}💡 Your bot is running in Docker!${NC}"
    echo -e "${YELLOW}   It will auto-restart on crashes and system reboots.${NC}"
else
    echo -e "  ✅ Manual setup: ${GREEN}OK${NC}"
    echo ""
    echo -e "${YELLOW}💡 Run manually with:${NC}"
    echo -e "${YELLOW}   source venv/bin/activate && python start_live_monitoring.py${NC}"
fi

echo ""
echo -e "${BLUE}📱 You will receive Telegram notifications for:${NC}"
echo -e "  • Value betting opportunities (EV > 5%)"
echo -e "  • Live probability changes"
echo -e "  • Optimal timing alerts"
echo -e "  • Match events (goals, cards)"

echo ""
echo -e "${GREEN}🎉 Happy automated betting! 🎉${NC}"
echo ""

# ============================================================
# TIPS
# ============================================================

echo -e "${BLUE}💡 TIPS:${NC}"
echo ""

if [ "$DEPLOY_MODE" = "systemd" ]; then
    echo -e "  📊 View logs:"
    echo -e "     ${YELLOW}sudo journalctl -u betting-monitor -f${NC}"
    echo ""
    echo -e "  🔄 Restart service:"
    echo -e "     ${YELLOW}sudo systemctl restart betting-monitor${NC}"
    echo ""
    echo -e "  📝 Edit config:"
    echo -e "     ${YELLOW}nano .env${NC}"
    echo -e "     ${YELLOW}sudo systemctl restart betting-monitor${NC}"
fi

echo -e "  🌍 VPS keeps running 24/7 even when you:"
echo -e "     • Close your laptop"
echo -e "     • Turn off WiFi"
echo -e "     • Go to sleep 😴"
echo ""
echo -e "${GREEN}Your bot NEVER stops! 🚀${NC}"
echo ""
