#!/bin/bash
#
# 🧪 VPS Setup Test Script
# =========================
# Verifica che il VPS sia configurato correttamente
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════╗"
echo "║                                           ║"
echo "║     🧪  VPS SETUP TEST SCRIPT  🧪         ║"
echo "║                                           ║"
echo "╚═══════════════════════════════════════════╝"
echo -e "${NC}"
echo ""

ERRORS=0
WARNINGS=0

# ============================================================
# TEST 1: Python Version
# ============================================================

echo -e "${BLUE}[1/10] Checking Python version...${NC}"

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
        echo -e "${GREEN}✅ Python $PYTHON_VERSION (OK)${NC}"
    else
        echo -e "${RED}❌ Python $PYTHON_VERSION (Need >= 3.8)${NC}"
        ERRORS=$((ERRORS+1))
    fi
else
    echo -e "${RED}❌ Python not found${NC}"
    ERRORS=$((ERRORS+1))
fi

# ============================================================
# TEST 2: Virtual Environment
# ============================================================

echo -e "${BLUE}[2/10] Checking virtual environment...${NC}"

if [ -d "venv" ]; then
    echo -e "${GREEN}✅ Virtual environment exists${NC}"

    # Check activation
    if [ -f "venv/bin/activate" ]; then
        echo -e "${GREEN}✅ Activation script found${NC}"
    else
        echo -e "${RED}❌ Activation script missing${NC}"
        ERRORS=$((ERRORS+1))
    fi
else
    echo -e "${YELLOW}⚠️  Virtual environment not found (run setup.sh first)${NC}"
    WARNINGS=$((WARNINGS+1))
fi

# ============================================================
# TEST 3: Dependencies
# ============================================================

echo -e "${BLUE}[3/10] Checking Python dependencies...${NC}"

if [ -f "requirements.txt" ]; then
    echo -e "${GREEN}✅ requirements.txt found${NC}"

    if [ -d "venv" ]; then
        source venv/bin/activate
        MISSING_DEPS=0

        # Check key dependencies
        for pkg in "requests" "python-telegram-bot"; do
            if ! pip show $pkg &> /dev/null; then
                echo -e "${RED}❌ Missing: $pkg${NC}"
                MISSING_DEPS=$((MISSING_DEPS+1))
            fi
        done

        if [ $MISSING_DEPS -eq 0 ]; then
            echo -e "${GREEN}✅ All key dependencies installed${NC}"
        else
            echo -e "${RED}❌ $MISSING_DEPS dependencies missing${NC}"
            ERRORS=$((ERRORS+1))
        fi
    fi
else
    echo -e "${RED}❌ requirements.txt not found${NC}"
    ERRORS=$((ERRORS+1))
fi

# ============================================================
# TEST 4: Environment Variables
# ============================================================

echo -e "${BLUE}[4/10] Checking environment variables...${NC}"

if [ -f ".env" ]; then
    echo -e "${GREEN}✅ .env file exists${NC}"

    # Source .env
    export $(grep -v '^#' .env | xargs)

    # Check Telegram token
    if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
        echo -e "${RED}❌ TELEGRAM_BOT_TOKEN not set${NC}"
        ERRORS=$((ERRORS+1))
    else
        TOKEN_LEN=${#TELEGRAM_BOT_TOKEN}
        if [ $TOKEN_LEN -lt 40 ]; then
            echo -e "${RED}❌ TELEGRAM_BOT_TOKEN looks invalid (too short)${NC}"
            ERRORS=$((ERRORS+1))
        else
            echo -e "${GREEN}✅ TELEGRAM_BOT_TOKEN set (${TOKEN_LEN} chars)${NC}"
        fi
    fi

    # Check Chat ID
    if [ -z "$TELEGRAM_CHAT_ID" ]; then
        echo -e "${RED}❌ TELEGRAM_CHAT_ID not set${NC}"
        ERRORS=$((ERRORS+1))
    else
        echo -e "${GREEN}✅ TELEGRAM_CHAT_ID set${NC}"
    fi

else
    echo -e "${RED}❌ .env file not found (run setup.sh)${NC}"
    ERRORS=$((ERRORS+1))
fi

# ============================================================
# TEST 5: Telegram API Connectivity
# ============================================================

echo -e "${BLUE}[5/10] Testing Telegram API connectivity...${NC}"

if [ ! -z "$TELEGRAM_BOT_TOKEN" ]; then
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
        "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getMe")

    if [ "$HTTP_CODE" == "200" ]; then
        echo -e "${GREEN}✅ Telegram API reachable (HTTP $HTTP_CODE)${NC}"

        # Get bot info
        BOT_INFO=$(curl -s "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getMe")
        BOT_USERNAME=$(echo $BOT_INFO | grep -o '"username":"[^"]*' | cut -d'"' -f4)

        if [ ! -z "$BOT_USERNAME" ]; then
            echo -e "${GREEN}✅ Bot username: @$BOT_USERNAME${NC}"
        fi
    elif [ "$HTTP_CODE" == "401" ]; then
        echo -e "${RED}❌ Invalid bot token (HTTP 401)${NC}"
        ERRORS=$((ERRORS+1))
    else
        echo -e "${RED}❌ Cannot reach Telegram API (HTTP $HTTP_CODE)${NC}"
        ERRORS=$((ERRORS+1))
    fi
else
    echo -e "${YELLOW}⚠️  Skipping (no token configured)${NC}"
    WARNINGS=$((WARNINGS+1))
fi

# ============================================================
# TEST 6: Network Connectivity
# ============================================================

echo -e "${BLUE}[6/10] Testing network connectivity...${NC}"

# Test DNS
if ping -c 1 google.com &> /dev/null; then
    echo -e "${GREEN}✅ DNS resolution working${NC}"
else
    echo -e "${RED}❌ DNS resolution failed${NC}"
    ERRORS=$((ERRORS+1))
fi

# Test HTTPS
if curl -s -o /dev/null -w "%{http_code}" https://www.google.com | grep -q "200"; then
    echo -e "${GREEN}✅ HTTPS connectivity working${NC}"
else
    echo -e "${RED}❌ HTTPS connectivity failed${NC}"
    ERRORS=$((ERRORS+1))
fi

# ============================================================
# TEST 7: System Resources
# ============================================================

echo -e "${BLUE}[7/10] Checking system resources...${NC}"

# RAM
TOTAL_RAM=$(free -m | awk '/^Mem:/ {print $2}')
FREE_RAM=$(free -m | awk '/^Mem:/ {print $7}')

echo -e "  RAM: ${TOTAL_RAM}MB total, ${FREE_RAM}MB available"

if [ $FREE_RAM -lt 100 ]; then
    echo -e "${RED}❌ Low RAM (< 100MB free)${NC}"
    ERRORS=$((ERRORS+1))
elif [ $FREE_RAM -lt 200 ]; then
    echo -e "${YELLOW}⚠️  Low RAM (< 200MB free)${NC}"
    WARNINGS=$((WARNINGS+1))
else
    echo -e "${GREEN}✅ RAM OK${NC}"
fi

# Disk Space
DISK_FREE=$(df -h . | awk 'NR==2 {print $4}')
DISK_FREE_GB=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')

echo -e "  Disk: ${DISK_FREE} free"

if [ $DISK_FREE_GB -lt 1 ]; then
    echo -e "${RED}❌ Low disk space (< 1GB)${NC}"
    ERRORS=$((ERRORS+1))
elif [ $DISK_FREE_GB -lt 2 ]; then
    echo -e "${YELLOW}⚠️  Low disk space (< 2GB)${NC}"
    WARNINGS=$((WARNINGS+1))
else
    echo -e "${GREEN}✅ Disk space OK${NC}"
fi

# CPU
CPU_COUNT=$(nproc)
echo -e "  CPU: ${CPU_COUNT} core(s)"

if [ $CPU_COUNT -lt 1 ]; then
    echo -e "${RED}❌ No CPU detected?${NC}"
    ERRORS=$((ERRORS+1))
else
    echo -e "${GREEN}✅ CPU OK${NC}"
fi

# ============================================================
# TEST 8: Firewall
# ============================================================

echo -e "${BLUE}[8/10] Checking firewall...${NC}"

if command -v ufw &> /dev/null; then
    UFW_STATUS=$(sudo ufw status 2>/dev/null || echo "inactive")

    if echo "$UFW_STATUS" | grep -q "Status: active"; then
        echo -e "${GREEN}✅ UFW firewall active${NC}"

        # Check if 443 is allowed
        if echo "$UFW_STATUS" | grep -q "443"; then
            echo -e "${GREEN}✅ Port 443 (HTTPS) allowed${NC}"
        else
            echo -e "${YELLOW}⚠️  Port 443 not explicitly allowed (may still work)${NC}"
            WARNINGS=$((WARNINGS+1))
        fi
    else
        echo -e "${YELLOW}⚠️  UFW not active (OK if using cloud provider firewall)${NC}"
        WARNINGS=$((WARNINGS+1))
    fi
else
    echo -e "${YELLOW}⚠️  UFW not installed (OK if using cloud provider firewall)${NC}"
    WARNINGS=$((WARNINGS+1))
fi

# ============================================================
# TEST 9: Required Files
# ============================================================

echo -e "${BLUE}[9/10] Checking required files...${NC}"

REQUIRED_FILES=(
    "start_live_monitoring.py"
    "requirements.txt"
    "ai_system/__init__.py"
    "ai_system/telegram_notifier.py"
    "ai_system/live_monitor.py"
)

MISSING_FILES=0

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅ $file${NC}"
    else
        echo -e "${RED}❌ Missing: $file${NC}"
        MISSING_FILES=$((MISSING_FILES+1))
    fi
done

if [ $MISSING_FILES -gt 0 ]; then
    echo -e "${RED}❌ $MISSING_FILES required files missing${NC}"
    ERRORS=$((ERRORS+1))
fi

# ============================================================
# TEST 10: Systemd Service (if exists)
# ============================================================

echo -e "${BLUE}[10/10] Checking systemd service...${NC}"

if [ -f "/etc/systemd/system/betting-monitor.service" ]; then
    echo -e "${GREEN}✅ Service file exists${NC}"

    # Check if enabled
    if systemctl is-enabled betting-monitor &> /dev/null; then
        echo -e "${GREEN}✅ Service enabled (will start at boot)${NC}"
    else
        echo -e "${YELLOW}⚠️  Service not enabled${NC}"
        WARNINGS=$((WARNINGS+1))
    fi

    # Check if running
    if systemctl is-active betting-monitor &> /dev/null; then
        echo -e "${GREEN}✅ Service running${NC}"
    else
        echo -e "${YELLOW}⚠️  Service not running${NC}"
        WARNINGS=$((WARNINGS+1))
    fi
else
    echo -e "${YELLOW}⚠️  Systemd service not installed (run setup.sh systemd)${NC}"
    WARNINGS=$((WARNINGS+1))
fi

# ============================================================
# SUMMARY
# ============================================================

echo ""
echo -e "${BLUE}═══════════════════════════════════════════${NC}"
echo -e "${BLUE}              TEST SUMMARY                 ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════${NC}"
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✅ ALL TESTS PASSED! 🎉${NC}"
    echo ""
    echo -e "${GREEN}Your VPS is ready for 24/7 monitoring!${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo -e "  1. ${YELLOW}sudo systemctl start betting-monitor${NC}"
    echo -e "  2. ${YELLOW}sudo journalctl -u betting-monitor -f${NC}"
    echo ""
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠️  TESTS PASSED WITH WARNINGS${NC}"
    echo ""
    echo -e "Errors: ${GREEN}$ERRORS${NC}"
    echo -e "Warnings: ${YELLOW}$WARNINGS${NC}"
    echo ""
    echo -e "${YELLOW}Your VPS should work, but check warnings above.${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}❌ TESTS FAILED${NC}"
    echo ""
    echo -e "Errors: ${RED}$ERRORS${NC}"
    echo -e "Warnings: ${YELLOW}$WARNINGS${NC}"
    echo ""
    echo -e "${RED}Please fix errors above before deploying.${NC}"
    echo ""
    echo -e "${BLUE}Common fixes:${NC}"
    echo -e "  • Missing dependencies: ${YELLOW}pip install -r requirements.txt${NC}"
    echo -e "  • Missing .env: ${YELLOW}./deploy/setup.sh systemd${NC}"
    echo -e "  • Network issues: ${YELLOW}Check firewall/security groups${NC}"
    echo ""
    exit 1
fi
