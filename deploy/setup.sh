#!/bin/bash
#
# AsianOdds Live Betting Monitor - Setup Script
# ==============================================
#
# Automatic setup for 24/7 monitoring system
#
# Usage:
#   chmod +x deploy/setup.sh
#   ./deploy/setup.sh [systemd|docker|manual]
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_NAME="betting-monitor"
TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-8530766126:AAHs1ZoLwrwvT7JuPyn_9ymNVyddPtUXi-g}"
TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID:--1003278011521}"

# ============================================================
# Helper Functions
# ============================================================

print_header() {
    echo -e "${GREEN}"
    echo "============================================================"
    echo "$1"
    echo "============================================================"
    echo -e "${NC}"
}

print_info() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# ============================================================
# Pre-flight Checks
# ============================================================

check_dependencies() {
    print_header "Checking Dependencies"

    # Check Python
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version)
        print_info "Python found: $PYTHON_VERSION"
    else
        print_error "Python 3 not found!"
        exit 1
    fi

    # Check pip
    if command -v pip3 &> /dev/null; then
        print_info "pip3 found"
    else
        print_error "pip3 not found!"
        exit 1
    fi

    # Check requirements.txt
    if [ ! -f "$PROJECT_DIR/requirements.txt" ]; then
        print_error "requirements.txt not found!"
        exit 1
    fi

    print_info "All dependencies checked"
}

# ============================================================
# Installation Methods
# ============================================================

install_systemd() {
    print_header "Setting up Systemd Service"

    # Check if systemd is available
    if ! command -v systemctl &> /dev/null; then
        print_error "systemd not available on this system"
        exit 1
    fi

    # Install Python dependencies
    print_info "Installing Python dependencies..."
    cd "$PROJECT_DIR"
    pip3 install -r requirements.txt --user || {
        print_warning "Some dependencies failed, continuing..."
    }

    # Update service file paths
    SERVICE_FILE="$PROJECT_DIR/deploy/betting-monitor.service"

    if [ ! -f "$SERVICE_FILE" ]; then
        print_error "Service file not found: $SERVICE_FILE"
        exit 1
    fi

    # Update environment variables in service file
    sed -i "s|Environment=\"TELEGRAM_BOT_TOKEN=.*\"|Environment=\"TELEGRAM_BOT_TOKEN=$TELEGRAM_BOT_TOKEN\"|" "$SERVICE_FILE"
    sed -i "s|Environment=\"TELEGRAM_CHAT_ID=.*\"|Environment=\"TELEGRAM_CHAT_ID=$TELEGRAM_CHAT_ID\"|" "$SERVICE_FILE"

    # Copy service file to systemd directory
    print_info "Installing systemd service..."
    sudo cp "$SERVICE_FILE" /etc/systemd/system/
    sudo systemctl daemon-reload

    # Enable and start service
    print_info "Enabling service to start on boot..."
    sudo systemctl enable $SERVICE_NAME

    print_info "Starting service..."
    sudo systemctl start $SERVICE_NAME

    # Check status
    sleep 2
    if sudo systemctl is-active --quiet $SERVICE_NAME; then
        print_info "Service started successfully!"
    else
        print_error "Service failed to start. Check logs with: sudo journalctl -u $SERVICE_NAME -f"
        exit 1
    fi

    print_header "Systemd Setup Complete!"
    echo ""
    echo "Service is now running 24/7. Useful commands:"
    echo ""
    echo "  View status:  sudo systemctl status $SERVICE_NAME"
    echo "  View logs:    sudo journalctl -u $SERVICE_NAME -f"
    echo "  Stop:         sudo systemctl stop $SERVICE_NAME"
    echo "  Restart:      sudo systemctl restart $SERVICE_NAME"
    echo "  Disable:      sudo systemctl disable $SERVICE_NAME"
    echo ""
}

install_docker() {
    print_header "Setting up Docker Container"

    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker not installed!"
        print_info "Install Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose not installed!"
        print_info "Install Docker Compose: https://docs.docker.com/compose/install/"
        exit 1
    fi

    cd "$PROJECT_DIR/deploy"

    # Build and start container
    print_info "Building Docker image..."
    docker-compose build

    print_info "Starting container..."
    docker-compose up -d

    # Check if running
    sleep 3
    if docker-compose ps | grep -q "Up"; then
        print_info "Container started successfully!"
    else
        print_error "Container failed to start. Check logs with: docker-compose logs -f"
        exit 1
    fi

    print_header "Docker Setup Complete!"
    echo ""
    echo "Container is now running 24/7. Useful commands:"
    echo ""
    echo "  View logs:    cd deploy && docker-compose logs -f"
    echo "  Stop:         cd deploy && docker-compose stop"
    echo "  Restart:      cd deploy && docker-compose restart"
    echo "  Remove:       cd deploy && docker-compose down"
    echo "  Rebuild:      cd deploy && docker-compose up -d --build"
    echo ""
}

install_manual() {
    print_header "Manual Setup"

    # Install dependencies
    print_info "Installing Python dependencies..."
    cd "$PROJECT_DIR"
    pip3 install -r requirements.txt --user || {
        print_warning "Some dependencies failed, continuing..."
    }

    # Create .env file
    ENV_FILE="$PROJECT_DIR/.env"
    if [ ! -f "$ENV_FILE" ]; then
        print_info "Creating .env file..."
        cat > "$ENV_FILE" << EOF
# Telegram Configuration
TELEGRAM_BOT_TOKEN=$TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID=$TELEGRAM_CHAT_ID

# Monitoring Settings
UPDATE_INTERVAL=60
LOG_LEVEL=INFO
EOF
        print_info ".env file created"
    else
        print_warning ".env file already exists, skipping"
    fi

    # Create start script
    START_SCRIPT="$PROJECT_DIR/start.sh"
    cat > "$START_SCRIPT" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
export $(cat .env | xargs)
nohup python3 start_live_monitoring.py >> monitor.log 2>&1 &
echo "Monitor started in background. PID: $!"
echo "View logs: tail -f monitor.log"
EOF

    chmod +x "$START_SCRIPT"
    print_info "Start script created: $START_SCRIPT"

    # Create stop script
    STOP_SCRIPT="$PROJECT_DIR/stop.sh"
    cat > "$STOP_SCRIPT" << 'EOF'
#!/bin/bash
pkill -f "start_live_monitoring.py"
echo "Monitor stopped"
EOF

    chmod +x "$STOP_SCRIPT"
    print_info "Stop script created: $STOP_SCRIPT"

    print_header "Manual Setup Complete!"
    echo ""
    echo "Start monitoring:"
    echo "  ./start.sh"
    echo ""
    echo "Stop monitoring:"
    echo "  ./stop.sh"
    echo ""
    echo "View logs:"
    echo "  tail -f monitor.log"
    echo ""
    echo "For 24/7 operation, consider using systemd or Docker."
    echo ""
}

# ============================================================
# Main Script
# ============================================================

main() {
    print_header "AsianOdds Live Betting Monitor - Setup"

    # Check dependencies
    check_dependencies

    # Determine installation method
    METHOD="${1:-manual}"

    case "$METHOD" in
        systemd)
            install_systemd
            ;;
        docker)
            install_docker
            ;;
        manual)
            install_manual
            ;;
        *)
            print_error "Unknown method: $METHOD"
            echo ""
            echo "Usage: $0 [systemd|docker|manual]"
            echo ""
            echo "  systemd - Install as system service (Linux)"
            echo "  docker  - Run in Docker container"
            echo "  manual  - Manual setup with scripts"
            echo ""
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
