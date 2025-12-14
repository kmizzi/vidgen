#!/bin/bash
#
# WAN 2.2 NSFW Video Generation Server - One-Shot Deployment
#
# This script provisions a DigitalOcean GPU droplet and sets up everything
# needed for video generation using the Phr00t Rapid AIO NSFW model.
#
# Usage:
#   ./deploy.sh              # Create new droplet and deploy
#   ./deploy.sh <IP>         # Deploy to existing server
#   ./deploy.sh --destroy    # Destroy the droplet
#   ./deploy.sh --name NAME  # Create droplet with custom name
#
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

# Default values
DEFAULT_DROPLET_NAME="wan22-nsfw-gpu"
DROPLET_SIZE="gpu-h100x1-80gb"
DROPLET_REGION="tor1"
DROPLET_IMAGE="gpu-h100x1-base"

# Load existing config from .env if it exists
if [[ -f "$ENV_FILE" ]]; then
    source "$ENV_FILE"
fi

# Use saved droplet name or default
DROPLET_NAME="${DROPLET_NAME:-$DEFAULT_DROPLET_NAME}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $1"; }
success() { echo -e "${GREEN}[$(date +%H:%M:%S)] ✓${NC} $1"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] !${NC} $1"; }
error() { echo -e "${RED}[$(date +%H:%M:%S)] ✗${NC} $1"; exit 1; }

# Progress bar function
# Usage: progress_bar <current> <total> <message>
progress_bar() {
    local current=$1
    local total=$2
    local message=$3
    local width=40
    local percent=$((current * 100 / total))
    local filled=$((current * width / total))
    local empty=$((width - filled))

    printf "\r${BLUE}[%3d%%]${NC} [" "$percent"
    printf "%${filled}s" | tr ' ' '█'
    printf "%${empty}s" | tr ' ' '░'
    printf "] %s" "$message"
}

# Check prerequisites
check_prereqs() {
    if ! command -v doctl &> /dev/null; then
        error "doctl not found. Install with: brew install doctl"
    fi
    if ! doctl account get &> /dev/null; then
        error "doctl not authenticated. Run: doctl auth init"
    fi
}

# Destroy droplet
destroy_droplet() {
    if [[ -z "$DROPLET_NAME" ]]; then
        error "No droplet name found. Check .env file or specify manually."
    fi
    log "Destroying droplet $DROPLET_NAME..."
    if doctl compute droplet delete "$DROPLET_NAME" --force 2>/dev/null; then
        success "Droplet destroyed"
        rm -f "$ENV_FILE"
        success "Cleared .env file"
    else
        warn "Droplet not found or already destroyed"
        rm -f "$ENV_FILE"
    fi
    exit 0
}

# Get or create droplet
get_or_create_droplet() {
    local ip="$1"

    if [[ -n "$ip" ]]; then
        SERVER_IP="$ip"
        log "Using provided IP: $SERVER_IP"
        return
    fi

    # Check for existing droplet
    local existing_ip=$(doctl compute droplet list --format Name,PublicIPv4 --no-header | grep "^$DROPLET_NAME " | awk '{print $2}')
    if [[ -n "$existing_ip" ]]; then
        SERVER_IP="$existing_ip"
        success "Found existing droplet: $SERVER_IP"
        return
    fi

    # Create new droplet
    log "Creating GPU droplet (this takes ~2 minutes)..."
    doctl compute droplet create "$DROPLET_NAME" \
        --size "$DROPLET_SIZE" \
        --image "$DROPLET_IMAGE" \
        --region "$DROPLET_REGION" \
        --ssh-keys "$(doctl compute ssh-key list --format ID --no-header | head -1)" \
        --wait

    # Get IP
    sleep 5
    SERVER_IP=$(doctl compute droplet get "$DROPLET_NAME" --format PublicIPv4 --no-header)
    success "Droplet created: $SERVER_IP"

    # Wait for SSH
    log "Waiting for SSH to become available..."
    for i in {1..30}; do
        if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no root@"$SERVER_IP" exit 2>/dev/null; then
            success "SSH ready"
            return
        fi
        sleep 10
    done
    error "Timeout waiting for SSH"
}

# Deploy to server
deploy() {
    log "Starting deployment to $SERVER_IP..."

    # Copy files
    progress_bar 1 8 "Copying files to server...                    "
    scp -o StrictHostKeyChecking=no -q \
        "$SCRIPT_DIR/generate_video.py" \
        "$SCRIPT_DIR/nsfw_t2v_proper_workflow.json" \
        "$SCRIPT_DIR/nsfw_i2v_workflow.json" \
        root@"$SERVER_IP":/root/
    echo ""

    # Run setup with progress parsing
    log "Running server setup..."
    echo ""

    ssh -o StrictHostKeyChecking=no root@"$SERVER_IP" 'bash -s' << 'SETUP_SCRIPT' 2>&1 | while IFS= read -r line; do
#!/bin/bash
set -e

export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a

# Progress markers: PROGRESS:<step>:<total>:<message>
echo "PROGRESS:2:8:Installing system packages..."
apt-get update -qq
apt-get install -y -qq git python3-pip python3-venv ffmpeg libgl1 > /dev/null

echo "PROGRESS:3:8:Setting up ComfyUI..."
if [[ ! -d /opt/comfyui/ComfyUI ]]; then
    mkdir -p /opt/comfyui
    cd /opt/comfyui
    git clone -q https://github.com/comfyanonymous/ComfyUI.git
fi

cd /opt/comfyui/ComfyUI

if [[ ! -d venv ]]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -q --upgrade pip
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -q -r requirements.txt

echo "PROGRESS:4:8:Installing VideoHelperSuite..."
if [[ ! -d custom_nodes/ComfyUI-VideoHelperSuite ]]; then
    cd custom_nodes
    git clone -q https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
    cd ComfyUI-VideoHelperSuite
    pip install -q -r requirements.txt
    cd ../..
fi

echo "PROGRESS:5:8:Downloading models (~23GB + 850MB, please wait)..."
mkdir -p models/checkpoints
mkdir -p models/clip_vision
if [[ ! -f models/checkpoints/wan2.2-rapid-mega-aio-nsfw-v12.1.safetensors ]]; then
    wget -q -O models/checkpoints/wan2.2-rapid-mega-aio-nsfw-v12.1.safetensors \
        "https://huggingface.co/Phr00t/WAN2.2-14B-Rapid-AllInOne/resolve/main/Mega-v12/wan2.2-rapid-mega-aio-nsfw-v12.1.safetensors"
else
    echo "Main model already downloaded"
fi
# Download CLIP Vision model for I2V
if [[ ! -f models/clip_vision/sigclip_vision_patch14_384.safetensors ]]; then
    wget -q -O models/clip_vision/sigclip_vision_patch14_384.safetensors \
        "https://huggingface.co/Comfy-Org/sigclip_vision_384/resolve/main/sigclip_vision_patch14_384.safetensors"
else
    echo "CLIP Vision model already downloaded"
fi

echo "PROGRESS:6:8:Creating systemd service..."
cat > /etc/systemd/system/comfyui.service << 'EOF'
[Unit]
Description=ComfyUI Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/comfyui/ComfyUI
Environment=PATH=/opt/comfyui/ComfyUI/venv/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/opt/comfyui/ComfyUI/venv/bin/python main.py --listen 0.0.0.0 --port 8188
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable comfyui > /dev/null 2>&1
systemctl restart comfyui

echo "PROGRESS:7:8:Installing CLI tool..."
cp /root/generate_video.py /usr/local/bin/generate-video
chmod +x /usr/local/bin/generate-video

mkdir -p /opt/comfyui/ComfyUI/user/default/workflows
cp /root/nsfw_t2v_proper_workflow.json /opt/comfyui/ComfyUI/user/default/workflows/
cp /root/nsfw_i2v_workflow.json /opt/comfyui/ComfyUI/user/default/workflows/

touch /var/log/video-prompts.log

echo "PROGRESS:8:8:Waiting for ComfyUI to start..."
for i in {1..30}; do
    if curl -s http://localhost:8188 > /dev/null 2>&1; then
        break
    fi
    sleep 2
done

echo "PROGRESS:DONE"
SETUP_SCRIPT
        # Parse progress markers
        if [[ "$line" == PROGRESS:* ]]; then
            IFS=':' read -r _ step total message <<< "$line"
            if [[ "$step" == "DONE" ]]; then
                echo ""
            else
                progress_bar "$step" "$total" "$message                    "
            fi
        fi
    done

    # Save env file
    cat > "$ENV_FILE" << EOF
DROPLET_NAME=$DROPLET_NAME
SERVER_IP=$SERVER_IP
EOF

    echo ""
    echo "=============================================="
    success "Deployment Complete!"
    echo "=============================================="
    echo ""
    echo "  ComfyUI Web UI:  http://$SERVER_IP:8188"
    echo ""
    echo "  Text-to-Video (T2V):"
    echo "    ./generate-video -p \"your prompt\" -o output"
    echo ""
    echo "  Image-to-Video (I2V):"
    echo "    ./generate-video -p \"animate this\" --image photo.png -o output"
    echo ""
    echo "  Download videos:"
    echo "    scp root@$SERVER_IP:/opt/comfyui/ComfyUI/output/*.mp4 ./"
    echo ""
    echo "  View prompt history:"
    echo "    ssh root@$SERVER_IP cat /var/log/video-prompts.log"
    echo ""
    echo "  Cost: \$3.39/hour - destroy when not in use:"
    echo "    ./deploy.sh --destroy"
    echo ""
}

# Main
main() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║  WAN 2.2 NSFW Video Generation - Deployment Script        ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo ""

    check_prereqs

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --destroy|-d)
                destroy_droplet
                ;;
            --name|-n)
                DROPLET_NAME="$2"
                log "Using custom droplet name: $DROPLET_NAME"
                shift 2
                ;;
            *)
                # Assume it's an IP address
                get_or_create_droplet "$1"
                deploy
                exit 0
                ;;
        esac
    done

    # No IP provided, create/find droplet
    get_or_create_droplet ""
    deploy
}

main "$@"
