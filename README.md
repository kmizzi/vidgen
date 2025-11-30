# WAN 2.2 NSFW Video Generation Server

Generate uncensored AI videos using WAN 2.2 Rapid AIO on a DigitalOcean GPU Droplet.

## Features

- **Phr00t's Rapid AIO NSFW Model** - Fast, high-quality uncensored video generation
- **~10 second generation** - 2-second video in ~10 seconds on H100
- **CLI + Web UI** - Use `generate-video` command or ComfyUI interface
- **One-shot deployment** - Single script sets up everything

## Quick Start

### Prerequisites

```bash
# Install DigitalOcean CLI
brew install doctl  # macOS
# or: snap install doctl  # Linux

# Authenticate
doctl auth init
```

### Deploy

```bash
# One command to deploy everything (~15-20 min)
./deploy.sh
```

This will:
1. Create a GPU droplet (H100 80GB)
2. Install ComfyUI with CUDA
3. Download the NSFW model (~23GB)
4. Set up systemd service
5. Install CLI tools

### Generate Videos

**CLI (recommended):**
```bash
# SSH to server and generate
ssh root@<IP> generate-video -p "your prompt here" -o output_name

# Queue multiple jobs
ssh root@<IP> 'generate-video -p "prompt 1" -o v1 --queue && generate-video -p "prompt 2" -o v2 --queue'
```

**Web UI:**
```
http://<IP>:8188
```
Load the `nsfw_t2v_proper_workflow` workflow.

## CLI Reference

```bash
generate-video -p "prompt" [options]

Options:
  -p, --prompt     Text prompt (required)
  -o, --output     Output filename prefix (default: nsfw_output)
  -n, --negative   Negative prompt (default: "blurry, low quality, distorted")
  --width          Video width (default: 480)
  --height         Video height (default: 320)
  --frames         Number of frames (default: 33 = ~2 sec)
  --steps          Sampling steps (default: 4)
  --cfg            CFG scale (default: 1.0)
  --seed           Random seed
  --queue          Submit and exit without waiting
  --timeout        Timeout in seconds (default: 900)
```

### Examples

```bash
# Basic generation
generate-video -p "A woman dancing gracefully"

# Higher quality, longer video
generate-video -p "A couple walking on the beach" --frames 81 --steps 6

# Queue multiple jobs
generate-video -p "Scene 1" -o scene1 --queue
generate-video -p "Scene 2" -o scene2 --queue
generate-video -p "Scene 3" -o scene3 --queue
```

### Frame Count Reference

| Frames | Duration |
|--------|----------|
| 33 | ~2 sec |
| 49 | ~3 sec |
| 65 | ~4 sec |
| 81 | ~5 sec |
| 97 | ~6 sec |

## Recommended Settings

For Phr00t Rapid AIO model:
- **CFG:** 1.0 (required for distilled model)
- **Steps:** 4 (default, increase to 6-8 for more detail)
- **Sampler:** euler_ancestral
- **Scheduler:** beta
- **Shift:** 8.0 (handled automatically)

## Server Management

```bash
# Check status
ssh root@<IP> systemctl status comfyui

# View logs
ssh root@<IP> journalctl -u comfyui -f

# Restart ComfyUI
ssh root@<IP> systemctl restart comfyui

# View prompt history
ssh root@<IP> cat /var/log/video-prompts.log

# List output videos
ssh root@<IP> ls -la /opt/comfyui/ComfyUI/output/
```

## Download Videos

```bash
# Download specific video
scp root@<IP>:/opt/comfyui/ComfyUI/output/your_video.mp4 ./

# Download all videos
scp root@<IP>:/opt/comfyui/ComfyUI/output/*.mp4 ./
```

## Cost Management

| Resource | Hourly | Note |
|----------|--------|------|
| H100 80GB GPU | $3.39 | Destroy when not in use |

```bash
# Destroy droplet to stop charges
./deploy.sh --destroy

# Re-deploy later (model re-download required)
./deploy.sh
```

## Troubleshooting

### Videos show flashing images instead of smooth motion
Use the `nsfw_t2v_proper_workflow` which uses `WanImageToVideo` node. The `standard` workflow uses `EmptyLatentImage` which doesn't have temporal consistency.

### Out of VRAM
- Reduce resolution to 480x320
- Reduce frames to 33
- Model uses FP8 quantization (already optimized)

### Generation timeout
- Increase timeout: `--timeout 1200`
- Check GPU utilization: `ssh root@<IP> nvidia-smi`

### ComfyUI not responding
```bash
ssh root@<IP> systemctl restart comfyui
```

## File Structure

```
pn/
├── deploy.sh                    # One-shot deployment script
├── generate_video.py            # CLI tool source
├── nsfw_t2v_proper_workflow.json # ComfyUI workflow (use this one)
└── README.md
```

## Technical Details

- **Model:** Phr00t WAN 2.2 Rapid Mega AIO NSFW v12.1 (~23GB)
- **Architecture:** 14B parameter video diffusion model
- **Key Node:** `WanImageToVideo` - creates temporally coherent video latents
- **Output:** H.264 MP4 @ 16fps

## Resources

- [Phr00t Rapid AIO Model](https://huggingface.co/Phr00t/WAN2.2-14B-Rapid-AllInOne)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [DigitalOcean GPU Droplets](https://www.digitalocean.com/products/gpu-droplets)
