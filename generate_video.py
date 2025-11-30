#!/usr/bin/env python3
"""
CLI script for generating NSFW videos using WAN 2.2 Rapid AIO model via ComfyUI API.

Usage:
    python generate_video.py -p "your prompt here" -o output.mp4
    python generate_video.py -p "A woman dancing gracefully" --width 720 --height 480 --frames 65

Requires ComfyUI to be running on the server.
"""
import argparse
import json
import urllib.request
import urllib.error
import time
import sys
import random
import os

DEFAULT_SERVER = "http://localhost:8188"


def build_workflow(prompt, negative_prompt, width, height, frames, steps, cfg, seed, filename_prefix):
    """Build proper T2V workflow using native WanImageToVideo for temporal consistency.

    The WanImageToVideo node creates a proper 3D video latent tensor that maintains
    temporal coherence across frames, unlike EmptyLatentImage which creates independent 2D frames.

    Workflow: CheckpointLoader -> ModelSamplingSD3 (shift=8) -> CLIPTextEncode x2 ->
              WanImageToVideo -> KSampler -> VAEDecode -> VHS_VideoCombine
    """
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "wan2.2-rapid-mega-aio-nsfw-v12.1.safetensors"
            }
        },
        "2": {
            "class_type": "ModelSamplingSD3",
            "inputs": {
                "model": ["1", 0],
                "shift": 8.0
            }
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["1", 1]
            }
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_prompt,
                "clip": ["1", 1]
            }
        },
        "5": {
            "class_type": "WanImageToVideo",
            "inputs": {
                "positive": ["3", 0],
                "negative": ["4", 0],
                "vae": ["1", 2],
                "width": width,
                "height": height,
                "length": frames,
                "batch_size": 1
            }
        },
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["2", 0],
                "positive": ["5", 0],
                "negative": ["5", 1],
                "latent_image": ["5", 2],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler_ancestral",
                "scheduler": "beta",
                "denoise": 1.0
            }
        },
        "7": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["6", 0],
                "vae": ["1", 2]
            }
        },
        "8": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["7", 0],
                "frame_rate": 16,
                "loop_count": 0,
                "filename_prefix": filename_prefix,
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": True,
                "pingpong": False,
                "save_output": True
            }
        }
    }


def queue_prompt(server, prompt_data):
    """Submit prompt to ComfyUI"""
    data = json.dumps({"prompt": prompt_data}).encode("utf-8")
    req = urllib.request.Request(
        f"{server}/prompt",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        print(f"HTTP Error {e.code}: {error_body}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def get_history(server, prompt_id):
    """Get execution history"""
    try:
        resp = urllib.request.urlopen(f"{server}/history/{prompt_id}", timeout=10)
        return json.loads(resp.read())
    except Exception:
        return {}


def get_queue(server):
    """Get current queue status"""
    try:
        resp = urllib.request.urlopen(f"{server}/queue", timeout=10)
        return json.loads(resp.read())
    except Exception:
        return {}


def wait_for_completion(server, prompt_id, timeout=900):
    """Wait for workflow to complete"""
    start_time = time.time()
    last_status = ""

    while True:
        elapsed = time.time() - start_time

        # Check history for completion
        history = get_history(server, prompt_id)

        if prompt_id in history:
            outputs = history[prompt_id].get("outputs", {})
            status = history[prompt_id].get("status", {})

            if status.get("status_str") == "error":
                print(f"\nExecution failed after {elapsed:.1f}s!")
                messages = status.get("messages", [])
                for msg in messages:
                    print(f"  {msg}")
                return None

            if outputs:
                print(f"\nCompleted in {elapsed:.1f}s!")
                return outputs

        # Show progress
        queue = get_queue(server)
        running = queue.get("queue_running", [])
        pending = queue.get("queue_pending", [])

        status_msg = f"Elapsed: {elapsed:.0f}s"
        if running:
            status_msg += f" | Running: {len(running)}"
        if pending:
            status_msg += f" | Pending: {len(pending)}"

        if status_msg != last_status:
            print(f"\r{status_msg}        ", end="", flush=True)
            last_status = status_msg

        if elapsed > timeout:
            print("\nTimeout!")
            return None

        time.sleep(2)


def main():
    parser = argparse.ArgumentParser(
        description="Generate NSFW video using WAN 2.2 Rapid AIO model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -p "A woman dancing in the rain"
  %(prog)s -p "A couple walking on the beach" -o beach.mp4
  %(prog)s -p "A dancer performing" --width 720 --height 480 --frames 65
  %(prog)s -p "Your prompt" --steps 6 --cfg 1.5 --seed 42

  # Queue multiple jobs quickly:
  %(prog)s -p "Prompt 1" -o video1 --queue
  %(prog)s -p "Prompt 2" -o video2 --queue
  %(prog)s -p "Prompt 3" -o video3 --queue

Notes:
  - Recommended settings for Rapid AIO: CFG=1, Steps=4
  - Higher resolution/frames = longer generation time
  - Output saved to ComfyUI output folder by default
        """
    )

    parser.add_argument("-p", "--prompt", required=True,
                        help="Text prompt for video generation")
    parser.add_argument("-n", "--negative", default="blurry, low quality, distorted, deformed, static, frozen, flickering, morphing, ugly, bad anatomy, extra limbs, missing limbs, watermark, text, logo",
                        help="Negative prompt (default: quality filters)")
    parser.add_argument("-o", "--output", default="nsfw_output",
                        help="Output filename prefix - just the name, no path (default: nsfw_output)")
    parser.add_argument("--width", type=int, default=480,
                        help="Video width (default: 480)")
    parser.add_argument("--height", type=int, default=320,
                        help="Video height (default: 320)")
    parser.add_argument("--frames", type=int, default=100,
                        help="Number of frames (default: 100 = ~6 sec at 16fps)")
    parser.add_argument("--steps", type=int, default=8,
                        help="Sampling steps (default: 8)")
    parser.add_argument("--cfg", type=float, default=1.0,
                        help="CFG scale (default: 1.0, recommended for Rapid AIO)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: random)")
    parser.add_argument("--server", default=DEFAULT_SERVER,
                        help=f"ComfyUI server URL (default: {DEFAULT_SERVER})")
    parser.add_argument("--timeout", type=int, default=900,
                        help="Timeout in seconds (default: 900)")
    parser.add_argument("--queue", action="store_true",
                        help="Queue only - submit and exit without waiting for completion")

    args = parser.parse_args()

    # Generate random seed if not provided
    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)

    # Strip path from output - ComfyUI only accepts filename prefix
    args.output = os.path.basename(args.output)
    # Remove extension if provided
    if args.output.endswith('.mp4'):
        args.output = args.output[:-4]

    # Print banner
    sep = "=" * 60
    print(sep)
    print("WAN 2.2 NSFW Video Generator")
    print(sep)
    print(f"Prompt:     {args.prompt[:50]}{'...' if len(args.prompt) > 50 else ''}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Frames:     {args.frames} (~{args.frames/16:.1f}s at 16fps)")
    print(f"Settings:   CFG={args.cfg}, Steps={args.steps}, Seed={args.seed}")
    print(f"Server:     {args.server}")
    print(sep)
    print()

    # Build workflow
    workflow = build_workflow(
        prompt=args.prompt,
        negative_prompt=args.negative,
        width=args.width,
        height=args.height,
        frames=args.frames,
        steps=args.steps,
        cfg=args.cfg,
        seed=args.seed,
        filename_prefix=args.output
    )

    # Submit to ComfyUI
    print("Submitting workflow...")
    result = queue_prompt(args.server, workflow)

    if result is None:
        print("Failed to submit workflow!")
        sys.exit(1)

    if "error" in result:
        print(f"Error: {result['error']}")
        if "node_errors" in result:
            for node_id, errors in result["node_errors"].items():
                print(f"  Node {node_id}: {errors}")
        sys.exit(1)

    prompt_id = result.get("prompt_id")
    print(f"Queued with ID: {prompt_id}")

    # Log prompt to history file
    log_file = "/var/log/video-prompts.log"
    try:
        import datetime
        with open(log_file, "a") as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} | {prompt_id[:8]} | {args.output} | {args.prompt}\n")
    except Exception:
        pass  # Don't fail if logging fails

    # If queue-only mode, exit now
    if args.queue:
        print()
        print(f"Job queued successfully. Output will be saved as: {args.output}_*.mp4")
        print(f"Check progress at: {args.server}")
        sys.exit(0)

    print()
    print("Generating video...")

    # Wait for completion
    outputs = wait_for_completion(args.server, prompt_id, args.timeout)

    if outputs is None:
        sys.exit(1)

    # Print output info
    print()
    print(sep)
    print("Generation Complete!")
    print(sep)

    for node_id, output in outputs.items():
        if "gifs" in output:
            for gif in output["gifs"]:
                filename = gif.get("filename", "unknown")
                subfolder = gif.get("subfolder", "")
                print(f"Output: {subfolder}/{filename}" if subfolder else f"Output: {filename}")
                print(f"Location: /opt/comfyui/ComfyUI/output/{filename}")

    print(sep)


if __name__ == "__main__":
    main()
