#!/usr/bin/env python3
"""
CLI script for generating NSFW content using WAN 2.2 and SDXL models via ComfyUI API.

Usage:
    # Text-to-Video (T2V):
    python generate_video.py -p "your prompt here" -o output
    python generate_video.py -p "A woman dancing gracefully" --width 720 --height 480 --frames 65

    # Image-to-Video (I2V):
    python generate_video.py -p "animate this person dancing" --image input.png -o output

    # Image-to-Image (I2I) - Identity-preserving transform:
    python generate_video.py -p "nude, realistic photo" --image photo.png --mode i2i -o output

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
import mimetypes

DEFAULT_SERVER = "http://localhost:8188"


def upload_image(server, image_path):
    """Upload an image to ComfyUI's input folder.

    Returns the filename that ComfyUI uses to reference the image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    filename = os.path.basename(image_path)
    content_type = mimetypes.guess_type(image_path)[0] or 'image/png'

    # Build multipart form data
    boundary = '----WebKitFormBoundary' + ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=16))

    with open(image_path, 'rb') as f:
        file_data = f.read()

    body = b''
    # Add file field
    body += f'--{boundary}\r\n'.encode()
    body += f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'.encode()
    body += f'Content-Type: {content_type}\r\n\r\n'.encode()
    body += file_data
    body += b'\r\n'
    # Add overwrite field
    body += f'--{boundary}\r\n'.encode()
    body += b'Content-Disposition: form-data; name="overwrite"\r\n\r\n'
    body += b'true\r\n'
    body += f'--{boundary}--\r\n'.encode()

    req = urllib.request.Request(
        f"{server}/upload/image",
        data=body,
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
    )

    try:
        resp = urllib.request.urlopen(req, timeout=60)
        result = json.loads(resp.read())
        return result.get('name', filename)
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        raise Exception(f"Upload failed: HTTP {e.code}: {error_body}")
    except Exception as e:
        raise Exception(f"Upload failed: {e}")


def build_i2v_workflow(prompt, negative_prompt, image_filename, width, height, frames, steps, cfg, seed, filename_prefix):
    """Build I2V (Image-to-Video) workflow using official WAN 2.2 I2V models with two-stage sampling.

    This workflow uses the official WAN 2.2 I2V architecture with:
    - High-noise model for initial steps (0-50% of steps)
    - Low-noise model for remaining steps (50-100%)
    - UMT5-XXL text encoder (not CLIP)
    - WAN 2.1 VAE
    - CLIP Vision for image conditioning

    This two-stage approach significantly improves identity preservation from the input image.
    """
    # Calculate step split (50% for each stage)
    stage1_end = steps // 2
    stage2_start = stage1_end

    return {
        "1": {
            "class_type": "LoadImage",
            "inputs": {
                "image": image_filename
            }
        },
        "2": {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["1", 0],
                "upscale_method": "lanczos",
                "width": width,
                "height": height,
                "crop": "disabled"
            }
        },
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                "type": "wan"
            }
        },
        "4": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "wan_2.1_vae.safetensors"
            }
        },
        "5": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
                "weight_dtype": "default"
            }
        },
        "6": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
                "weight_dtype": "default"
            }
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["3", 0]
            }
        },
        "8": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_prompt,
                "clip": ["3", 0]
            }
        },
        "9": {
            "class_type": "CLIPVisionLoader",
            "inputs": {
                "clip_name": "sigclip_vision_patch14_384.safetensors"
            }
        },
        "10": {
            "class_type": "CLIPVisionEncode",
            "inputs": {
                "clip_vision": ["9", 0],
                "image": ["2", 0]
            }
        },
        "11": {
            "class_type": "WanImageToVideo",
            "inputs": {
                "positive": ["7", 0],
                "negative": ["8", 0],
                "vae": ["4", 0],
                "clip_vision_output": ["10", 0],
                "start_image": ["2", 0],
                "width": width,
                "height": height,
                "length": frames,
                "batch_size": 1
            }
        },
        "12": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["5", 0],
                "add_noise": "enable",
                "noise_seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "beta",
                "positive": ["11", 0],
                "negative": ["11", 1],
                "latent_image": ["11", 2],
                "start_at_step": 0,
                "end_at_step": stage1_end,
                "return_with_leftover_noise": "enable"
            }
        },
        "13": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["6", 0],
                "add_noise": "disable",
                "noise_seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "beta",
                "positive": ["11", 0],
                "negative": ["11", 1],
                "latent_image": ["12", 0],
                "start_at_step": stage2_start,
                "end_at_step": steps,
                "return_with_leftover_noise": "disable"
            }
        },
        "14": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["13", 0],
                "vae": ["4", 0]
            }
        },
        "15": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["14", 0],
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


def build_i2i_workflow(prompt, negative_prompt, image_filename, width, height, steps, cfg, seed, filename_prefix):
    """Build I2I (Image-to-Image) workflow using IP-Adapter FaceID for identity preservation.

    This workflow uses RealVisXL SDXL with IP-Adapter FaceID Plus V2 to:
    - Extract face identity from input image
    - Generate new image based on prompt while preserving identity
    - Perfect for transforming images while keeping the same person

    Flow:
    LoadImage -> IPAdapterFaceID -> SDXL -> VAEDecode -> SaveImage
    """
    return {
        "1": {
            "class_type": "LoadImage",
            "inputs": {
                "image": image_filename
            }
        },
        "2": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "realvisxl_v50.safetensors"
            }
        },
        "3": {
            "class_type": "IPAdapterUnifiedLoaderFaceID",
            "inputs": {
                "model": ["2", 0],
                "preset": "FACEID PLUS V2",
                "lora_strength": 0.85,
                "provider": "CUDA"
            }
        },
        "4": {
            "class_type": "IPAdapterFaceID",
            "inputs": {
                "model": ["3", 0],
                "ipadapter": ["3", 1],
                "image": ["1", 0],
                "weight": 0.85,
                "weight_faceidv2": 0.5,
                "weight_type": "linear",
                "combine_embeds": "concat",
                "start_at": 0.0,
                "end_at": 1.0,
                "embeds_scaling": "V only"
            }
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["2", 1]
            }
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_prompt,
                "clip": ["2", 1]
            }
        },
        "7": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1
            }
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["4", 0],
                "positive": ["5", 0],
                "negative": ["6", 0],
                "latent_image": ["7", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "denoise": 1.0
            }
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["8", 0],
                "vae": ["2", 2]
            }
        },
        "10": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["9", 0],
                "filename_prefix": filename_prefix
            }
        }
    }


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
  # Text-to-Video:
  %(prog)s -p "A woman dancing in the rain"
  %(prog)s -p "A couple walking on the beach" -o beach.mp4
  %(prog)s -p "A dancer performing" --width 720 --height 480 --frames 65

  # Image-to-Video:
  %(prog)s -p "animate this person dancing" --image input.png -o output
  %(prog)s -p "make her smile and wave" -i photo.jpg --frames 50

  # Queue multiple jobs:
  %(prog)s -p "Prompt 1" -o video1 --queue
  %(prog)s -p "Prompt 2" -o video2 --queue

Modes:
  - T2V (default): Text-to-Video using WAN 2.2
  - I2V: Image-to-Video (--image required, default when image provided)
  - I2I: Image-to-Image with identity preservation (--mode i2i --image)

Notes:
  - I2I mode preserves face identity while transforming the image
  - Output saved to ComfyUI output folder
        """
    )

    parser.add_argument("-p", "--prompt", required=True,
                        help="Text prompt for generation")
    parser.add_argument("-i", "--image", default=None,
                        help="Input image for I2V or I2I mode (optional)")
    parser.add_argument("-m", "--mode", default=None, choices=["t2v", "i2v", "i2i"],
                        help="Generation mode: t2v (text-to-video), i2v (image-to-video), i2i (image-to-image). Auto-detected if not specified.")
    parser.add_argument("-n", "--negative", default=None,
                        help="Negative prompt (auto-set based on mode if not specified)")
    parser.add_argument("-o", "--output", default="nsfw_output",
                        help="Output filename prefix - just the name, no path (default: nsfw_output)")
    parser.add_argument("--width", type=int, default=480,
                        help="Video width (default: 480)")
    parser.add_argument("--height", type=int, default=320,
                        help="Video height (default: 320)")
    parser.add_argument("--frames", type=int, default=100,
                        help="Number of frames (default: 100 = ~6 sec at 16fps)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Sampling steps (default: 8 for T2V, 20 for I2V)")
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

    # Determine mode
    if args.mode:
        mode = args.mode
    elif args.image:
        mode = "i2v"  # Default to I2V when image provided
    else:
        mode = "t2v"

    # Validate mode requirements
    if mode in ["i2v", "i2i"] and not args.image:
        print(f"Error: --image is required for {mode.upper()} mode")
        sys.exit(1)

    # Generate random seed if not provided
    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)

    # Set mode-specific defaults
    if mode == "i2i":
        # I2I defaults: SDXL resolution, 25 steps, CFG 7
        if args.width == 480:
            args.width = 1024
        if args.height == 320:
            args.height = 1024
        if args.steps is None:
            args.steps = 25
        if args.cfg == 1.0:
            args.cfg = 7.0
        if args.negative is None:
            args.negative = "ugly, deformed, noisy, blurry, low quality, cartoon, anime, 3d render, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, mutated hands, poorly drawn face, poorly drawn hands"
    elif mode == "i2v":
        if args.steps is None:
            args.steps = 20
        if args.negative is None:
            args.negative = "blurry, low quality, distorted, deformed, static, frozen, flickering, morphing, ugly, bad anatomy, extra limbs, missing limbs, watermark, text, logo, different person, changing face"
    else:  # t2v
        if args.steps is None:
            args.steps = 8
        if args.negative is None:
            args.negative = "blurry, low quality, distorted, deformed, static, frozen, flickering, morphing, ugly, bad anatomy, extra limbs, missing limbs, watermark, text, logo"

    # Strip path from output - ComfyUI only accepts filename prefix
    args.output = os.path.basename(args.output)
    # Remove extension if provided
    if args.output.endswith('.mp4') or args.output.endswith('.png'):
        args.output = args.output[:-4]

    # Set mode string for display
    mode_names = {
        "t2v": "Text-to-Video (T2V)",
        "i2v": "Image-to-Video (I2V)",
        "i2i": "Image-to-Image (I2I)"
    }
    mode_str = mode_names[mode]

    # Print banner
    sep = "=" * 60
    print(sep)
    print(f"NSFW Generator - {mode_str}")
    print(sep)
    if mode in ["i2v", "i2i"]:
        print(f"Image:      {args.image}")
    print(f"Prompt:     {args.prompt[:50]}{'...' if len(args.prompt) > 50 else ''}")
    print(f"Resolution: {args.width}x{args.height}")
    if mode != "i2i":
        print(f"Frames:     {args.frames} (~{args.frames/16:.1f}s at 16fps)")
    print(f"Settings:   CFG={args.cfg}, Steps={args.steps}, Seed={args.seed}")
    print(f"Server:     {args.server}")
    print(sep)
    print()

    # Handle image upload for I2V/I2I modes
    image_filename = None
    if mode in ["i2v", "i2i"]:
        print(f"Uploading image: {args.image}...")
        try:
            image_filename = upload_image(args.server, args.image)
            print(f"Image uploaded as: {image_filename}")
        except Exception as e:
            print(f"Failed to upload image: {e}")
            sys.exit(1)
        print()

    # Build workflow based on mode
    if mode == "i2i":
        workflow = build_i2i_workflow(
            prompt=args.prompt,
            negative_prompt=args.negative,
            image_filename=image_filename,
            width=args.width,
            height=args.height,
            steps=args.steps,
            cfg=args.cfg,
            seed=args.seed,
            filename_prefix=args.output
        )
    elif mode == "i2v":
        workflow = build_i2v_workflow(
            prompt=args.prompt,
            negative_prompt=args.negative,
            image_filename=image_filename,
            width=args.width,
            height=args.height,
            frames=args.frames,
            steps=args.steps,
            cfg=args.cfg,
            seed=args.seed,
            filename_prefix=args.output
        )
    else:
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
            mode_tag = mode.upper()
            f.write(f"{timestamp} | {mode_tag} | {prompt_id[:8]} | {args.output} | {args.prompt}\n")
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
