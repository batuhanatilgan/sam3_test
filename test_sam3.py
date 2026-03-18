"""
SAM3 Local Test Script - GPU & CPU
Teknopark test_image.jpg uzerinde SAM3 inference testi
"""

import argparse
import os
import time

import numpy as np
import torch
import contextlib
from PIL import Image


def run_inference(device_str: str, checkpoint_path: str, image_path: str, prompt: str, use_half: bool = False):
    """Run SAM3 inference on the given image with the given prompt."""
    print(f"\n{'='*60}")
    print(f"  SAM3 Inference - Device: {device_str.upper()}")
    print(f"{'='*60}")

    # Import SAM3 modules
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    # ---- Model Loading ----
    print(f"\n[1/3] Model yukleniyor ({device_str})...")
    t0 = time.time()
    model = build_sam3_image_model(
        device=device_str,
        eval_mode=True,
        checkpoint_path=checkpoint_path,
        load_from_HF=False,  # Lokal sam3.pt kullanilacak
    )
    t_model = time.time() - t0
    print(f"      Model yukleme suresi: {t_model:.2f} saniye")

    if use_half:
        print("      Model yariya indiriliyor (bfloat16)...")
        model = model.bfloat16()

    # ---- Image Loading ----
    print(f"\n[2/3] Gorsel yukleniyor: {image_path}")
    image = Image.open(image_path).convert("RGB")
    print(f"      Gorsel boyutu: {image.size}")

    processor = Sam3Processor(model, device=device_str)
    
    if use_half:
        from torchvision.transforms import v2
        # Patch the transform to output bfloat16 instead of float32
        new_transforms = []
        for t in processor.transform.transforms:
            if isinstance(t, v2.ToDtype) and t.dtype == torch.float32:
                new_transforms.append(v2.ToDtype(torch.bfloat16, scale=True))
            else:
                new_transforms.append(t)
        processor.transform = v2.Compose(new_transforms)

    # ---- Inference ----
    print(f"\n[3/3] Inference basliyor (prompt: '{prompt}')...")
    t1 = time.time()
    
    # Use autocast to handle Float32 to BFloat16 conversions seamlessly
    autocast_ctx = torch.autocast(device_type=device_str if device_str != "mps" else "cpu", dtype=torch.bfloat16) if use_half else contextlib.nullcontext()
    
    with torch.no_grad(), autocast_ctx:
        inference_state = processor.set_image(image)
        t_set_image = time.time() - t1

        t2 = time.time()
        output = processor.set_text_prompt(state=inference_state, prompt=prompt)
        t_prompt = time.time() - t2

    t_total_inference = time.time() - t1

    masks = output["masks"]
    boxes = output["boxes"]
    scores = output["scores"]

    print(f"\n{'-'*60}")
    print(f"  SONUCLAR ({device_str.upper()})")
    print(f"{'-'*60}")
    print(f"  Bulunan nesne sayisi  : {len(masks)}")
    print(f"  set_image suresi      : {t_set_image:.2f} s")
    print(f"  set_text_prompt suresi: {t_prompt:.2f} s")
    print(f"  Toplam inference      : {t_total_inference:.2f} s")
    print(f"  Model yukleme         : {t_model:.2f} s")

    if len(scores) > 0:
        print(f"\n  Skorlar:")
        for i, (score, box) in enumerate(zip(scores, boxes)):
            s = score.item() if hasattr(score, "item") else float(score)
            b = box.tolist() if hasattr(box, "tolist") else list(box)
            print(f"    [{i}] Skor: {s:.4f} | Box: {[f'{v:.1f}' for v in b]}")

    # ---- Save Results ----
    save_results(image, masks, boxes, scores, device_str, image_path, prompt)

    # Cleanup GPU memory
    del model, processor, inference_state, output
    if device_str == "cuda":
        torch.cuda.empty_cache()

    return {
        "device": device_str,
        "num_objects": len(masks),
        "model_load_time": t_model,
        "set_image_time": t_set_image,
        "prompt_time": t_prompt,
        "total_inference_time": t_total_inference,
    }


def save_results(image, masks, boxes, scores, device_str, image_path, prompt):
    """Save inference results as an overlay image."""
    try:
        import cv2
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("  [!] matplotlib/opencv yuklu degil, sonuc gorseli kaydedilemedi.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(np.array(image))

    # Draw masks
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(masks), 1)))
    for i, mask in enumerate(masks):
        if hasattr(mask, "cpu"):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = np.array(mask)

        # Handle multi-dimensional masks
        if mask_np.ndim > 2:
            mask_np = mask_np.squeeze()
        if mask_np.ndim > 2:
            mask_np = mask_np[0]

        color = colors[i % len(colors)]
        colored_mask = np.zeros((*mask_np.shape, 4))
        colored_mask[mask_np > 0.5] = [*color[:3], 0.4]
        ax.imshow(colored_mask)

    # Draw boxes
    for i, box in enumerate(boxes):
        if hasattr(box, "cpu"):
            box_np = box.cpu().numpy()
        else:
            box_np = np.array(box)

        x1, y1, x2, y2 = box_np[:4]
        color = colors[i % len(colors)]
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color[:3], facecolor="none"
        )
        ax.add_patch(rect)

        score_val = scores[i].item() if hasattr(scores[i], "item") else float(scores[i])
        ax.text(
            x1, y1 - 5, f"{score_val:.2f}",
            color="white", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color[:3], alpha=0.8),
        )

    ax.set_title(f"SAM3 | Device: {device_str.upper()} | Prompt: '{prompt}' | {len(masks)} objects", fontsize=14)
    ax.axis("off")

    out_dir = os.path.dirname(image_path)
    out_path = os.path.join(out_dir, f"sam3_result_{device_str}.jpg")
    fig.savefig(out_path, bbox_inches="tight", dpi=150, pad_inches=0.1)
    plt.close(fig)
    print(f"  Sonuc gorseli: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="SAM3 Local Test - GPU & CPU")
    parser.add_argument(
        "--device", type=str, default="both",
        choices=["cuda", "cpu", "both"],
        help="Test cihazi: cuda, cpu, veya both (varsayilan: both)"
    )
    parser.add_argument(
        "--image", type=str,
        default=os.path.join(os.path.dirname(__file__), "teknopark", "test_image.jpg"),
        help="Test gorseli yolu"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default=os.path.join(os.path.dirname(__file__), "sam3.pt"),
        help="SAM3 checkpoint dosyasi yolu"
    )
    parser.add_argument(
        "--prompt", type=str, default="car",
        help="Text prompt (varsayilan: car)"
    )
    parser.add_argument(
        "--half", action="store_true",
        help="Modeli bfloat16 veya float16 olarak yukleyip VRAM kullanimini yariya indirir"
    )
    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.image):
        print(f"HATA: Gorsel bulunamadi: {args.image}")
        return
    if not os.path.exists(args.checkpoint):
        print(f"HATA: Checkpoint bulunamadi: {args.checkpoint}")
        return

    print(f"\nSAM3 Local Test")
    print(f"  Gorsel     : {args.image}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Prompt     : {args.prompt}")
    print(f"  PyTorch    : {torch.__version__}")
    print(f"  CUDA mevcut: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM       : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    results = []

    devices_to_test = []
    if args.device == "both":
        if torch.cuda.is_available():
            devices_to_test = ["cuda", "cpu"]
        else:
            print("\n[!] CUDA mevcut degil, sadece CPU testi yapilacak.")
            devices_to_test = ["cpu"]
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            print("\n[!] CUDA mevcut degil, CPU'ya geciliyor.")
            devices_to_test = ["cpu"]
        else:
            devices_to_test = [args.device]

    for dev in devices_to_test:
        try:
            result = run_inference(dev, args.checkpoint, args.image, args.prompt, args.half)
            results.append(result)
        except torch.cuda.OutOfMemoryError:
            print(f"\n  [!] CUDA OUT OF MEMORY! GPU VRAM yetersiz (4GB).")
            print(f"      GPU testi basarisiz — CPU testi ile devam ediliyor.")
            torch.cuda.empty_cache()
            if "cpu" not in devices_to_test:
                try:
                    result = run_inference("cpu", args.checkpoint, args.image, args.prompt, args.half)
                    results.append(result)
                except Exception as e:
                    print(f"\n  [!] CPU testi de basarisiz: {e}")
        except Exception as e:
            print(f"\n  [!] {dev.upper()} testi basarisiz: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if len(results) > 0:
        print(f"\n\n{'='*60}")
        print(f"  OZET KARSILASTIRMA")
        print(f"{'='*60}")
        print(f"  {'Device':<8} {'Objects':<10} {'Model Load':<14} {'set_image':<14} {'Prompt':<14} {'Total Inf.':<14}")
        print(f"  {'-'*8} {'-'*10} {'-'*14} {'-'*14} {'-'*14} {'-'*14}")
        for r in results:
            print(
                f"  {r['device'].upper():<8} "
                f"{r['num_objects']:<10} "
                f"{r['model_load_time']:<14.2f} "
                f"{r['set_image_time']:<14.2f} "
                f"{r['prompt_time']:<14.2f} "
                f"{r['total_inference_time']:<14.2f}"
            )

    print(f"\nTest tamamlandi!")


if __name__ == "__main__":
    main()
