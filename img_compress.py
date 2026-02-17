import sys
import os
from PIL import Image
from pathlib import Path

def compress_image_to_target(input_path, output_path, target_size_mb=10):
    target_size = target_size_mb * 1024 * 1024  # MB → bytes
    
    img = Image.open(input_path)

    # Convert to RGB for JPEG compatibility
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    quality = 95  # Start high
    img.save(output_path, "JPEG", quality=quality, optimize=True)

    while os.path.getsize(output_path) > target_size and quality > 10:
        quality -= 5
        img.save(output_path, "JPEG", quality=quality, optimize=True)

    final_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✔ Compression Done!")
    print(f"Saved to: {output_path}")
    print(f"Final quality: {quality}")
    print(f"Final size: {final_size:.2f} MB")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compress.py <input_image> [target_size_mb]")
        sys.exit(1)

    input_image = sys.argv[1]

    # Optional target size (default 10MB)
    target_size = float(sys.argv[2]) if len(sys.argv) > 2 else 10

    # Ask user where to save
    print("Enter full path to save compressed image (Press Enter for default Downloads folder):")
    save_path = input().strip()

    if not save_path:
        # Default to system Downloads folder
        downloads = Path.home() / "Downloads"
        save_path = downloads / ("compressed_" + os.path.basename(input_image))
    else:
        # If user enters a folder, append filename
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, "compressed_" + os.path.basename(input_image))

    compress_image_to_target(input_image, save_path, target_size)
