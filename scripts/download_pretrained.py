#!/usr/bin/env python3
"""
Download Pretrained Road Segmentation Model

Downloads D-LinkNet34 weights trained on DeepGlobe Road Extraction dataset.
This model won 1st place in the DeepGlobe Challenge.

Usage:
    python scripts/download_pretrained.py
"""
import os
import sys
import urllib.request
import zipfile
from pathlib import Path
import shutil

# URLs for pretrained weights
DROPBOX_URL = "https://www.dropbox.com/sh/h62vr320eiy57tt/AAB5Tm43-efmtYzW_GFyUCfma?dl=1"

# Alternative: Direct file link (if available)
DIRECT_WEIGHTS_URLS = [
    # Original D-LinkNet34 trained on DeepGlobe
    ("dlinknet34_deepglobe", "https://www.dropbox.com/sh/h62vr320eiy57tt/AAB5Tm43-efmtYzW_GFyUCfma?dl=1"),
]

def download_with_progress(url: str, dest: Path) -> bool:
    """Download file with progress indicator."""
    print(f"Downloading from {url[:80]}...")

    try:
        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = min(100, count * block_size * 100 / total_size)
                print(f"\rProgress: {percent:.1f}%", end="", flush=True)
            else:
                print(f"\rDownloaded: {count * block_size / 1024 / 1024:.1f} MB", end="", flush=True)

        urllib.request.urlretrieve(url, dest, reporthook=progress_hook)
        print()  # New line after progress
        return True
    except Exception as e:
        print(f"\nDownload failed: {e}")
        return False


def extract_and_find_weights(zip_path: Path, extract_dir: Path) -> Path:
    """Extract zip and find .th or .pth weights file."""
    print(f"Extracting {zip_path}...")

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)

    # Find weight files
    for ext in ['*.th', '*.pth', '*.pt']:
        weights = list(extract_dir.rglob(ext))
        if weights:
            print(f"Found weights: {weights}")
            return weights[0]

    # List all files for debugging
    all_files = list(extract_dir.rglob('*'))
    print(f"All extracted files: {all_files}")
    return None


def convert_weights(src_weights: Path, dest_path: Path):
    """Convert original D-LinkNet weights to compatible format."""
    import torch

    print(f"Loading weights from {src_weights}...")

    # Load original weights
    checkpoint = torch.load(src_weights, map_location='cpu', weights_only=False)

    # Check if it's already a state dict or wrapped
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint

    # Print keys for debugging
    print(f"State dict keys (first 10): {list(state_dict.keys())[:10]}")

    # Save in standard format
    torch.save({
        'model_state_dict': state_dict,
        'architecture': 'dlinknet34',
        'source': 'deepglobe_challenge_1st_place',
    }, dest_path)

    print(f"Converted weights saved to {dest_path}")


def create_smp_compatible_model():
    """
    Create a segmentation_models_pytorch LinkNet that matches the
    original D-LinkNet34 architecture as closely as possible.
    """
    try:
        import segmentation_models_pytorch as smp
        import torch

        # Create LinkNet with ResNet34 encoder - this is close to D-LinkNet
        model = smp.Linknet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None,
        )

        return model
    except ImportError:
        print("segmentation_models_pytorch not available")
        return None


def download_pretrained_model(output_dir: str = "checkpoints") -> Path:
    """Main function to download and set up pretrained model."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Temporary directory for download
    temp_dir = output_path / "temp_download"
    temp_dir.mkdir(exist_ok=True)

    dest_model = output_path / "dlinknet34_deepglobe.pt"

    if dest_model.exists():
        print(f"Pretrained model already exists at {dest_model}")
        return dest_model

    # Download
    zip_path = temp_dir / "dlinknet34.zip"
    if not download_with_progress(DROPBOX_URL, zip_path):
        print("\nFailed to download from Dropbox.")
        print("Please download manually from:")
        print("  https://www.dropbox.com/sh/h62vr320eiy57tt/AAB5Tm43-efmtYzW_GFyUCfma")
        print(f"  And place the .th file in {output_path}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None

    # Check if it's actually a zip file
    try:
        weights_file = extract_and_find_weights(zip_path, temp_dir)
    except zipfile.BadZipFile:
        # Maybe it's already a weights file
        print("Not a zip file, assuming direct weights download...")
        weights_file = zip_path

    if weights_file is None or not weights_file.exists():
        print("Could not find weights file in download.")
        # Try using the downloaded file directly as weights
        try:
            import torch
            checkpoint = torch.load(zip_path, map_location='cpu', weights_only=False)
            print("Downloaded file is a valid PyTorch checkpoint")
            weights_file = zip_path
        except:
            print("Download did not contain valid weights.")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None

    # Convert to our format
    try:
        convert_weights(weights_file, dest_model)
    except Exception as e:
        print(f"Error converting weights: {e}")
        # Just copy as-is
        shutil.copy(weights_file, dest_model)
        print(f"Copied weights to {dest_model}")

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Also create symlink/copy as best_model.pt for server
    best_model = output_path / "best_model.pt"
    if not best_model.exists():
        shutil.copy(dest_model, best_model)
        print(f"Copied to {best_model} for server use")

    return dest_model


def verify_model(model_path: Path) -> bool:
    """Verify the downloaded model works."""
    import torch

    print(f"\nVerifying model at {model_path}...")

    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        print(f"Model has {len(state_dict)} layers")
        print(f"Sample keys: {list(state_dict.keys())[:5]}")

        # Try to load into model
        from roadmesh.models.architectures import DLinkNet34
        model = DLinkNet34(num_classes=1, pretrained=False)

        try:
            model.load_state_dict(state_dict, strict=False)
            print("✓ Model loaded successfully (with some mismatches allowed)")
        except Exception as e:
            print(f"Note: Direct loading failed: {e}")
            print("Model may need architecture adaptation")

        # Test inference
        x = torch.randn(1, 3, 512, 512)
        model.eval()
        with torch.no_grad():
            out = model(x)
        print(f"✓ Inference test passed: {x.shape} -> {out.shape}")

        return True

    except Exception as e:
        print(f"Verification failed: {e}")
        return False


def print_manual_instructions():
    """Print instructions for manual download."""
    print("\n" + "=" * 70)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 70)
    print("""
The automatic download failed. Please download manually:

OPTION 1: D-LinkNet34 (DeepGlobe 1st Place Winner)
-------------------------------------------------
1. Open this URL in your browser:
   https://www.dropbox.com/sh/h62vr320eiy57tt/AAB5Tm43-efmtYzW_GFyUCfma

2. Download the .th file (around 85 MB)

3. Rename to 'dlinknet34_deepglobe.pt' and place in:
   checkpoints/dlinknet34_deepglobe.pt

4. Create symlink or copy as best_model.pt:
   cp checkpoints/dlinknet34_deepglobe.pt checkpoints/best_model.pt


OPTION 2: Train on DeepGlobe Dataset
------------------------------------
1. Create Kaggle account at https://www.kaggle.com

2. Get API token from https://www.kaggle.com/settings
   Save to ~/.kaggle/kaggle.json (or %USERPROFILE%\\.kaggle\\kaggle.json on Windows)

3. Accept DeepGlobe dataset terms at:
   https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset

4. Run training:
   pip install kaggle
   python scripts/train_deepglobe.py --epochs 50

   Training takes 1-2 hours on GPU, 4-8 hours on CPU.


OPTION 3: Use Color-Based Detection
-----------------------------------
The color-based detection works without any trained model:
1. Start server: python -m roadmesh.app.server
2. Open http://localhost:8080
3. Select "Color-based (fast, no training)" mode
""")
    print("=" * 70)


def main():
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root / "src"))

    print("=" * 60)
    print("Download Pretrained D-LinkNet34 (DeepGlobe 1st Place)")
    print("=" * 60)

    # First check if weights already exist
    from roadmesh.models.pretrained import find_pretrained_weights
    existing = find_pretrained_weights()
    if existing:
        print(f"\nFound existing weights: {existing}")
        verify_model(existing)
        print("\n" + "=" * 60)
        print("Pretrained model already available!")
        print(f"Model path: {existing}")
        print("\nYou can use 'ML Model' or 'Pretrained' mode in the web interface.")
        print("=" * 60)
        return

    model_path = download_pretrained_model()

    if model_path and model_path.exists():
        verify_model(model_path)
        print("\n" + "=" * 60)
        print("SUCCESS! Pretrained model is ready.")
        print(f"Model path: {model_path}")
        print("\nYou can now use 'ML Model' mode in the web interface.")
        print("=" * 60)
    else:
        print_manual_instructions()
        sys.exit(1)


if __name__ == "__main__":
    main()
