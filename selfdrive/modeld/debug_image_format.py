#!/usr/bin/env python3
"""
Debug script to understand image data format and expected input shapes
"""

import pickle
import numpy as np
from pathlib import Path

# Model paths
MODELS_DIR = Path(__file__).parent / 'models'
VISION_METADATA_PATH = MODELS_DIR / 'driving_vision_metadata.pkl'
POLICY_METADATA_PATH = MODELS_DIR / 'driving_policy_metadata.pkl'

def load_metadata(metadata_path):
    """Load model metadata"""
    try:
        with open(metadata_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading metadata from {metadata_path}: {e}")
        return None

def analyze_vision_metadata():
    """Analyze vision model metadata"""
    print("=== Vision Model Metadata Analysis ===")
    metadata = load_metadata(VISION_METADATA_PATH)
    if metadata is None:
        return

    print(f"Input shapes: {metadata['input_shapes']}")
    print(f"Output shapes: {metadata['output_shapes']}")
    print(f"Output slices: {metadata['output_slices']}")

    # Calculate expected input sizes
    for name, shape in metadata['input_shapes'].items():
        total_size = np.prod(shape)
        print(f"Input '{name}': shape {shape}, total size {total_size}")

        # For image inputs, analyze the format
        if len(shape) == 4:  # Batch, Channels, Height, Width
            batch, channels, height, width = shape
            print(f"  - Image format: {batch} batch, {channels} channels, {height}x{width}")
            print(f"  - Expected YUV420 data size: {height * width * 6} (6 channels)")
            print(f"  - Expected total size: {total_size}")

def analyze_policy_metadata():
    """Analyze policy model metadata"""
    print("\n=== Policy Model Metadata Analysis ===")
    metadata = load_metadata(POLICY_METADATA_PATH)
    if metadata is None:
        return

    print(f"Input shapes: {metadata['input_shapes']}")
    print(f"Output shapes: {metadata['output_shapes']}")
    print(f"Output slices: {metadata['output_slices']}")

    # Calculate expected input sizes
    for name, shape in metadata['input_shapes'].items():
        total_size = np.prod(shape)
        print(f"Input '{name}': shape {shape}, total size {total_size}")

def create_sample_inputs():
    """Create sample inputs for testing"""
    print("\n=== Sample Input Creation ===")

    vision_metadata = load_metadata(VISION_METADATA_PATH)
    policy_metadata = load_metadata(POLICY_METADATA_PATH)

    if vision_metadata:
        print("Vision model sample inputs:")
        for name, shape in vision_metadata['input_shapes'].items():
            sample_input = np.random.rand(*shape).astype(np.float32)
            print(f"  '{name}': shape {sample_input.shape}, dtype {sample_input.dtype}")

    if policy_metadata:
        print("Policy model sample inputs:")
        for name, shape in policy_metadata['input_shapes'].items():
            sample_input = np.random.rand(*shape).astype(np.float32)
            print(f"  '{name}': shape {sample_input.shape}, dtype {sample_input.dtype}")

def main():
    print("ONNX Model Input/Output Analysis")
    print("=" * 50)

    analyze_vision_metadata()
    analyze_policy_metadata()
    create_sample_inputs()

    print("\n=== Notes ===")
    print("1. YUV420 format has 6 channels:")
    print("   - Channels 0,1,2,3: Full-res Y channel (subsampled)")
    print("   - Channel 4: Half-res U channel")
    print("   - Channel 5: Half-res V channel")
    print("2. Image data from VisionBuf.data is raw bytes that need proper reshaping")
    print("3. The reshape error suggests the data size doesn't match expected input size")

if __name__ == "__main__":
    main()