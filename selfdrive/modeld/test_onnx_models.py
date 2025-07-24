#!/usr/bin/env python3
"""
Simple test script for ONNX models in openpilot
This script demonstrates how to load and run the ONNX models
"""

import os
import time
import pickle
import numpy as np
from pathlib import Path
import argparse

# Try to import onnxruntime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    print("Warning: onnxruntime not installed. Please install with: pip install onnxruntime")
    ONNX_AVAILABLE = False

# Model paths
MODELS_DIR = Path(__file__).parent / 'models'
VISION_ONNX_PATH = MODELS_DIR / 'driving_vision.onnx'
POLICY_ONNX_PATH = MODELS_DIR / 'driving_policy.onnx'
BIG_VISION_ONNX_PATH = MODELS_DIR / 'big_driving_vision.onnx'
BIG_POLICY_ONNX_PATH = MODELS_DIR / 'big_driving_policy.onnx'

# Metadata paths
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


def create_dummy_inputs(vision_metadata, policy_metadata):
    """Create dummy inputs for testing"""
    vision_inputs = {}
    for name, shape in vision_metadata['input_shapes'].items():
        # Create random input data with proper shape and dtype
        if len(shape) == 4 and shape[1] > 6:  # Image input with multiple channels
            # Use uint8 for image inputs
            vision_inputs[name] = np.random.randint(0, 256, shape, dtype=np.uint8)
        else:
            # Use float32 for other inputs
            vision_inputs[name] = np.random.rand(*shape).astype(np.float32)

    policy_inputs = {}
    for name, shape in policy_metadata['input_shapes'].items():
        # Create random input data with proper shape and dtype
        # Policy models typically expect float16
        policy_inputs[name] = np.random.rand(*shape).astype(np.float16)

    return vision_inputs, policy_inputs


def test_vision_model(model_path, metadata_path, use_big_models=False):
    """Test vision model inference"""
    if not ONNX_AVAILABLE:
        print("ONNX runtime not available")
        return False

    print(f"Testing vision model: {model_path}")

    # Load metadata
    metadata = load_metadata(metadata_path)
    if metadata is None:
        return False

    print(f"Vision model input shapes: {metadata['input_shapes']}")
    print(f"Vision model output shapes: {metadata['output_shapes']}")

    # Create dummy inputs
    vision_inputs, _ = create_dummy_inputs(metadata, {})

    try:
        # Create ONNX session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Try to use GPU if available
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        session = ort.InferenceSession(str(model_path), sess_options, providers=providers)
        print(f"Successfully loaded vision model")
        print(f"Available providers: {session.get_providers()}")

        # Run inference
        print("Running vision model inference...")
        start_time = time.time()
        outputs = session.run(None, vision_inputs)
        inference_time = time.time() - start_time

        print(f"Vision model inference completed in {inference_time:.3f} seconds")

        # Handle different output types
        if isinstance(outputs[0], np.ndarray):
            output_shape = outputs[0].shape
            output_dtype = outputs[0].dtype
        else:
            # Convert to numpy array if needed
            output_array = np.array(outputs[0])
            output_shape = output_array.shape
            output_dtype = output_array.dtype

        print(f"Output shape: {output_shape}")
        print(f"Output dtype: {output_dtype}")

        return True

    except Exception as e:
        print(f"Error running vision model: {e}")
        return False


def test_policy_model(model_path, metadata_path, use_big_models=False):
    """Test policy model inference"""
    if not ONNX_AVAILABLE:
        print("ONNX runtime not available")
        return False

    print(f"Testing policy model: {model_path}")

    # Load metadata
    metadata = load_metadata(metadata_path)
    if metadata is None:
        return False

    print(f"Policy model input shapes: {metadata['input_shapes']}")
    print(f"Policy model output shapes: {metadata['output_shapes']}")

    # Create dummy inputs
    _, policy_inputs = create_dummy_inputs({}, metadata)

    try:
        # Create ONNX session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Try to use GPU if available
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        session = ort.InferenceSession(str(model_path), sess_options, providers=providers)
        print(f"Successfully loaded policy model")
        print(f"Available providers: {session.get_providers()}")

        # Run inference
        print("Running policy model inference...")
        start_time = time.time()
        outputs = session.run(None, policy_inputs)
        inference_time = time.time() - start_time

        print(f"Policy model inference completed in {inference_time:.3f} seconds")

        # Handle different output types
        if isinstance(outputs[0], np.ndarray):
            output_shape = outputs[0].shape
            output_dtype = outputs[0].dtype
        else:
            # Convert to numpy array if needed
            output_array = np.array(outputs[0])
            output_shape = output_array.shape
            output_dtype = output_array.dtype

        print(f"Output shape: {output_shape}")
        print(f"Output dtype: {output_dtype}")

        return True

    except Exception as e:
        print(f"Error running policy model: {e}")
        return False


def list_available_models():
    """List available ONNX models"""
    print("Available ONNX models:")
    print(f"  Vision models:")
    if VISION_ONNX_PATH.exists():
        print(f"    - {VISION_ONNX_PATH.name} ({VISION_ONNX_PATH.stat().st_size / (1024*1024):.1f} MB)")
    if BIG_VISION_ONNX_PATH.exists():
        print(f"    - {BIG_VISION_ONNX_PATH.name} ({BIG_VISION_ONNX_PATH.stat().st_size / (1024*1024):.1f} MB)")

    print(f"  Policy models:")
    if POLICY_ONNX_PATH.exists():
        print(f"    - {POLICY_ONNX_PATH.name} ({POLICY_ONNX_PATH.stat().st_size / (1024*1024):.1f} MB)")
    if BIG_POLICY_ONNX_PATH.exists():
        print(f"    - {BIG_POLICY_ONNX_PATH.name} ({BIG_POLICY_ONNX_PATH.stat().st_size / (1024*1024):.1f} MB)")

    print(f"  Metadata files:")
    if VISION_METADATA_PATH.exists():
        print(f"    - {VISION_METADATA_PATH.name}")
    if POLICY_METADATA_PATH.exists():
        print(f"    - {POLICY_METADATA_PATH.name}")


def main():
    parser = argparse.ArgumentParser(description='Test ONNX models in openpilot')
    parser.add_argument('--list', action='store_true', help='List available models')
    parser.add_argument('--vision', action='store_true', help='Test vision model')
    parser.add_argument('--policy', action='store_true', help='Test policy model')
    parser.add_argument('--big-models', action='store_true', help='Use big models')
    parser.add_argument('--all', action='store_true', help='Test all models')

    args = parser.parse_args()

    if args.list:
        list_available_models()
        return

    if not ONNX_AVAILABLE:
        print("ONNX runtime is not available. Please install it first:")
        print("pip install onnxruntime")
        return

    success_count = 0
    total_tests = 0

    if args.all or args.vision:
        total_tests += 1
        vision_path = BIG_VISION_ONNX_PATH if args.big_models else VISION_ONNX_PATH
        if test_vision_model(vision_path, VISION_METADATA_PATH, args.big_models):
            success_count += 1

    if args.all or args.policy:
        total_tests += 1
        policy_path = BIG_POLICY_ONNX_PATH if args.big_models else POLICY_ONNX_PATH
        if test_policy_model(policy_path, POLICY_METADATA_PATH, args.big_models):
            success_count += 1

    if total_tests == 0:
        print("No tests specified. Use --help for options.")
        list_available_models()
    else:
        print(f"\nTest results: {success_count}/{total_tests} tests passed")


if __name__ == "__main__":
    main()