#!/usr/bin/env python3
"""
Check ONNX model input/output specifications
"""

import onnxruntime as ort
from pathlib import Path

# Model paths
MODELS_DIR = Path(__file__).parent / 'models'
VISION_ONNX_PATH = MODELS_DIR / 'driving_vision.onnx'
POLICY_ONNX_PATH = MODELS_DIR / 'driving_policy.onnx'

def check_model_specs(model_path, model_name):
    """Check ONNX model input/output specifications"""
    print(f"\n=== {model_name} Model Specifications ===")

    try:
        # Load the model
        session = ort.InferenceSession(str(model_path))

        # Get input specifications
        print("Input specifications:")
        for input_info in session.get_inputs():
            print(f"  Name: {input_info.name}")
            print(f"  Shape: {input_info.shape}")
            print(f"  Type: {input_info.type}")
            print()

        # Get output specifications
        print("Output specifications:")
        for output_info in session.get_outputs():
            print(f"  Name: {output_info.name}")
            print(f"  Shape: {output_info.shape}")
            print(f"  Type: {output_info.type}")
            print()

        # Get available providers
        print(f"Available providers: {session.get_providers()}")

    except Exception as e:
        print(f"Error loading {model_name} model: {e}")

def main():
    print("ONNX Model Specifications Check")
    print("=" * 50)

    check_model_specs(VISION_ONNX_PATH, "Vision")
    check_model_specs(POLICY_ONNX_PATH, "Policy")

    print("\n=== Notes ===")
    print("1. Check the input data types to ensure compatibility")
    print("2. Vision models typically expect uint8 for images")
    print("3. Policy models typically expect float16 for numerical inputs")
    print("4. Make sure input shapes match the expected specifications")

if __name__ == "__main__":
    main()