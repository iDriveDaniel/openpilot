# ONNX Model Inference for openpilot

This directory contains scripts for running openpilot's neural network models using ONNX Runtime instead of the default tinygrad implementation.

## Files

- `onnx_modeld.py` - Full ONNX-based model inference script (alternative to `modeld.py`)
- `test_onnx_models.py` - Simple test script to verify ONNX models work
- `debug_image_format.py` - Debug script to analyze model input/output formats
- `check_onnx_models.py` - Check ONNX model input/output specifications
- `debug_model_output.py` - Debug script to understand model output structure
- `visualize_lane_lines.py` - Live lane line visualization from model output
- `visualize_onnx_output.py` - Demo visualization of ONNX model outputs
- `requirements_onnx.txt` - Python dependencies for ONNX inference
- `README_ONNX.md` - This file

## Prerequisites

1. Install ONNX Runtime:
```bash
pip install onnxruntime
```

Or install from the requirements file:
```bash
pip install -r requirements_onnx.txt
```

2. Make sure you have the ONNX models in the `models/` directory:
   - `driving_vision.onnx` (44MB)
   - `driving_policy.onnx` (15MB)
   - `big_driving_vision.onnx` (44MB)
   - `big_driving_policy.onnx` (15MB)
   - `driving_vision_metadata.pkl`
   - `driving_policy_metadata.pkl`

## Testing ONNX Models

First, test that the ONNX models work correctly:

```bash
# List available models
python test_onnx_models.py --list

# Test vision model
python test_onnx_models.py --vision

# Test policy model
python test_onnx_models.py --policy

# Test all models
python test_onnx_models.py --all

# Test big models
python test_onnx_models.py --all --big-models
```

## Debugging Model Formats

To understand the expected input/output formats:

```bash
python debug_image_format.py
```

This will show you:
- Expected input shapes for each model
- Output shapes and slices
- Sample input creation

## Checking ONNX Model Specifications

To check the exact input/output specifications of the ONNX models:

```bash
python check_onnx_models.py
```

This will show you:
- Input names, shapes, and data types
- Output names, shapes, and data types
- Available execution providers

## Debugging Model Output

To understand the structure of model output messages:

```bash
python debug_model_output.py
```

This will show you:
- Available fields in modelV2 messages
- Data types and structures
- How to access lane line data
- Message format details

**Important**: The model output structure is:
- **Lane Lines**: List of 4 single points (x, y, z coordinates), not lines
- **Road Edges**: List of 2 single points (x, y, z coordinates), not edges
- **Lane Probabilities**: List of 4 float values (0.0-1.0)
- All data is in Cap'n Proto format and needs to be converted to lists before processing

## Visualization Tools

### Demo Visualization
To see a demo of lane line visualization with simulated data:

```bash
python visualize_onnx_output.py --demo
```

This will show:
- Real-time lane line plots
- Lane line probabilities
- Vehicle position indicator
- Animated demo data

### Live Visualization
To visualize live lane lines from the model output:

```bash
# First run the ONNX model
python onnx_modeld.py --demo

# In another terminal, run the visualization
python visualize_lane_lines.py
```

This will display:
- Live lane line detection
- Road edges
- Lane probabilities
- Real-time updates

### Visualization Troubleshooting

If you get errors like "object is not subscriptable":
1. Run `python debug_model_output.py` to understand the message structure
2. The visualization script now properly handles Cap'n Proto objects
3. Make sure the model is running and producing output

## Running ONNX-based Model Inference

The `onnx_modeld.py` script provides a drop-in replacement for the standard `modeld.py`:

```bash
# Run in demo mode
python onnx_modeld.py --demo

# Run with big models
python onnx_modeld.py --big-models

# Run with both options
python onnx_modeld.py --demo --big-models
```

## Data Type Requirements

The ONNX models have specific data type requirements:

- **Vision Model Inputs**: `uint8` for image data
- **Policy Model Inputs**: `float16` for numerical data
- **Model Outputs**: Automatically converted to `float32` for compatibility
- **Downstream Processing**: Uses `float32` (NumPy linear algebra requires float32)

## Key Features

### ONNXModelState Class
- Loads ONNX models using ONNX Runtime
- Supports both regular and "big" model variants
- Handles input preprocessing and output parsing
- Maintains temporal buffers for model state

### Image Preprocessing
- Converts YUV420 camera data to expected 12-channel format
- Handles data size mismatches gracefully
- Maintains uint8 data type for vision model inputs
- Supports both regular and wide camera inputs

### Performance Optimizations
- GPU acceleration via CUDA (if available)
- Graph optimization enabled
- Parallel execution mode
- Proper memory management

### Compatibility
- Uses the same metadata files as tinygrad models
- Compatible with existing openpilot messaging system
- Maintains the same output format

## Differences from tinygrad version

1. **Runtime**: Uses ONNX Runtime instead of tinygrad
2. **Performance**: May be faster on some hardware, especially with GPU acceleration
3. **Dependencies**: Requires onnxruntime instead of tinygrad
4. **Memory**: Different memory management approach
5. **Data Types**: Uses uint8 for images and float16 for policy inputs

## Troubleshooting

### ONNX Runtime not found
```
pip install onnxruntime
```

### Model files not found
Make sure the ONNX model files are present in the `models/` directory.

### Data type errors
If you get errors like "Unexpected input data type":
1. Run `python check_onnx_models.py` to see expected data types
2. Vision models expect uint8 for images
3. Policy models expect float16 for numerical inputs

### NumPy linear algebra errors
If you get errors like "array type float16 is unsupported in linalg":
1. This is automatically fixed by converting outputs to float32
2. NumPy's linear algebra functions require float32 or float64
3. The script now handles this conversion automatically

### Image reshape errors
If you get reshape errors like "cannot reshape array of size X into shape Y":
1. Run `python debug_image_format.py` to understand expected formats
2. Check that camera data is being received properly
3. The script now handles size mismatches automatically

### GPU acceleration not working
ONNX Runtime will automatically fall back to CPU if CUDA is not available.

### Performance issues
- Try different ONNX Runtime providers
- Check if GPU acceleration is working
- Monitor memory usage

## Integration with openpilot

To use the ONNX version instead of the default tinygrad version:

1. Replace calls to `modeld.py` with `onnx_modeld.py`
2. Update any launch scripts or systemd services
3. Ensure all dependencies are installed

## Recent Fixes

### Output Data Type Fix (Latest)
- Fixed NumPy linear algebra compatibility issues
- Automatically convert ONNX outputs to float32
- Ensures compatibility with downstream processing functions
- Resolves "array type float16 is unsupported in linalg" errors

### Data Type Fix
- Fixed input data types to match ONNX model requirements
- Vision models now use uint8 for images
- Policy models now use float16 for numerical inputs
- Added check script for model specifications

### Image Preprocessing Fix
- Fixed YUV420 to 12-channel conversion
- Added proper data size handling
- Improved error handling for size mismatches
- Added debug script for format analysis

## Notes

- This is an experimental implementation
- May need adjustments for production use
- Image preprocessing uses simplified YUV conversion
- Test thoroughly before deployment
- The debug and check scripts help understand model requirements