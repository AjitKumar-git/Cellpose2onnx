"""Example usage script for Cellpose2onnx.

This script demonstrates:
1. Converting a Cellpose model to ONNX format.
2. Loading and running inference on the converted ONNX model using ONNX Runtime.
"""

import os
import numpy as np
import onnxruntime as ort
import cellpose.models as cp_model
from cellpose2onnx import convert_to_ONNX


def main():
    output_dir = "onnx_models"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Obtain path to a built-in Cellpose model (e.g. 'cyto' model 0)
    print("Fetching built-in 'cyto' Cellpose model path...")
    model_path = cp_model.model_path("cyto", 0)
    print(f"Model path: {model_path}")

    # 2. Convert model to ONNX format
    print("\nConverting Cellpose model to ONNX...")
    onnx_model_path = convert_to_ONNX(
        model_path=model_path,
        output_directory=output_dir,
        diam_mean=30.0,
        batch_size=1,
        opset_version=18,
    )
    print(f"ONNX model saved at: {onnx_model_path}")

    # 3. Load ONNX model and run inference with ONNX Runtime
    print("\nLoading ONNX model into ONNX Runtime session...")
    session = ort.InferenceSession(onnx_model_path)

    # Inspect inputs and outputs
    input_info = session.get_inputs()[0]
    print(f"Model input name: {input_info.name}, shape: {input_info.shape}, type: {input_info.type}")
    for idx, out in enumerate(session.get_outputs()):
        print(f"Model output {idx} name: {out.name}, shape: {out.shape}, type: {out.type}")

    # Prepare dummy input tensor: batch_size=1, channels=2 (image + optional vector/channel), height=224, width=224
    dummy_input = np.random.randn(1, 2, 224, 224).astype(np.float32)

    print("\nRunning inference on dummy input tensor...")
    outputs = session.run(None, {input_info.name: dummy_input})

    cell_probability_and_flows, style_vector = outputs[0], outputs[1]
    print(f"Inference output shape (Flows & Probabilities): {cell_probability_and_flows.shape}")
    print(f"Inference output shape (Style vector): {style_vector.shape}")
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
