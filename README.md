# Cellpose2onnx

Adapted from arivis V4D

Cellpose2onnx is a tool for converting Cellpose models to the ONNX format. This conversion allows you to use Cellpose models in various frameworks and environments that support ONNX, enhancing interoperability and deployment options.

## Features

- Convert individual Cellpose models to ONNX format.
- Convert all available Cellpose models to ONNX format.
- Command line interface (CLI) and easy-to-use graphical user interface (GUI).
- Default ONNX opset version 18 for compatibility with modern PyTorch and ONNX runtime releases.
- Input validation and status reporting.
- Example usage script demonstrating conversion and inference with ONNX Runtime.

## Requirements

- Python 3.x
- Cellpose
- PyTorch
- ONNX and ONNXScript
- ONNX Runtime
- tkinter (for GUI)

## Installation

1. Clone the repository:

```sh
git clone https://github.com/yourusername/Cellpose2onnx.git
cd Cellpose2onnx
```

2. Install the required dependencies:

```sh
pip install -r requirements.txt
```

## Usage

### Using the Example Usage Script

To convert a model and run an ONNX inference sample:

```sh
python example_usage.py
```

### Using the GUI

1. Run the GUI application:

```sh
python cellpose2onnx_gui.py
```

2. Use the GUI to select the model path, output directory, and mean diameter (default `30.0`). Click the "Convert" button to start the conversion.

### Using the Command Line

1. Convert an individual model:

```sh
python cellpose2onnx.py --model_path /path/to/your/model --output_directory /path/to/output --mean_diameter 30.0
```

2. Convert all available models:

```sh
python cellpose2onnx.py --output_directory /path/to/output
```

3. Optional arguments:

```sh
python cellpose2onnx.py --help
```

### Running Inference with ONNX Runtime

Once converted, ONNX models can be loaded into ONNX Runtime:

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("path/to/model.onnx")
input_name = session.get_inputs()[0].name

# Cellpose model inputs expect [batch_size, 2, height, width] float32 arrays
dummy_input = np.random.randn(1, 2, 224, 224).astype(np.float32)

outputs = session.run(None, {input_name: dummy_input})
flows_and_probabilities, style_vector = outputs[0], outputs[1]
```

## Running Tests

To run the test suite:

```sh
pytest
```

## GUI Screenshot

![Cellpose2onnx GUI](screenshot.png)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

Adapted from arivis V4D.
