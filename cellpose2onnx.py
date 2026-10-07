import argparse
import os
import cellpose.models as cp_model
import cellpose.resnet_torch as cp_net
import torch


def convert_to_ONNX(
    model_path: str,
    output_directory: str,
    diam_mean: float,
    batch_size: int = 1,
    opset_version: int = 18,
) -> str:
    """Converts a PyTorch Cellpose model file to ONNX format."""
    os.makedirs(output_directory, exist_ok=True)

    model = cp_net.CPnet(
        [2, 32, 64, 128, 256],
        3,
        sz=3,
        mkldnn=None,
        diam_mean=diam_mean,
    )

    model.residual_on = True
    model.style_on = True
    model.concatenation = False

    model.load_model(model_path)
    model.eval()

    # convert to onnx
    model_filename = os.path.basename(model_path)
    onnx_model_path = os.path.join(output_directory, model_filename + ".onnx")
    dummy = torch.randn(batch_size, 2, 224, 224, requires_grad=True)

    torch.onnx.export(
        model,
        dummy,
        onnx_model_path,
        verbose=False,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output", "style"],
    )
    print(f"Successfully converted model {model_path} -> {onnx_model_path}")
    return onnx_model_path


def convert_all_models(output_directory: str, opset_version: int = 18):
    """Converts all built-in and user Cellpose models to ONNX format."""
    os.makedirs(output_directory, exist_ok=True)

    # get built-in model names and custom model names
    all_models = cp_model.MODEL_NAMES.copy()
    model_strings = cp_model.get_user_models()
    all_models.extend(model_strings)

    for model_type in all_models:
        model_string = model_type if model_type is not None else "cyto"
        if model_string == "nuclei":
            diam_mean = 17.0
        else:
            diam_mean = 30.0

        if model_type in ["cyto", "cyto2", "cyto3", "nuclei"]:
            model_range = range(4)
        else:
            model_range = range(1)

        model_paths = [
            cp_model.model_path(model_string, j, True) for j in model_range
        ]

        for model_path in model_paths:
            convert_to_ONNX(
                model_path, output_directory, diam_mean, opset_version=opset_version
            )


def main():
    parser = argparse.ArgumentParser(description="Cellpose to ONNX converter CLI")
    parser.add_argument(
        "--output_directory",
        required=False,
        default=os.path.join(cp_model.MODEL_DIR, "output"),
        type=str,
        help="Output directory for converted models.",
    )
    parser.add_argument(
        "--model_path",
        required=False,
        default=None,
        type=str,
        help="Full path to the individual cellpose model file.",
    )
    parser.add_argument(
        "--mean_diameter",
        required=False,
        type=float,
        help="Mean diameter used for training the given model (17.0 for nuclei-based models, otherwise 30.0).",
    )
    parser.add_argument(
        "--opset_version",
        required=False,
        default=18,
        type=int,
        help="ONNX opset version for export (default: 18).",
    )

    args = parser.parse_args()

    os.makedirs(args.output_directory, exist_ok=True)

    if args.model_path:
        if args.mean_diameter not in [17.0, 30.0]:
            raise ValueError(
                "mean_diameter must be either 17.0 (for nuclei-based models) or 30.0 for all other models."
            )
        convert_to_ONNX(
            model_path=args.model_path,
            output_directory=args.output_directory,
            diam_mean=args.mean_diameter,
            opset_version=args.opset_version,
        )
    else:
        convert_all_models(
            output_directory=args.output_directory,
            opset_version=args.opset_version,
        )

    print(f"Output models are saved here: {args.output_directory}")
    print("Conversion completed successfully.")


if __name__ == "__main__":
    main()
