import os
import tempfile
import pytest
import torch
import cellpose.resnet_torch as cp_net
import cellpose2onnx
import cellpose2onnx_gui


@pytest.fixture
def dummy_model_file(tmp_path):
    """Creates a dummy CPnet state dict file for testing."""
    model = cp_net.CPnet([2, 32, 64, 128, 256], 3, sz=3, mkldnn=None, diam_mean=30.0)
    model_path = tmp_path / "dummy_cpnet_model"
    torch.save(model.state_dict(), model_path)
    return str(model_path)


def test_convert_to_onnx_success(dummy_model_file, tmp_path):
    out_dir = str(tmp_path / "onnx_out")
    onnx_path = cellpose2onnx.convert_to_ONNX(
        model_path=dummy_model_file,
        output_directory=out_dir,
        diam_mean=30.0,
        batch_size=1,
        opset_version=18,
    )

    assert os.path.exists(onnx_path)
    assert onnx_path.endswith(".onnx")


def test_cli_argument_validation(dummy_model_file, tmp_path, monkeypatch):
    out_dir = str(tmp_path / "cli_out")
    test_args = [
        "cellpose2onnx.py",
        "--model_path",
        dummy_model_file,
        "--output_directory",
        out_dir,
        "--mean_diameter",
        "25.0",  # Invalid diameter (must be 17.0 or 30.0)
    ]
    monkeypatch.setattr("sys.argv", test_args)

    with pytest.raises(ValueError, match="mean_diameter must be either 17.0"):
        cellpose2onnx.main()


def test_gui_start_conversion_validation(tmp_path, monkeypatch):
    # Mock messagebox in cellpose2onnx_gui to prevent Tkinter window display initialization in headless test environment
    dummy_calls = []

    def mock_showerror(title, message):
        dummy_calls.append(("error", title, message))

    def mock_showinfo(title, message):
        dummy_calls.append(("info", title, message))

    monkeypatch.setattr("cellpose2onnx_gui.messagebox.showerror", mock_showerror)
    monkeypatch.setattr("cellpose2onnx_gui.messagebox.showinfo", mock_showinfo)

    # Test missing output dir
    cellpose2onnx_gui.start_conversion("", "", "30.0")
    assert len(dummy_calls) == 1
    assert dummy_calls[-1][1] == "Error"

    # Test invalid mean diameter float
    out_dir = str(tmp_path / "gui_out")
    cellpose2onnx_gui.start_conversion("some_path", out_dir, "invalid")
    assert dummy_calls[-1][2] == "Mean diameter must be a valid number (e.g. 17.0 or 30.0)."

    # Test unallowed mean diameter float
    cellpose2onnx_gui.start_conversion("some_path", out_dir, "20.0")
    assert "either 17.0" in dummy_calls[-1][2]
