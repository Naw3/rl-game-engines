import sys
from pathlib import Path
import torch
import numpy as np
import onnx

sys.path.append(str(Path(__file__).parent.parent))
from src_python.model import Connect4Net
from src_python.utils.onnx_export import export_onnx

def main():
    print("[python] Creating baseline Connect4Net...")
    model = Connect4Net(d_model=64, num_layers=4, nhead=4)
    model.eval()

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    fp32_path = models_dir / "test_fp32.onnx"
    int8_path = models_dir / "test_int8_qdq.onnx"

    print(f"[python] Exporting FP32 model to {fp32_path}...")
    
    # Custom export to avoid dynamic axes (TensorRT needs fixed shapes or explicit profiles)
    dummy_input = torch.randn(64, 3, 6, 7)
    
    torch.onnx.export(
        model,
        dummy_input,
        str(fp32_path),
        input_names=["input"],
        output_names=["policy", "value"],
        opset_version=18,
    )

    print(f"[python] Exporting INT8 QDQ model to {int8_path}...")
    import onnx
    from onnxruntime.quantization import QuantType, quantize_static, QuantFormat, CalibrationDataReader

    class DummyReader(CalibrationDataReader):
        def __init__(self):
            self.enum_data = iter([{"input": np.random.randn(64, 3, 6, 7).astype(np.float32)}])
        def get_next(self):
            return next(self.enum_data, None)

    quantize_static(
        str(fp32_path),
        str(int8_path),
        calibration_data_reader=DummyReader(),
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        nodes_to_exclude=["node_linear_17", "node_masked_fill"],
        extra_options={"DisableShapeInference": True},
    )

    print("[python] Export complete.")

if __name__ == "__main__":
    main()
