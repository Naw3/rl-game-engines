import argparse
import numpy as np
import onnxruntime as ort

def main():
    parser = argparse.ArgumentParser(description="Compare FP32 and INT8 QDQ ONNX models.")
    parser.add_argument("--fp32", default="connect4_model.onnx")
    parser.add_argument("--int8", default="connect4_model_int8.onnx")
    parser.add_argument("--batch", type=int, default=100)
    args = parser.parse_args()
    
    print(f"Loading FP32 model from {args.fp32}")
    try:
        sess_fp32 = ort.InferenceSession(args.fp32, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"Failed to load FP32 model: {e}")
        return

    print(f"Loading INT8 model from {args.int8}")
    try:
        sess_int8 = ort.InferenceSession(args.int8, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"Failed to load INT8 model: {e}")
        return
    
    # Generate random dummy input mirroring a 3-plane Connect4 board
    dummy_input = np.random.randn(args.batch, 3, 6, 7).astype(np.float32)
    
    out_fp32 = sess_fp32.run(None, {"input": dummy_input})
    out_int8 = sess_int8.run(None, {"input": dummy_input})
    
    pol_fp32, val_fp32 = out_fp32
    pol_int8, val_int8 = out_int8
    
    pol_diff = np.abs(pol_fp32 - pol_int8).max()
    val_diff = np.abs(val_fp32 - val_int8).max()
    
    argmax_fp32 = np.argmax(pol_fp32, axis=1)
    argmax_int8 = np.argmax(pol_int8, axis=1)
    diff_moves = np.sum(argmax_fp32 != argmax_int8)
    
    print(f"\n--- Comparison on {args.batch} random samples ---")
    print(f"Max Policy Error: {pol_diff:.6f}")
    print(f"Max Value Error:  {val_diff:.6f}")
    print(f"Differing Argmax: {diff_moves}/{args.batch} ({(diff_moves/args.batch)*100:.1f}%)")

if __name__ == "__main__":
    main()
