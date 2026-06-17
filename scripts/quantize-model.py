#!/usr/bin/env python3

import os
import sys

from onnxruntime.quantization import QuantType, quantize_dynamic

def main():
    input_dir = sys.argv[1]

    quantize_dynamic(
        model_input=os.path.join(input_dir, "model.onnx"),
        model_output=os.path.join(input_dir, "model.int8.onnx"),
        per_channel=True,
        weight_type=QuantType.QUInt8,
    )


if __name__ == "__main__":
    main()
