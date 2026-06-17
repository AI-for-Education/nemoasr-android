#!/usr/bin/env python3

import argparse

import onnx_asr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="nemo-parakeet-ctc-0.6b")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--quantization", default="int8")
    args = parser.parse_args()

    model = onnx_asr.load_model(
        args.model_name,
        args.model_dir,
        quantization=args.quantization,
    )
    print(model.recognize(args.wav))


if __name__ == "__main__":
    main()
