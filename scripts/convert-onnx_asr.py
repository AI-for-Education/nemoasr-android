#!/usr/bin/env python3

import argparse
from pathlib import Path

import nemo.collections.asr as nemo_asr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo", required=True, help="Path to the input .nemo model")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where model.onnx and tokens.txt will be written",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = nemo_asr.models.ASRModel.restore_from(args.nemo)
    model.export(str(output_dir / "model.onnx"))

    with (output_dir / "vocab.txt").open("w", encoding="utf-8") as f:
        for i, token in enumerate([*model.tokenizer.vocab, "<blk>"]):
            f.write(f"{token} {i}\n")

    print(f"Created {output_dir / 'model.onnx'}")
    print(f"Created {output_dir / 'vocab.txt'}")


if __name__ == "__main__":
    main()

