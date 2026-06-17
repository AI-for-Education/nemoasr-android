#!/usr/bin/env python3

import argparse
import json
import onnx_asr
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-manifest",
        required=True,
        help="Input JSONL manifest",
    )
    parser.add_argument(
        "--output-manifest",
        default=None,
        help="Output JSONL manifest. Defaults to input_manifest + .pred.jsonl",
    )
    parser.add_argument(
        "--model-name",
        default="nemo-parakeet-ctc-0.6b",
        help="Model name for onnx_asr.load_model()",
    )
    parser.add_argument(
        "--model-dir",
        default="nemo-onnx",
        help="Directory containing the exported ONNX model files",
    )
    parser.add_argument(
        "--quantization",
        default=None,#"int8",
        help="Quantization mode",
    )

    args = parser.parse_args()

    output_manifest = (
        args.output_manifest
        if args.output_manifest
        else args.input_manifest + ".pred.jsonl"
    )

    print("Loading model...")
    model = onnx_asr.load_model(
        args.model_name,
        args.model_dir,
        quantization=args.quantization,
    )

    total = 0

    with open(args.input_manifest, "r", encoding="utf-8") as fin, \
         open(output_manifest, "w", encoding="utf-8") as fout:

        for line in tqdm(fin):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            wav_path = item["audio_filepath"]

            try:
                pred_text = model.recognize(wav_path)

                if pred_text is None:
                    pred_text = ""

                pred_text = str(pred_text).strip()

            except Exception as e:
                print(f"ERROR: {wav_path}: {e}")
                pred_text = ""

            item["pred_text"] = pred_text

            fout.write(
                json.dumps(item, ensure_ascii=False) + "\n"
            )

            total += 1

            if total % 100 == 0:
                print(f"Processed {total} utterances")

    print(f"Done. Wrote {total} entries to {output_manifest}")


if __name__ == "__main__":
    main()
