from __future__ import annotations

import argparse
import json

from nuosu_llm.mt_data import prepare_mt_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build target-only Nuosu MT train/eval JSONL")
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    manifest = prepare_mt_dataset(
        train_path=args.train,
        validation_path=args.validation,
        test_path=args.test,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
