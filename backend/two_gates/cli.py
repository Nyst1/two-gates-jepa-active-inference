from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluation import evaluate_behavior
from .replay import export_replays
from .training import TrainingConfig, train_production_checkpoint, validate_production_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="two-gates", description="Two Gates demo utilities")
    subcommands = parser.add_subparsers(dest="command", required=True)

    train = subcommands.add_parser("train", help="Train three seeds and select the median checkpoint")
    train.add_argument("--transitions", type=int, default=50_000)
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--seeds", type=int, nargs="+", default=[11, 29, 47])

    evaluate = subcommands.add_parser("evaluate", help="Run the locked behavioral evaluation")
    evaluate.add_argument("--episodes", type=int, default=200)

    subcommands.add_parser("export-replays", help="Export replay JSON for backend and static frontend")
    validate = subcommands.add_parser("validate", help="Run model quality gates and update the manifest")
    validate.add_argument("--samples", type=int, default=5_000)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "train":
        manifest = train_production_checkpoint(
            PROJECT_ROOT / "artifacts" / "checkpoints",
            config=TrainingConfig(
                transitions=args.transitions,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seeds=tuple(args.seeds),
            ),
        )
        print(json.dumps(manifest, indent=2))
    elif args.command == "evaluate":
        report = evaluate_behavior(args.episodes)
        print(json.dumps(report, indent=2))
        if not report["passed"]:
            raise SystemExit(1)
    elif args.command == "export-replays":
        print(json.dumps(export_replays(PROJECT_ROOT), indent=2))
    elif args.command == "validate":
        report = validate_production_checkpoint(
            PROJECT_ROOT / "artifacts" / "checkpoints",
            samples=args.samples,
        )
        print(json.dumps(report, indent=2))
        if not report["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
