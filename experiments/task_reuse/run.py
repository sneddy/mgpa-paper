#!/usr/bin/env python3
"""Published task-reuse phases; install the mgpa release before running."""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None):
    """Dispatch a protocol phase using explicit data, run and report locations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "fit", "evaluate", "report", "verify", "verify-selection", "all"))
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/paper.json")
    parser.add_argument("--run-dir", type=Path, help="New run directory; default artifacts/runs/reproduction")
    parser.add_argument("--manifest", type=Path, help="Portable raw-epoch/full-token input manifest")
    parser.add_argument("--raw-root", type=Path, help="Original simultaneous Neuroscan/Flex dataset; lazily prepare paired epochs")
    parser.add_argument("--checkpoint", type=Path, help="Frozen EEGPT safetensors, needed for raw epoch inputs")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--published", action="store_true", help="Use retained paper cells for reporting or selection verification")
    parser.add_argument("--output-dir", type=Path, help="Report destination")
    args = parser.parse_args(argv)
    if args.published and args.phase not in ("report", "verify-selection"):
        parser.error("--published is available for report and verify-selection")
    from experiment import pipeline, reporting
    from mgpa.data.reuse import write_json, raw_manifest
    cfg = pipeline.config(args.config)
    root = (args.run_dir or Path(__file__).parent / "artifacts/runs/reproduction").resolve()
    if args.published:
        root = Path(__file__).parent / "artifacts"
    if args.raw_root is not None:
        if args.manifest is not None or args.checkpoint is None:
            parser.error("--raw-root requires --checkpoint and cannot be combined with --manifest")
        args.manifest = raw_manifest(args.raw_root, root / "inputs.json", checkpoint=args.checkpoint)
    if args.phase in ("prepare", "evaluate", "all") and args.manifest is None:
        parser.error("--manifest is required for this phase")
    if args.phase == "report":
        reporting.report(root, args.output_dir)
        print(f"Report written to {args.output_dir or root / 'report'}")
        return 0
    if args.phase == "verify-selection":
        import json
        print(json.dumps(reporting.verify_dev_evidence(root, cfg), indent=2, sort_keys=True))
        return 0
    from threadpoolctl import threadpool_limits
    import torch
    torch.set_num_threads(cfg["threads"])
    phases = ("prepare", "fit", "evaluate", "report", "verify") if args.phase == "all" else (args.phase,)
    with threadpool_limits(limits=cfg["threads"]):
        for phase in phases:
            print(f"Task reuse: {phase}", flush=True)
            if phase in ("prepare", "evaluate"):
                getattr(pipeline, phase)(root, cfg, args.manifest, checkpoint=args.checkpoint, device=args.device)
            elif phase == "report":
                reporting.report(root, args.output_dir)
            elif phase == "verify":
                write_json(root / "logs/verification.json", pipeline.verify(root, cfg))
            else:
                pipeline.fit(root, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
