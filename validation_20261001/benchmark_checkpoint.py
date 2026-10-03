"""New hardware-documented inference benchmark; never identifies a historical host."""
import argparse
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

from validation_common import ROOT, FROZEN, historical_split, sha256, write_json
sys.path.insert(0, str(FROZEN / "scripts"))
sys.path.insert(0, str(ROOT / "runtime"))
from pc_mps import Discretizer
from pc_cmps_rwgan.model import ConditionalMPSMixture, ResidualTrajectoryDecoder


def hardware():
    cpu = None
    if platform.system() == "Windows":
        try:
            command = "Get-CimInstance Win32_Processor | Select-Object Name,Manufacturer,Architecture,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress"
            cpu = json.loads(subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", command], text=True, timeout=30))
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    elif Path("/proc/cpuinfo").exists():
        text = Path("/proc/cpuinfo").read_text()
        cpu = {"model_names": sorted(set(line.split(":", 1)[1].strip()
                 for line in text.splitlines() if line.startswith("model name")))}
    return {
        "platform": platform.platform(), "architecture": platform.machine(),
        "cpu": cpu, "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "pandas": pd.__version__,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_build_config": torch.__config__.show(),
    }


def load_models(run_dir, device):
    checkpoint_path = run_dir / "pc_cmps_rwgan_final.pt"
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    if ckpt.get("phase") != "complete" or ckpt.get("epoch") != cfg["adversarial_epochs"]:
        raise ValueError("Checkpoint is not a completed final run")
    state = ckpt["decoder"]
    features, conditions = ckpt["feature_names"], ckpt["condition_names"]
    prior = ConditionalMPSMixture(
        cfg["seq_len"] * len(features), cfg["bins"], len(conditions),
        experts=cfg["mps_experts"], bond_dim=cfg["mps_bond_dim"],
        gate_hidden_dim=cfg["gate_hidden_dim"], gate_constant=cfg["gate_constant"], seed=cfg["seed"])
    decoder = ResidualTrajectoryDecoder(
        cfg["seq_len"], len(features), len(conditions), cfg["bins"],
        state["centers_normalized"], state["lower_bounds_normalized"],
        state["upper_bounds_normalized"], embedding_dim=cfg["embedding_dim"],
        hidden_dim=cfg["decoder_hidden_dim"], num_layers=cfg["num_layers"],
        noise_dim=cfg["noise_dim"], residual_scale=cfg["residual_scale"],
        non_residual=cfg["non_residual_decoder"])
    prior.load_state_dict(ckpt["prior"], strict=True)
    decoder.load_state_dict(state, strict=True)
    prior.to(device).eval()
    decoder.to(device).eval()
    geometry = json.loads((run_dir / "hybrid_geometry.json").read_text())
    if geometry["discretizers"] != ckpt["discretizers"]:
        raise ValueError("Checkpoint/geometry discretizers differ")
    allowed_features = [Discretizer(**item).allowed_indices(*bounds, device=device)
                        for item, bounds in zip(ckpt["discretizers"], geometry["physical_bounds"])]
    allowed = [allowed_features[f] for _ in range(cfg["seq_len"]) for f in range(len(features))]
    return prior, decoder, allowed, cfg, conditions, sha256(checkpoint_path)


def run(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    if args.repeats < 2 or args.warmup < 1 or args.threads < 1:
        raise ValueError("Need repeats >= 2, warmup >= 1, threads >= 1")
    if (args.out / "benchmark.json").exists():
        raise FileExistsError("Choose a fresh output directory to preserve previous measurements")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    environment = hardware()
    frame = pd.read_csv(FROZEN / "data/hydro_dataset_pcwgan_gap_clean.csv")
    _, holdout_idx, _ = historical_split(frame)
    environment.update({
        "utc_started": datetime.now(timezone.utc).isoformat(),
        "kind": "new_checkpoint_inference_benchmark_not_historical_replication",
        "device": args.device, "warmup_per_operation": args.warmup,
        "timed_repetitions_per_operation": args.repeats,
        "batch_sizes": args.batches, "seeds": args.seeds, "variants": args.variants,
        "measurement": "perf_counter_ns; synchronize CUDA before and after if applicable",
        "execution_mode": "torch.inference_mode; eval; float32; resident tensors",
        "noise": "fixed seed=1000+model_seed; preallocated shared noise for paired operations",
        "operation_order": "rotated across timed rounds",
        "scope": "no checkpoint loading, disk I/O, transfer, optimizer or backward inside timings",
        "script_sha256": sha256(__file__),
        "model_source_sha256": sha256(FROZEN / "scripts/pc_cmps_rwgan/model.py"),
        "mps_source_sha256": sha256(ROOT / "runtime/pc_mps.py"),
        "mps_compatibility_fix": "optional log_norm cache hooks only; original contractions and sampling retained",
    })
    write_json(args.out / "environment.json", environment)
    raw, checkpoints = [], []
    for variant in args.variants:
        for seed in args.seeds:
            directory = args.runs / variant / f"seed_{seed}"
            prior, decoder, allowed, cfg, cond_names, digest = load_models(directory, args.device)
            if cfg["seed"] != seed or cfg["non_residual_decoder"] != (variant == "non_residual_decoder"):
                raise ValueError("Requested variant/seed does not match checkpoint")
            frozen_cfg = json.loads((FROZEN / f"configs/{variant}/seed_{seed}.json").read_text())
            if any(cfg[key] != frozen_cfg[key] for key in cfg if key not in ("data_path", "output_dir") and key in frozen_cfg):
                raise ValueError("Checkpoint configuration differs from frozen run")
            stored = np.load(directory / "fake_sequences_real_unit.npy", allow_pickle=False)
            canonical = np.load(FROZEN / f"data/{variant}/seed_{seed}/fake_unit.npy", allow_pickle=False)
            if stored.shape != canonical.shape or not np.allclose(stored, canonical, rtol=1e-6, atol=1e-7):
                raise ValueError("Checkpoint directory outputs do not match the frozen result")
            conditions_all = frame[cond_names].to_numpy(np.float32)[holdout_idx]
            codes_all = np.load(directory / "sampled_mps_codes.npy", allow_pickle=False)
            checkpoints.append({
                "variant": variant, "seed": seed, "checkpoint_sha256": digest,
                "geometry_sha256": sha256(directory / "hybrid_geometry.json"),
                "codes_sha256": sha256(directory / "sampled_mps_codes.npy"),
                "frozen_outputs_match": True, "checkpoint_epoch": cfg["adversarial_epochs"],
                "prior_parameters": sum(p.numel() for p in prior.parameters()),
                "decoder_parameters": sum(p.numel() for p in decoder.parameters()),
            })
            for batch in args.batches:
                if batch > len(conditions_all) or batch < 1:
                    raise ValueError("Invalid batch size")
                torch.manual_seed(1000 + seed)
                cond = torch.from_numpy(conditions_all[:batch]).to(args.device)
                codes = torch.from_numpy(codes_all[:batch]).long().to(args.device)
                noise = torch.randn(batch, cfg["seq_len"], cfg["noise_dim"], device=args.device)
                def sampling_decoding():
                    sampled, _ = prior.sample(cond)
                    return decoder(sampled, cond, noise)
                operations = {
                    "decoder_only_fixed_inputs": lambda: decoder(codes, cond, noise),
                    "sampling_plus_decoder_fixed_noise": sampling_decoding,
                    "standalone_exact_box_audit": lambda: prior.allowed_log_barrier(allowed, cond, per_site=False),
                    "prior_nll_forward": lambda: prior.nll(codes, cond),
                    "prior_nll_plus_audit_shared_cache_forward": lambda: prior.nll_and_allowed_log_barrier(codes, allowed, cond),
                }
                def sync():
                    if args.device == "cuda":
                        torch.cuda.synchronize()
                with torch.inference_mode():
                    for operation in operations.values():
                        for _ in range(args.warmup):
                            value = operation()
                            tensors = value if isinstance(value, tuple) else (value,)
                            if not all(torch.isfinite(t).all().item() for t in tensors):
                                raise ValueError("Non-finite output in benchmark")
                        sync()
                    names = list(operations)
                    for repeat in range(args.repeats):
                        order = names[repeat % len(names):] + names[:repeat % len(names)]
                        for name in order:
                            sync()
                            start = time.perf_counter_ns()
                            value = operations[name]()
                            sync()
                            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                            raw.append(dict(variant=variant, seed=seed, batch=batch,
                                            operation=name, repeat=repeat,
                                            milliseconds_per_batch=elapsed_ms,
                                            milliseconds_per_window_amortized=elapsed_ms / batch))
                print(f"{variant} seed={seed} batch={batch} complete", flush=True)
                pd.DataFrame(raw).to_csv(args.out / "timings_raw.csv", index=False)
            del prior, decoder
    table = pd.DataFrame(raw)
    summary = table.groupby(["variant", "seed", "batch", "operation"]).agg(
        repeats=("milliseconds_per_batch", "count"),
        mean_ms_batch=("milliseconds_per_batch", "mean"),
        median_ms_batch=("milliseconds_per_batch", "median"),
        std_ms_batch=("milliseconds_per_batch", "std"),
        min_ms_batch=("milliseconds_per_batch", "min"),
        max_ms_batch=("milliseconds_per_batch", "max"),
        mean_ms_window=("milliseconds_per_window_amortized", "mean")).reset_index()
    summary.to_csv(args.out / "timings_by_checkpoint.csv", index=False)
    aggregate = summary.groupby(["variant", "batch", "operation"]).agg(
        checkpoints=("seed", "count"), mean_of_checkpoint_means_ms_batch=("mean_ms_batch", "mean"),
        mean_of_checkpoint_means_ms_window=("mean_ms_window", "mean"),
        between_checkpoint_std_ms_batch=("mean_ms_batch", "std")).reset_index()
    aggregate.to_csv(args.out / "timings_summary.csv", index=False)
    wide = table.pivot(index=["variant", "seed", "batch", "repeat"], columns="operation", values="milliseconds_per_batch")
    wide["extra_forward_ms"] = wide["prior_nll_plus_audit_shared_cache_forward"] - wide["prior_nll_forward"]
    wide["extra_forward_fraction"] = wide["extra_forward_ms"] / wide["prior_nll_forward"]
    wide.reset_index().to_csv(args.out / "paired_forward_overhead.csv", index=False)
    write_json(args.out / "benchmark.json", {
        "environment": environment, "checkpoints": checkpoints, "timed_calls": len(table),
        "limitations": [
            "New local measurements do not identify the historical 2.738 ms CPU or validate GPU latency.",
            "Standalone box audit and incremental prior-objective forward cost are distinct.",
            "No backward/optimizer/full-training audit overhead or memory overhead is measured.",
            "All holdout actuator conditions are passive; active workloads require separate validation.",
            "This box audit does not include marginal or transition audits.",
            "Batch-16 per-window times are amortized throughput, not batch-one latency.",
            "Timing repeats are technical repeats, not independent training replicates.",
        ],
    })
    print(aggregate.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--variants", nargs="+", choices=["non_residual_decoder", "full"], default=["non_residual_decoder", "full"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 47)))
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    run(parser.parse_args())
