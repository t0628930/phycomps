"""Launch the existing training tools with the documented compatibility overlay.

This does not change the historical split or envelope. Strict preparation alone
must not be described as a leakage-free rerun of these training tools.
"""
import runpy
import sys
from validation_common import ROOT, FROZEN

sys.path.insert(0, str(FROZEN / "scripts"))
sys.path.insert(0, str(ROOT / "runtime"))
targets = {"train": "pc_cmps_rwgan.train", "ablation": "pc_cmps_rwgan.ablation"}
if len(sys.argv) < 2 or sys.argv[1] not in targets:
    raise SystemExit("Usage: python runtime_entry.py {train|ablation} [existing tool arguments]")
target = targets[sys.argv.pop(1)]
runpy.run_module(target, run_name="__main__")
