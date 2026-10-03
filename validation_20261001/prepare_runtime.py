"""Build a narrowly corrected runtime copy; preserve the frozen evidence unchanged."""
import difflib
from pathlib import Path
from validation_common import FROZEN, ROOT, sha256, write_json

source = FROZEN / "scripts/pc_mps.py"
original = source.read_text(encoding="utf-8-sig")
replacements = [
    ("    def log_prob(self, indices: torch.Tensor) -> torch.Tensor:\n        return 2.0 * self._log_abs_amplitude(indices) - self.log_norm()",
     "    def log_prob(self, indices: torch.Tensor, log_norm: Optional[torch.Tensor] = None) -> torch.Tensor:\n        if log_norm is None:\n            log_norm = self.log_norm()\n        return 2.0 * self._log_abs_amplitude(indices) - log_norm"),
    ("    def allowed_log_barrier(self, allowed_per_site: Sequence[torch.Tensor], per_site: bool = True) -> torch.Tensor:",
     "    def allowed_log_barrier(self, allowed_per_site: Sequence[torch.Tensor], per_site: bool = True, log_norm: Optional[torch.Tensor] = None) -> torch.Tensor:"),
    ("        penalty = (self.log_norm() - log_allowed).clamp_min(0.0)",
     "        if log_norm is None:\n            log_norm = self.log_norm()\n        penalty = (log_norm - log_allowed).clamp_min(0.0)"),
]
changed = original
for old, new in replacements:
    if changed.count(old) != 1:
        raise ValueError("Frozen source no longer matches the reviewed cache-hook patch")
    changed = changed.replace(old, new)
target = ROOT / "runtime/pc_mps.py"
target.parent.mkdir(exist_ok=True)
if target.exists() and target.read_text(encoding="utf-8") != changed:
    raise FileExistsError("Existing runtime differs; inspect it before replacing")
target.write_text(changed, encoding="utf-8")
(ROOT / "runtime/cache_compatibility.patch").write_text("".join(difflib.unified_diff(
    original.splitlines(True), changed.splitlines(True),
    fromfile="Frozen_Reproducibility/scripts/pc_mps.py", tofile="validation_20261001/runtime/pc_mps.py")), encoding="utf-8")
write_json(ROOT / "runtime/provenance.json", {
    "source_sha256": sha256(source), "runtime_sha256": sha256(target),
    "reason": "Frozen mixture uses log_norm keyword unsupported by frozen GenerativeMPS.",
    "changes": "Two optional log_norm cache hooks; no change to sampling, contractions, parameters or architecture.",
    "historical_training_source_equivalence": "not established; this is an explicit revision compatibility fix",
})
print("Runtime compatibility copy prepared; frozen source unchanged.")
