# Release preparation and public-release gates

Prepared 3 October 2026 for public release. A private Git commit is not a public
release. There is no published DOI.

Before any visibility change:
- Software license approved on 3 October 2026: GPL-3.0-only, with a non-binding
  scientific citation request. See LICENSE, COPYRIGHT.md and CITATION.cff.
  The earlier noncommercial-only preference has been superseded.
- Confirm the author declarations and any institutional/third-party constraints.
- Retain the distinction between source data, aggregate results and code permissions.
- Keep raw/prepared sensor rows, trajectory arrays and checkpoints out of this repository.
- Run the regression tests, result-table checks and release-gate tests.
- Regenerate RELEASE_MANIFEST.csv after any content edit; do not reinterpret changed
  checksums as proof of historical-source identity.
- Publish a fixed version and independently verify unauthenticated access. Only then
  record public_access_verified, published, and the actual release identifier/DOI.
- Update the relevant manuscript data-availability statement with that verified
  information and recompile the manuscript.

The original local evidence and submission PDFs remain outside this repository
and unchanged. Full retraining, agronomic calibration and historical-host recovery
are not implied by successful repository preparation.

## Update on 4 October 2026
The authors confirmed the funding, conflicts, CRediT and AI-assistance statements.
Public access was authorized and anonymously verified on 4 October 2026. No DOI
is claimed. This repository is journal-neutral; release names and manuscript
links should likewise remain journal-neutral.
