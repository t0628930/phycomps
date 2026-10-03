# Prepublication checks and conditions for public release

This private release candidate was prepared for `t0628930/phycomps-ijece-42922` on 3 October 2026. A commit to a private repository does not constitute public dissemination and does not, by itself, resolve reviewer comment C3. No publication DOI has been assigned.

Before changing the repository's visibility, the following conditions should be satisfied:

- Approve the definitive software license and verify the corresponding rights. The proposed policy requires a separate agreement for commercial use; `LICENSE_PENDING.md` is **not** an operative software license.
- Verify the author declarations and applicable institutional or third-party obligations.
- Distinguish explicitly among permissions concerning source data, aggregate results, and software.
- Exclude raw and processed sensor observations, trajectory arrays, and model checkpoints from this repository.
- Execute the regression tests, numerical-table verification, and release-gate tests.
- Regenerate `RELEASE_MANIFEST.csv` after **any** content modification. Changes in checksums must not be interpreted as evidence of historical source identity.
- Publish an immutable version and independently verify access without authentication. Only after these checks should the repository record public access as verified, identify the release as published, and provide its actual version identifier or DOI.
- Revise the manuscript's Data Availability statement, declarations, and response to reviewer C3 using the verified release information. Recompile the manuscript and confirm all page references in the response letter.

The underlying local evidence and submission PDFs remain outside this repository and have not been modified. Successful preparation of this repository does not imply full model retraining, independent agronomic calibration, or recovery of the original execution environment.
