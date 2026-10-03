# Provenance and release transformations

`SOURCE_PROVENANCE.csv` records each copied file's revision-relative source path,
original and release-candidate SHA-256 values. Metadata serialization can change
bytes without changing scientific values; path-valued fields may be redacted.
`REDACTED_METADATA_FIELDS.json` records field locations, not private original values.
The original private 80-file snapshot remains unchanged outside this candidate.

The MPS runtime overlay adds two optional normalization-cache arguments. Its patch
and original source are both supplied; tests compare values and gradients.
The recovered pc_mps_train.py dependency is copied unchanged. These facts do not
prove byte-identical source use in every historical training session.

Source inspection found no explicit license file in the selected frozen code.
Author authorization was supplied for scripts/results; that does not resolve any
unidentified third-party ownership. Preserve notices and confirm provenance before
applying a software license. This is not a comprehensive legal or security audit.
