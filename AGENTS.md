# Project instructions

- Purpose: polarization measurement and reconstruction research prototype, with PyTorch as the primary runnable backend.
- Install: Python 3.12; `python -m pip install -r requirements-dev.txt`. Optional TensorFlow uses a separate environment and `requirements-tf.txt`.
- Verify: `python -m pytest -q`; `python scripts/smoke_test.py --out-dir runs/<new-name>`.
- Current facts: README → docs/STATUS.md; experiment sequence → docs/EXPERIMENT.md; data contract → docs/DATA.md.
- Keep `polarized_csi/io.py` shared between label generation and both backends. Preserve sensor precision; do not per-image min/max normalize.
- Baseline is affine, two convolutions, 796 parameters, no activation. Architecture changes need a controlled experiment; Stokes S1–S3 may be negative.
- Demo labels have unresolved provenance and high raw-Stokes constraint violations. Use only as software examples. Generated smoke data are explicitly synthetic.
- `docs/archive/` is historical evidence; preserve its contents. Never promote unverified report metrics to current results.
- Raw captures, private data, model weights and run outputs stay ignored unless the user explicitly selects them for publication.
- Before changing training, add regression coverage for data integrity, validation selection and provenance. Do not call skipped tests verified.
- No real optical calibration or independent reconstruction performance has yet been established by the supplied materials.
