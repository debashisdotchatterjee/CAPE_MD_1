# CAPE-MD Colab verification package

Open `CAPE_MD_complete_verification.ipynb` in Google Colab and run all cells. A GPU runtime is recommended.

The notebook includes a controlled synthetic molecular benchmark, matched direct-force and conservative baselines, CAPE curvature regularisation, rollout verification, ensemble uncertainty, split conformal calibration, and a separate revised MD17 aspirin analysis.

`CFG.fast_mode=True` is for an initial verification. Set it to `False` for the fuller publication experiment. All figures, CSV tables, checkpoints, configuration metadata and environment details are saved to `CAPE_MD_RESULTS.zip` and downloaded automatically in Colab.

The code never hard-codes a favourable result. It reports the observed winner for each lower-is-better metric and preserves negative findings. rMD17 velocities are generated only for optional numerical rollout diagnostics because velocity labels are not supplied by the dataset.
