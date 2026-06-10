"""Supplementary (附加) experiments for the SafeSight paper.

Two self-contained studies that *do not* require re-running the full
137k-record benchmark:

    mitigation/    — Yes-man false-alarm mitigation (Table 4)
                     · offline confidence-gated calibration baseline
                     · cognitive self-correction re-prompting (main method)

    visual_noise/  — RGB-level physical sensor corruption (Appendix C)
                     · motion blur / low illumination / sensor Gaussian noise
                     · feasibility study on image-track robustness

All inputs are the *existing* result JSONs under ``data/results/`` and the
clean scene renders under ``data/scenes/``.  All new artefacts are written
under ``data/supplementary/`` so nothing in the main pipeline is touched.
"""
