# Example renders

A curated subset of egocentric scene renders for illustration. The full scene
corpus (18,709 files) is withheld at review time and released on acceptance.

- `hazards/` — one representative render per hazard template (33 templates).
- `safe_controls/` — matched safe-control viewpoints (no hazardous state),
  illustrating the false-alarm evaluation design.
- `visual_noise/` — for a few scenes, the clean render alongside the same view
  under RGB-level corruption (`motion_blur`, `low_illumination`,
  `gaussian_sensor`) at intensity 0.7 (Appendix C).
