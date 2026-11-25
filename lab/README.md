# Cheap-reader lab

Can a small, free-to-run image model match the paid reader (the "teacher") on most frames, and leave only the hard ones to the model?

- `embed.py ART_DIR DB OUT.npz MODEL PRETRAINED`: embeds every archived frame that has a teacher reading, using open_clip on CPU. Set `TILES=1` to add the max over 4 quadrant tiles.
- `probe.py OUT.npz DB registry.yaml`: trains one logistic-regression probe per field and reports three things against the teacher:
  - leave-resort-out agreement (a resort the probe never saw);
  - a time split (the last 2 days);
  - coverage at 95% agreement.

  It also reports the teacher-vs-Sonnet second-opinion agreement on the same reads. That is the bar the teacher itself meets.
- `joint.py OUT.npz [0.95]`: the hybrid. Per-field confidence thresholds are picked on validation days. It measures the share of test frames where every field is confident, so the model call could be skipped.

Frames come from the private run artifacts and the snowcam-lab releases. They are working data and are never committed or shared; only metrics leave the lab. Writeups are in `docs/`.
