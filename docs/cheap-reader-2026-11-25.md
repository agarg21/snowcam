# Cheap cam reader, first prototype (2026-11-25)

**Question:** can a free-to-run image model give the same answers as our paid reader (the "teacher") on most webcam frames, so that only the hard frames go to the model?

**Short answer:** partly.
- A frozen CLIP model with one linear layer per field already matches the teacher on "is it snowing" (98%). It is close on new snow (93%) and snowmaking (93%).
- It is weaker on snow cover (89%) and visibility (81%).
- Used as a filter, it could skip the model call on 24–45% of frames while still agreeing with the teacher 91–100% of the time, depending on the field.
- At today's prices that saves little, because the cheap first read is only about a quarter of the spend. The bigger lever is the second-opinion checks (below).

## Data

| | |
|---|---|
| Frames with a teacher reading | 8,868 |
| Cams / resorts | 142 / 32 |
| Days | Nov 15–25, 2026 (109 daytime passes from the run artifacts) |
| Teacher | production reader: claude-haiku-5-5, escalating to claude-sonnet-5-5 |
| Second opinion | claude-sonnet-5-5 checks on a sample of the same frames (the publish check) |

Labels are the teacher's fields at confidence 0.6 or higher, mapped to the scales the pages publish:
- snow cover: good / thin
- new snow: none / some
- snowmaking: on / off
- snowing: no / yes
- visibility: clear / hazy / poor

Base-area busyness was skipped: 99% of base frames read "quiet", so there was nothing to learn.

## Method (lab/)

- **Embeddings:** open_clip ViT-B/32 (OpenAI weights), CPU only.
  - Each frame is embedded whole, plus the max over its 4 quadrant tiles, so small things like snow-gun plumes still register.
  - Each embedding is also taken relative to the cam's own average frame.
  - Cost: 0.20 s of CPU per frame (0.044 s without tiles).
- **Student:** one logistic regression per field.
- **Tests:**
  1. Leave-resort-out: train on 31 resorts and test on the one left out. This is the hardest test: a new resort the student has never seen.
  2. Time split: train on earlier days and test on the last two days. This is the realistic case for cams we already run.
  3. Hybrid: pick a confidence threshold per field on validation days, then measure on the last two days how often the student is sure about every field. In those cases the model call could be skipped.

## Results

| Field | Always guess the common answer | Student, new resort | Student, known cams (last 2 days) | Teacher vs Sonnet second opinion |
|---|---|---|---|---|
| Snowing | 92.0% | **97.7%** | 98.1% | 99.9% |
| New snow | 89.4% | 92.8% (finds 81% of "some") | 93.2% | 93.9% |
| Snowmaking | 88.9% | 93.1% (finds 83% of "on") | 88.2% | 99.6% |
| Snow cover | 59.8% | 88.5% | 88.0% | 91.7% |
| Visibility | 66.8% | 80.8% | 89.7% | 87.5% |

**Hybrid, known cams, last 2 days (1,428 frames):**

| Target agreement per field | Frames where the call could be skipped | Agreement with the teacher on those frames |
|---|---|---|
| 95% | 44.5% | snowing 99.7%, visibility 97.5%, new snow 96.3%, snow cover 93.5%, snowmaking 90.5% |
| 97% | 23.5% | snowing 99.7%, visibility 99.6%, new snow 98.2%, snow cover 97.4%, snowmaking 93.4% |

Thresholds picked on earlier days lose 1–5 points on later days. Snowmaking falls short of the target at both settings, so it would keep going to the model.

**Bigger model.** ViT-L/14 needs about 17× the CPU per frame. On the first 4,091 frames it did better on new snow (94.8%) and snowmaking (96.7%) for a new resort, but worse on snow cover (85.7%). It was not a clear win for the cost.

## What it would save

Model spend over the last week was about $2 a day for 32 resorts:

| Part | Share of the ~$2 a day |
|---|---|
| First reads (Haiku) | about $0.55 |
| Escalations (Sonnet) | about $0.70 |
| Second-opinion checks (Sonnet) | $0.75–1.50 |

Skipping 25–45% of first reads saves only $0.15–0.25 a day at this size. Bigger savings come from:
1. Using the student as a free second opinion: ask Sonnet only when the student and the teacher disagree.
2. Escalating to Sonnet only when the student is unsure.

At 300+ resorts this is the difference between about $20 and about $10 a day. That second figure is a projection, not a measurement.

## Next steps

1. Label with the stronger reader. Use the Sonnet second opinions, plus Sonnet on a sample of student-uncertain frames, as the teacher. Haiku's own errors cap what a student can learn from it.
2. Fine-tune instead of freezing. A small ViT or ConvNeXt fine-tuned on 10k+ frames, with per-cam reference frames as a second input (the teacher compares against a clear-day frame).
3. Keep collecting. The private archive (snowcam-lab releases) adds about 1,000 labelled daytime frames a day, and storm days are the rare cases that matter most.
4. Shadow mode before any switch. Run the student alongside the reader for a week and let the existing publish check judge it. Never lower the bar to make it pass.

Frames are private working data. Only these metrics leave the lab.

## Draft thread (not posted)

1. We read 142 ski-resort webcams across 32 US resorts every hour with a vision LLM, so families can see "is there snow, is it snowing" without squinting at cams. Can a free model do most of that work? A first test on 8,868 labelled frames:
2. Setup: frozen CLIP ViT-B/32 on CPU (0.2 s a frame), one linear layer per question, trained on the LLM's answers. Tested on resorts it never saw.
3. "Is it snowing?" 97.7% agreement with the LLM. New snow 92.8%, snowmaking 93.1%, snow cover 88.5%, visibility 80.8%.
4. As a filter it could skip the LLM on 24–45% of frames at 91–100% agreement, depending on the question. But first reads aren't the main cost; second-opinion checks are. Next: fine-tune on Sonnet labels, run in shadow mode, and let the same accuracy check that gates the pages judge it.
