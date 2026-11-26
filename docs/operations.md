# Operating the cam reader

The reader runs about 13 passes a day (`collect.yml`). Once a day, `health.yml` checks the database and the run list. It makes no model calls. It keeps one open issue labelled `health`: the issue body is rewritten each day, a comment is added only when a new finding turns up, and the issue is closed when everything is clear.

## Rules

1. **Fix the cause, never the bar.** The publish check (about 100 second-opinion checks, 90% agreement, 85% per snow field) stays as it is. When a resort fails, the fix is to the cam, the field or the gate input.
2. **Fix-twice rule.** If the same finding comes back after a fix, or the same fix is needed for a second cam, change the code or the registry rule so the whole class is handled. Don't patch a third case by hand. Record the change in the commit message, with the finding key, for example `cam:park-city-topobx`.
3. **Stale means hidden.** A resort or field the check doesn't support is left off the pages. Never paper over it with older reads.
4. **One finding, one owner.** Whoever picks up a finding leaves a note in the health issue (what they changed, the commit). A finding with no note after 3 days is raised with the owner.

## Playbooks

| Key | What it means | Standard fix |
|---|---|---|
| `stuck:<resort>` (1) | Not published 3+ days after its first read | Look at the weakest fields in the line. Block that field for the weak cam (`block:` in registry.yaml) rather than for the resort. |
| `stalled:<resort>` (2) | No new checks in 48 h before reaching 100 | Usually every cam is gated out or not read. See that resort's `cam:` findings and the latest `report.html` in the run artifact. |
| `recheck:<resort>` (2b) | Taken off the pages, or used all 100 checks without passing; re-checks run at 2 a pass | Only checks since it was taken off count (the latest 100), at the same bar. If it is still under the bar after 100, find the cam and field that fail most and fix or block them. Per-cam blocks still count every check ever made, so a fixed cam recovers slowly. |
| `unpublished:<resort>` (3) | Taken off the pages: early-release snow fields fell below 85%, or spot checks slipped | Read the reason. Fix the cam or field. |
| `no_read:<resort>` (4) | No usable read in 24 h | Check that the cam URLs still serve images (feed moved, player changed). Otherwise it's night, fog or closed lifts everywhere: nothing to fix. |
| `cam:<cam>` (5) | Fetched 8+ times in 48 h, never read | If the gate reason repeats (`moved`, `stale`, `dark`), fix the gate input (new reference frame, crop, `night_ok`) or mark the cam unusable. Don't keep paying to fetch it. |
| `runs_failed` (6) | A collect run failed | Open the log. Two failures in a row is a code fix, not a re-run. |
| `runs_missing` (6) | Fewer than 9 passes in 24 h | GitHub delayed or dropped runs. Act only if it repeats 3 days. |
| `spend` (7) | Yesterday's estimated spend is above 1.5x the 7-day median | Check for a new resort, a skip rule that stopped firing (`savings`), or verification picking too many reads. |

Spend is estimated from calls × the measured per-call cost (`EST_COST` in cli.py), not from the bill.

## Re-checks (owner, 2026-11-26)

A resort taken off the pages, or one that used all 100 checks without passing, keeps getting 2 Sonnet checks a pass, about $0.03 a pass per resort. It is judged only on checks made since it was taken off (the latest 100), at the unchanged bar. Old passing checks never put it back. It can come back early at 20 fresh checks over 2 days, like a new resort, and then checking continues to 100.
