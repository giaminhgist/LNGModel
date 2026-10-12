# The 45 hand-crafted eye-movement features (EMS)

Reference for the 45-feature stimulus representation used by every
experiment in this repository. Authoritative implementation:
`src/features/stimulus.py::compute_stimulus_features` (single source of truth; this
document mirrors it).

## Computation context

- Computed **per (subject, stimulus)** from the *cleaned* fixations of that
  stimulus, ordered by `FIX_INDEX` (cleaning rules:
  `src/scripts/preprocess.py::clean_fixations`).
- Input arrays: `x, y` — fixation coordinates in **pixels** (screen
  1024 × 768, Eyelink display); `dur` — fixation duration in **ms**;
  `pup` — pupil size in **Eyelink arbitrary units**.
- `n` = number of clean fixations of the stimulus; `n < 2` stimuli are
  missing (all-NaN rows) by construction (`MIN_FIX = 2`).
- Standard deviations use `ddof = 1` (sample SD).
- **Missingness policy** (see `src/data/tabular.py`): a feature that is NaN
  in an otherwise valid row is a *missing feature* and is filled with the
  train-fitted per-feature mean, so it contributes zero deviation; an
  all-NaN row is a *missing stimulus* and is masked everywhere.
- Interpretation notes: clinical statements are **associations reported in
  the eye-tracking literature**, not diagnostic claims, and none of these
  features is an established biomarker on its own.

## Overview table (45 features)

| # | feature | group | unit / range | one-line meaning |
|---|---|---|---|---|
| 1 | `spa_fix_count` | spatial | count | number of fixations |
| 2 | `spa_mean_x` | spatial | px | mean fixation x |
| 3 | `spa_mean_y` | spatial | px | mean fixation y |
| 4 | `spa_std_x` | spatial | px | horizontal spread |
| 5 | `spa_std_y` | spatial | px | vertical spread |
| 6 | `spa_dispersion` | spatial | px | mean distance to the centroid |
| 7 | `spa_bbox_area` | spatial | px² | area of the bounding box of fixations |
| 8 | `spa_center_dist_mean` | spatial | px | mean distance to screen center |
| 9 | `spa_center_dist_std` | spatial | px | variability of center distance |
| 10 | `spa_center_frac` | spatial | [0,1] | fraction inside the central quarter |
| 11 | `spa_q1` | spatial | [0,1] | fraction in quadrant 1 (top-left) |
| 12 | `spa_q2` | spatial | [0,1] | fraction in quadrant 2 (top-right) |
| 13 | `spa_q3` | spatial | [0,1] | fraction in quadrant 3 (bottom-left) |
| 14 | `spa_q4` | spatial | [0,1] | fraction in quadrant 4 (bottom-right) |
| 15 | `spa_entropy` | spatial | [0,1] | uniformity of the 8×6 fixation histogram |
| 16 | `spa_max_grid_frac` | spatial | [0,1] | peak density of one 8×6 cell |
| 17 | `spa_skew_x` | spatial | dimensionless | horizontal asymmetry of the fixation cloud |
| 18 | `geo_scanpath_len` | scanpath | px | total path length |
| 19 | `geo_sacc_amp_mean` | scanpath | px | mean step amplitude |
| 20 | `geo_sacc_amp_std` | scanpath | px | variability of step amplitude |
| 21 | `geo_sacc_amp_max` | scanpath | px | largest step amplitude |
| 22 | `geo_dx_mean` | scanpath | px | mean horizontal step (bias) |
| 23 | `geo_dy_mean` | scanpath | px | mean vertical step (bias) |
| 24 | `geo_angle_var` | scanpath | [0,1] | directional concentration (0 = straight, 1 = uniform) |
| 25 | `geo_revisit_rate` | scanpath | [0,1] | fraction of fixations revisiting an earlier location |
| 26 | `geo_nn_dist_mean` | scanpath | px | mean nearest-neighbour distance between fixations |
| 27 | `geo_hull_area` | scanpath | px² | convex-hull area of the fixation cloud |
| 28 | `tem_dur_mean` | temporal | ms | mean fixation duration |
| 29 | `tem_dur_std` | temporal | ms | variability of fixation duration |
| 30 | `tem_dur_total` | temporal | ms | summed dwell time |
| 31 | `tem_dur_max` | temporal | ms | longest fixation |
| 32 | `tem_first_dur` | temporal | ms | duration of the first fixation |
| 33 | `tem_last_dur` | temporal | ms | duration of the last fixation |
| 34 | `tem_ifi_mean` | temporal | ms | mean inter-fixation interval (approximation) |
| 35 | `tem_ifi_std` | temporal | ms | variability of the inter-fixation interval |
| 36 | `tem_velocity_mean` | temporal | px/ms | mean saccade velocity (approximation) |
| 37 | `tem_fix_rate` | temporal | fix/s | fixations per second of viewing |
| 38 | `tem_trans_entropy` | temporal | [0,1] | uniformity of fixation locations over the visit order |
| 39 | `pup_mean` | pupil | AU | mean pupil size |
| 40 | `pup_std` | pupil | AU | variability of pupil size |
| 41 | `pup_min` | pupil | AU | smallest pupil size |
| 42 | `pup_max` | pupil | AU | largest pupil size |
| 43 | `pup_median` | pupil | AU | median pupil size |
| 44 | `pup_slope` | pupil | AU/fix | linear trend of pupil over fixation order |
| 45 | `pup_first_last_diff` | pupil | AU | pupil change from first to last fixation |

Groups: **spatial** (1–17), **scanpath geometry** (18–27), **temporal**
(28–38), **pupil** (39–45). Prefixes `spa_`, `geo_`, `tem_`, `pup_`.

---

## Group A — spatial: position & dispersion (features 1–17)

Notation: `cx = 512, cy = 384` (screen center), `w = 1024, h = 768`.

| feature | formula (code) | NaN when | behavioral / clinical interpretation |
|---|---|---|---|
| `spa_fix_count` | `n` | never (n ≥ 2 for valid rows) | total exploration effort on this stimulus; fewer fixations in free viewing have been reported in schizophrenia (restricted scanning) |
| `spa_mean_x` | `mean(x)` | never | horizontal center of gaze; mixes stimulus content bias (salient regions) with subject bias |
| `spa_mean_y` | `mean(y)` | never | vertical center of gaze; same interpretation |
| `spa_std_x` | `x.std(ddof=1)` | `n < 2` | horizontal extent of exploration; lower spread ↔ restricted scanning |
| `spa_std_y` | `y.std(ddof=1)` | `n < 2` | vertical extent of exploration |
| `spa_dispersion` | `mean(sqrt((x-mean(x))² + (y-mean(y))²))` | never | average distance of fixations from their own centroid; a global exploration-extent measure |
| `spa_bbox_area` | `(max(x)-min(x)) · (max(y)-min(y))` | never | area actually explored; sensitive to a single far outlier, unlike `spa_dispersion` |
| `spa_center_dist_mean` | `mean(sqrt((x-cx)² + (y-cy)²))` | never | how far from screen center the gaze stays; the central fixation bias is normal, its magnitude differs across populations and tasks |
| `spa_center_dist_std` | `center distances .std(ddof=1)` | `n < 2` | consistency of the center bias |
| `spa_center_frac` | `mean(|x-cx| < w/4 & |y-cy| < h/4)` | never | fraction of fixations inside the central quarter of the screen; a compact center-bias measure |
| `spa_q1..spa_q4` | `mean(x<cx & y<cy)` etc. | never | per-quadrant attention allocation (screen quadrants, top-left/top-right/bottom-left/bottom-right); q1+q2+q3+q4 = 1 |
| `spa_entropy` | Shannon entropy of the 8×6 fixation histogram, divided by `log(48)` | `n == 0` (not reached) | how evenly fixations cover the screen: 1 = uniform exploration, 0 = all fixations in one cell |
| `spa_max_grid_frac` | `max(8×6 histogram)/n` | never | peak spatial density: how concentrated fixations are in the single most-visited cell |
| `spa_skew_x` | `mean((x-mean(x))³) / std(x)³` | `n < 3` or `std(x)==0` | asymmetry of the horizontal fixation distribution; large absolute values ↔ exploration skewed to one side |

## Group B — scanpath geometry (features 18–27)

Steps between consecutive fixations: `dx = diff(x)`, `dy = diff(y)`,
`amp = sqrt(dx² + dy²)`. These treat each consecutive-fixation transition as
an (inferred) saccade — the EMS release does not contain explicit saccade
events.

| feature | formula (code) | NaN when | behavioral / clinical interpretation |
|---|---|---|---|
| `geo_scanpath_len` | `sum(amp)` | `n < 2` | total distance travelled; shorter scanpaths are among the most replicated free-viewing findings in schizophrenia |
| `geo_sacc_amp_mean` | `mean(amp)` | `n < 2` | mean jump size between consecutive fixations; smaller amplitudes ↔ more local/restricted scanning |
| `geo_sacc_amp_std` | `amp.std(ddof=1)` | `n < 2` | variability of jump size; mixes ambient (large jumps) and focal (small jumps) phases |
| `geo_sacc_amp_max` | `max(amp)` | `n < 2` | largest single jump; outlier-sensitive |
| `geo_dx_mean` | `mean(dx)` | `n < 2` | net horizontal drift of the scanpath (reading-like or side-biased scanning) |
| `geo_dy_mean` | `mean(dy)` | `n < 2` | net vertical drift of the scanpath |
| `geo_angle_var` | `1 - sqrt(mean(cos a)² + mean(sin a)²)`, `a = atan2(dy, dx)` | `n < 2` | directional concentration: 0 = all steps in the same direction, 1 = directions uniform over the circle; lower values ↔ stereotyped scanning |
| `geo_revisit_rate` | fraction of fixations landing within 60 px of an *earlier* fixation | `n < 2` | how often the gaze returns to already-inspected locations; related to checking/inspection behaviour and working-memory-guided search |
| `geo_nn_dist_mean` | mean over fixations of the distance to the nearest *other* fixation | `n < 2` | local crowding of fixations; complements global spread measures |
| `geo_hull_area` | area of the convex hull (`scipy.spatial.ConvexHull`) | `n < 4` or hull fails | total explored region, robust to interior gaps; degenerate fixations on a line give 0 |

## Group C — temporal (features 28–38)

`half_gap = (dur[:-1] + dur[1:]) / 2` approximates the inter-fixation
interval (the EMS release has no saccade onset timestamps; the standard
approximation is half of each adjacent dwell time).

| feature | formula (code) | NaN when | behavioral / clinical interpretation |
|---|---|---|---|
| `tem_dur_mean` | `mean(dur)` | never | average fixation duration; a basic processing-depth / dwell measure |
| `tem_dur_std` | `dur.std(ddof=1)` | `n < 2` | regularity of dwell time; erratic dwell distributions are reported in some psychiatric populations |
| `tem_dur_total` | `sum(dur)` | never | total dwell time on the stimulus; correlates with `spa_fix_count` |
| `tem_dur_max` | `max(dur)` | never | longest dwell; catches prolonged single fixations (e.g. staring/lapses) |
| `tem_first_dur` | `dur[0]` | never | dwell on the first fixation (initial capture of the stimulus) |
| `tem_last_dur` | `dur[-1]` | never | dwell on the last fixation (end-of-viewing behaviour) |
| `tem_ifi_mean` | `mean(half_gap)` | `n < 2` | mean pause between successive fixation centers; longer IFI ↔ slower scanning rhythm |
| `tem_ifi_std` | `half_gap.std(ddof=1)` | `n < 2` | regularity of the scanning rhythm |
| `tem_velocity_mean` | `mean(amp/half_gap)` over finite values | `n < 2` or none finite | mean travel speed between fixations (px/ms); mixes saccade speed and pause length |
| `tem_fix_rate` | `n / total_time · 1000`, `total_time = sum(dur) + ifi_mean·(n-1)` | `total_time == 0` | fixations per second; a normalized exploration-rate measure, more comparable across viewing lengths than `spa_fix_count` |
| `tem_trans_entropy` | Shannon entropy of `(x[:-1], y[:-1])` over a 4×3 grid, normalized by `log(12)` | `n < 2` | how uniformly the *starting points* of steps cover the screen — a coarse scanpath-shape descriptor |

## Group D — pupil (features 39–45)

Pupil values are in Eyelink arbitrary units (AU). Pupil size responds to
luminance, arousal, cognitive load and medication; in this dataset each
stimulus is analysed independently, so luminance confounds are stimulus-
specific and largely absorbed by the normative (HC-bank) comparison.

| feature | formula (code) | NaN when | behavioral / clinical interpretation |
|---|---|---|---|
| `pup_mean` | `mean(pup)` | never | average pupil size (arousal/load proxy) |
| `pup_std` | `pup.std(ddof=1)` | `n < 2` | temporal variability of pupil size (arousal fluctuations) |
| `pup_min` | `min(pup)` | never | minimal pupil size |
| `pup_max` | `max(pup)` | never | maximal pupil size; range `pup_max - pup_min` reflects phasic response |
| `pup_median` | `median(pup)` | never | robust center of the pupil distribution |
| `pup_slope` | slope of `polyfit(0..n-1, pup, 1)` | `n < 2` | linear trend of pupil over the visit (AU per fixation step); negative slopes ↔ habituation/dilation decay, positive ↔ accumulating load |
| `pup_first_last_diff` | `pup[-1] - pup[0]` | `n < 2` | net pupil change across the stimulus (cruder version of `pup_slope`) |

## Notes and caveats

- **Correlations**: many features are collinear (e.g. `spa_fix_count`,
  `tem_dur_total`, `geo_scanpath_len` all scale with viewing length;
  `spa_std_*` with `spa_bbox_area`; `tem_first_dur`/`tem_last_dur` with
  `tem_dur_mean`). The model is not asked to interpret them individually.
- **Units**: no degrees-of-visual-angle conversion is applied anywhere
  (distances are raw pixels on the 1024 × 768 display); any degree-based
  interpretation requires the viewing geometry, which the EMS release does
  not fully document for our setup.
- **Approximations**: saccade-related quantities (`geo_sacc_amp_*`,
  `tem_ifi_*`, `tem_velocity_mean`, `geo_angle_var`) are computed from
  consecutive fixation centers, not from recorded saccade events; IFI uses
  the half-dwell approximation.
- **Clinical interpretation**: the statements above summarize associations
  reported in free-viewing eye-tracking literature (e.g. restricted
  scanpaths in schizophrenia, Benson et al. 2012). They describe typical
  group-level patterns, not individual diagnostic thresholds, and this
  pipeline only uses the features as model inputs — feature-level group
  differences are not part of the current reports.
- **Stimulus content**: features mix subject behaviour with stimulus
  content (e.g. `spa_mean_x` depends on the image). The normative model
  (EXP-PROP-001) is stimulus-conditioned precisely to separate the two.
