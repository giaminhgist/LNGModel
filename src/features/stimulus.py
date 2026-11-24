"""Hand-crafted spatial + temporal feature computation for one (subject, stimulus).

Each function takes the *cleaned* fixations of a single stimulus as numpy arrays
ordered by FIX_INDEX and returns a dict of scalar features (NaN-safe).
"""
import numpy as np
from scipy.spatial import ConvexHull

GRID_COLS, GRID_ROWS = 8, 6          # spatial histogram grid
TRANS_GRID_COLS, TRANS_GRID_ROWS = 4, 3   # coarser grid for transitions
REVISIT_RADIUS = 60.0                # px, ~1 deg at 0.6 m viewing distance


def _grid_entropy(x, y, cols, rows, w, h):
    """Normalized Shannon entropy of the (cols x rows) spatial histogram."""
    if len(x) == 0:
        return np.nan
    hist, _ = np.histogramdd(np.column_stack([x, y]), bins=(cols, rows),
                             range=((0, w), (0, h)))
    hist = hist.ravel().astype(np.float64)
    hist /= hist.sum()
    hist = hist[hist > 0]
    n_max = np.log(cols * rows)
    return -np.sum(hist * np.log(hist)) / n_max if n_max > 0 else np.nan


def compute_stimulus_features(x, y, dur, pup, screen_w=1024.0, screen_h=768.0):
    """Compute the 45 hand-crafted features for one stimulus.

    x, y, dur, pup: 1-D arrays (cleaned fixations, ordered by FIX_INDEX).
    Returns dict feature_name -> float (NaN where undefined).
    """
    f = {}
    n = len(x)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    dur = np.asarray(dur, dtype=np.float64)
    pup = np.asarray(pup, dtype=np.float64)
    cx, cy = screen_w / 2, screen_h / 2

    # ---- A. position & dispersion ----
    f["spa_fix_count"] = float(n)
    f["spa_mean_x"] = x.mean()
    f["spa_mean_y"] = y.mean()
    f["spa_std_x"] = x.std(ddof=1) if n > 1 else np.nan
    f["spa_std_y"] = y.std(ddof=1) if n > 1 else np.nan
    d_cent = np.sqrt((x - x.mean()) ** 2 + (y - y.mean()) ** 2)
    f["spa_dispersion"] = d_cent.mean()
    f["spa_bbox_area"] = (x.max() - x.min()) * (y.max() - y.min())

    # ---- B. center & region statistics ----
    d_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    f["spa_center_dist_mean"] = d_center.mean()
    f["spa_center_dist_std"] = d_center.std(ddof=1) if n > 1 else np.nan
    f["spa_center_frac"] = float(np.mean((np.abs(x - cx) < screen_w / 4)
                                         & (np.abs(y - cy) < screen_h / 4)))
    f["spa_q1"] = float(np.mean((x < cx) & (y < cy)))
    f["spa_q2"] = float(np.mean((x >= cx) & (y < cy)))
    f["spa_q3"] = float(np.mean((x < cx) & (y >= cy)))
    f["spa_q4"] = float(np.mean((x >= cx) & (y >= cy)))
    f["spa_entropy"] = _grid_entropy(x, y, GRID_COLS, GRID_ROWS, screen_w, screen_h)
    hist, _ = np.histogramdd(np.column_stack([x, y]), bins=(GRID_COLS, GRID_ROWS),
                             range=((0, screen_w), (0, screen_h)))
    f["spa_max_grid_frac"] = float(hist.max() / n)
    f["spa_skew_x"] = float(((x - x.mean()) ** 3).mean() / (x.std(ddof=1) ** 3)) \
        if n > 2 and x.std(ddof=1) > 0 else np.nan

    # ---- C. scanpath geometry ----
    if n > 1:
        dx = np.diff(x)
        dy = np.diff(y)
        amp = np.sqrt(dx ** 2 + dy ** 2)
        f["geo_scanpath_len"] = float(amp.sum())
        f["geo_sacc_amp_mean"] = float(amp.mean())
        f["geo_sacc_amp_std"] = float(amp.std(ddof=1))
        f["geo_sacc_amp_max"] = float(amp.max())
        f["geo_dx_mean"] = float(dx.mean())
        f["geo_dy_mean"] = float(dy.mean())
        angles = np.arctan2(dy, dx)
        R = np.sqrt(np.cos(angles).mean() ** 2 + np.sin(angles).mean() ** 2)
        f["geo_angle_var"] = float(1.0 - R)   # 0 = all same direction, 1 = uniform
        # revisit: fixation landing within REVISIT_RADIUS of an earlier fixation
        xy = np.column_stack([x, y])
        dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
        earlier = np.triu(dist < REVISIT_RADIUS, k=1)
        f["geo_revisit_rate"] = float(earlier.any(axis=1).mean())
        # nearest-neighbor distance
        np.fill_diagonal(dist, np.inf)
        f["geo_nn_dist_mean"] = float(dist.min(axis=1).mean())
    else:
        for k in ["geo_scanpath_len", "geo_sacc_amp_mean", "geo_sacc_amp_std",
                  "geo_sacc_amp_max", "geo_dx_mean", "geo_dy_mean", "geo_angle_var",
                  "geo_revisit_rate", "geo_nn_dist_mean"]:
            f[k] = np.nan
    if n >= 4:
        try:
            hull = ConvexHull(np.column_stack([x, y]))
            f["geo_hull_area"] = float(hull.volume)
        except Exception:
            f["geo_hull_area"] = np.nan
    else:
        f["geo_hull_area"] = np.nan

    # ---- D. temporal ----
    f["tem_dur_mean"] = float(dur.mean())
    f["tem_dur_std"] = float(dur.std(ddof=1)) if n > 1 else np.nan
    f["tem_dur_total"] = float(dur.sum())
    f["tem_dur_max"] = float(dur.max())
    f["tem_first_dur"] = float(dur[0])
    f["tem_last_dur"] = float(dur[-1])
    # inter-fixation interval (IFI): saccade time between fixation i and i+1.
    # Onset timestamps are not recorded, so the standard approximation is used:
    # IFI_i = (dur_i + dur_{i+1}) / 2 (half of each adjacent dwell time).
    if n > 1:
        half_gap = (dur[:-1] + dur[1:]) / 2.0
        f["tem_ifi_mean"] = float(half_gap.mean())
        f["tem_ifi_std"] = float(half_gap.std(ddof=1))
        amp = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            vel = amp / half_gap
        vel = vel[np.isfinite(vel)]
        f["tem_velocity_mean"] = float(vel.mean()) if len(vel) else np.nan
    else:
        f["tem_ifi_mean"] = f["tem_ifi_std"] = f["tem_velocity_mean"] = np.nan
    total_time = dur.sum() + (f["tem_ifi_mean"] * (n - 1) if n > 1 else 0.0)
    f["tem_fix_rate"] = float(n / total_time * 1000.0) if total_time > 0 else np.nan
    f["tem_trans_entropy"] = _grid_entropy(x[:-1], y[:-1], TRANS_GRID_COLS,
                                           TRANS_GRID_ROWS, screen_w, screen_h) \
        if n > 1 else np.nan

    # ---- E. pupil ----
    f["pup_mean"] = float(pup.mean())
    f["pup_std"] = float(pup.std(ddof=1)) if n > 1 else np.nan
    f["pup_min"] = float(pup.min())
    f["pup_max"] = float(pup.max())
    f["pup_median"] = float(np.median(pup))
    if n > 1:
        idx = np.arange(n, dtype=np.float64)
        slope = np.polyfit(idx, pup, 1)[0]
        f["pup_slope"] = float(slope)
    else:
        f["pup_slope"] = np.nan
    f["pup_first_last_diff"] = float(pup[-1] - pup[0]) if n > 1 else np.nan

    return f


# 3-point probe: 2 points would make ddof=1 stds degenerate (n-1=1 diff)
FEATURE_NAMES = list(compute_stimulus_features(
    np.array([512.0, 514.0, 520.0]), np.array([384.0, 390.0, 400.0]),
    np.array([200.0, 250.0, 300.0]), np.array([1300.0, 1320.0, 1350.0])).keys())
