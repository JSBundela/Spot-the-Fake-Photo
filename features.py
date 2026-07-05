"""
Shared, hand-engineered feature extractor for the real-vs-screen classifier.

Every feature here targets a specific physical artifact of photographing a
screen rather than a real scene: moire interference, color-gamut/contrast
compression, glare, loss of fine texture, bezel edges, and JPEG
recompression blockiness. Kept to classic, cheap image-processing ops
(FFT, Canny/Hough, Laplacian, color histograms) on a small, fixed-size
version of the image so this stays fast enough to run on a phone.

FEATURE_NAMES defines the fixed vector order used by both training and
predict.py.
"""
import cv2
import numpy as np

WORK_SIZE = 384  # longest edge used for all analysis; keeps runtime small & constant
RADIAL_BINS = 10  # radial power-spectrum bins used for the 1/f deviation features

FEATURE_NAMES = [
    "fft_high_freq_ratio",
    "fft_peak_ratio",
    "fft_peak_count",
    "autocorr_peak_strength",
    "sat_mean",
    "sat_std",
    "unique_color_ratio",
    "channel_gain_r",
    "channel_gain_b",
    "highlight_ratio",
    "highlight_softness",
    "laplacian_var",
    "hf_noise_energy",
    "border_line_score",
    "block_artifact_score",
    "channel_xcorr_hf",
] + [f"radial_bin_{i}" for i in range(RADIAL_BINS)] \
  + [f"radial_resid_{i}" for i in range(RADIAL_BINS - 2)]


def _resize_longest(bgr, size=WORK_SIZE):
    h, w = bgr.shape[:2]
    scale = size / max(h, w)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return bgr


def _fft_features(gray, n_radial_bins=RADIAL_BINS):
    """High-freq energy ratio and peak spikiness (generic moire/aliasing
    indicators), plus a radial power-spectrum profile and its residual from
    the fitted 1/f natural-image falloff -- real photos decay smoothly,
    screen moire shows up as a bump above the fit. The residual is the
    single biggest accuracy contributor."""
    h, w = gray.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32) * win))
    mag = np.abs(f)
    mag[h // 2, w // 2] = 0  # zero out DC
    log_mag = np.log1p(mag)

    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2, w / 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_norm = r / r.max()

    total = mag.sum() + 1e-8
    high_mask = r_norm > 0.5
    fft_high_freq_ratio = mag[high_mask].sum() / total

    band_mask = (r_norm > 0.15) & (r_norm < 0.85)
    band_vals = mag[band_mask]
    med = np.median(band_vals) + 1e-6
    fft_peak_ratio = band_vals.max() / med
    fft_peak_count = float(np.sum(band_vals > 8 * med))

    rmax = r.max()
    bin_idx = np.clip((r / rmax * n_radial_bins).astype(np.int32), 0, n_radial_bins - 1)
    sums = np.bincount(bin_idx.ravel(), weights=log_mag.ravel(), minlength=n_radial_bins)
    counts = np.bincount(bin_idx.ravel(), minlength=n_radial_bins)
    profile = sums / np.maximum(counts, 1)

    bin_centers = (np.arange(n_radial_bins) + 0.5) * rmax / n_radial_bins
    x = np.log(bin_centers[1:-1] + 1e-3)
    yv = profile[1:-1]
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, yv, rcond=None)[0]
    residual = yv - (slope * x + intercept)

    return fft_high_freq_ratio, fft_peak_ratio, fft_peak_count, profile, residual


def _autocorr_feature(gray):
    g = gray.astype(np.float32)
    g = (g - g.mean())
    h, w = g.shape
    f = np.fft.fft2(g)
    ac = np.fft.ifft2(f * np.conj(f)).real
    ac = np.fft.fftshift(ac)
    ac = ac / (ac.max() + 1e-8)

    cy, cx = h // 2, w // 2
    exclude = max(4, min(h, w) // 40)
    mask = np.ones_like(ac, dtype=bool)
    mask[cy - exclude:cy + exclude, cx - exclude:cx + exclude] = False
    ring = ac[mask]
    return float(np.percentile(ring, 99.9))


def _color_features(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1].astype(np.float32)
    sat_mean = sat.mean() / 255.0
    sat_std = sat.std() / 255.0

    small = cv2.resize(bgr, (96, 96), interpolation=cv2.INTER_AREA)
    quant = (small >> 3).astype(np.uint32).reshape(-1, 3)
    packed = (quant[:, 0] << 10) | (quant[:, 1] << 5) | quant[:, 2]
    unique = len(np.unique(packed))
    unique_color_ratio = unique / (96 * 96)

    b, g, r = cv2.split(bgr.astype(np.float32))
    g_mean = g.mean() + 1e-6
    channel_gain_r = r.mean() / g_mean
    channel_gain_b = b.mean() / g_mean

    return sat_mean, sat_std, unique_color_ratio, channel_gain_r, channel_gain_b


def _highlight_features(gray, lap):
    bright_mask = gray > 240
    ratio = bright_mask.mean()
    if bright_mask.sum() < 20:
        return float(ratio), 0.0
    softness = 1.0 / (1.0 + lap[bright_mask].std())
    return float(ratio), float(softness)


def _sharpness_and_noise(gray, lap, blurred):
    laplacian_var = float(lap.var())
    residual = gray.astype(np.float32) - blurred.astype(np.float32)
    hf_noise_energy = float(residual.var())
    return laplacian_var, hf_noise_energy


def _border_line_score(gray):
    h, w = gray.shape
    m = max(4, int(0.08 * min(h, w)))
    edges = cv2.Canny(gray, 60, 150)
    border = np.zeros_like(edges)
    border[:m, :] = edges[:m, :]
    border[-m:, :] = edges[-m:, :]
    border[:, :m] = edges[:, :m]
    border[:, -m:] = edges[:, -m:]
    lines = cv2.HoughLinesP(border, 1, np.pi / 180, threshold=60,
                             minLineLength=int(0.3 * min(h, w)), maxLineGap=10)
    if lines is None:
        return 0.0
    score = 0.0
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        length = np.hypot(x2 - x1, y2 - y1)
        angle = abs(np.arctan2(y2 - y1, x2 - x1))
        axis_aligned = min(angle, abs(angle - np.pi / 2), abs(angle - np.pi)) < 0.08
        if axis_aligned:
            score += length
    return float(score / (h + w))


def _block_artifact_score(gray):
    g = gray.astype(np.float32)
    h, w = g.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    g = g[:h8, :w8]
    dv = np.abs(np.diff(g, axis=0))
    dh = np.abs(np.diff(g, axis=1))
    boundary_rows = np.arange(7, h8 - 1, 8)
    boundary_cols = np.arange(7, w8 - 1, 8)
    on_boundary = dv[boundary_rows, :].mean() + dh[:, boundary_cols].mean()
    off_boundary = dv.mean() + dh.mean()
    return float(on_boundary / (off_boundary + 1e-6))


def _channel_xcorr_hf(bgr):
    small = cv2.resize(bgr, (192, 192), interpolation=cv2.INTER_AREA)
    b, g, r = cv2.split(small.astype(np.float32))

    def hf(c):
        blur = cv2.GaussianBlur(c, (0, 0), sigmaX=1.5)
        return c - blur

    rh, gh = hf(r).ravel(), hf(g).ravel()
    if rh.std() < 1e-6 or gh.std() < 1e-6:
        return 1.0
    corr = np.corrcoef(rh, gh)[0, 1]
    return float(corr)


def extract_features(bgr):
    """bgr: HxWx3 uint8 image (OpenCV BGR order). Returns dict of feature_name -> float."""
    bgr = _resize_longest(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5)

    fft_high_freq_ratio, fft_peak_ratio, fft_peak_count, radial_profile, radial_resid = _fft_features(gray)
    autocorr_peak_strength = _autocorr_feature(gray)
    sat_mean, sat_std, unique_color_ratio, channel_gain_r, channel_gain_b = _color_features(bgr)
    highlight_ratio, highlight_softness = _highlight_features(gray, lap)
    laplacian_var, hf_noise_energy = _sharpness_and_noise(gray, lap, blurred)
    border_line_score = _border_line_score(gray)
    block_artifact_score = _block_artifact_score(gray)
    channel_xcorr_hf = _channel_xcorr_hf(bgr)

    values = [
        fft_high_freq_ratio, fft_peak_ratio, fft_peak_count, autocorr_peak_strength,
        sat_mean, sat_std, unique_color_ratio, channel_gain_r, channel_gain_b,
        highlight_ratio, highlight_softness, laplacian_var, hf_noise_energy,
        border_line_score, block_artifact_score, channel_xcorr_hf,
    ] + list(radial_profile) + list(radial_resid)
    return dict(zip(FEATURE_NAMES, values))


# These are heavy-tailed (counts/variances/ratios spanning orders of magnitude);
# log1p compresses outliers so the linear/kernel classifier isn't dominated by them.
_LOG_SCALE_FEATURES = {"fft_peak_ratio", "fft_peak_count", "laplacian_var", "hf_noise_energy", "border_line_score"}


def feature_vector(bgr):
    feats = extract_features(bgr)
    vec = np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float64)
    for name in _LOG_SCALE_FEATURES:
        vec[FEATURE_NAMES.index(name)] = np.log1p(vec[FEATURE_NAMES.index(name)])
    return vec, feats
