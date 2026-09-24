#!/usr/bin/env python3
"""
space_fireflies.py -- Earth turning in deep space while satellites swarm around
it like fireflies, from Sputnik 1 (4 Oct 1957) to 2025.  Square 1:1 MP4.

Counts come from data/satellite_history_clean.csv (Our World in Data mirror of
the U.S. Space Force Space-Track catalogue: payloads + rocket bodies remaining in
orbit, per orbit regime, per year).  The number of glowing dots on screen equals
that count at every year boundary.  Individual orbits are simulated
(Kepler's laws, representative altitude/inclination for each regime) -- they
are illustrative, not real ephemerides.  Altitudes are log-compressed so the
dense low-orbit shell and the far geostationary ring fit in one frame.

Runs on CPU only.  Needs: numpy, opencv-python, pillow, and ffmpeg on PATH.

    python space_fireflies.py                          # full video -> output/
    python space_fireflies.py --size 720 --out preview.mp4
    python space_fireflies.py --stills 0,150,600,1250  # PNG test frames
"""
from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
MU = 398600.4418          # km^3/s^2
RE = 6371.0               # km
SIDEREAL_DAY = 86164.0905 # s

# ----------------------------------------------------------------- timeline
FPS = 30
INTRO_S, GROW1_S, GROW2_S, HOLD_S = 4.0, 24.0, 9.0, 5.0
T_SPUTNIK = 1957 + (277 - 1) / 365.25       # 4 Oct 1957
T_SPUTNIK_DECAY = 1958 + 3 / 365.25         # re-entered 4 Jan 1958
ORBIT_TIME_SCALE = 1250.0                   # 1 video second = 1250 real seconds
                                            # -> a LEO lap takes ~4.5 s on screen

CAPTIONS = [  # (calendar time, text)
    (T_SPUTNIK + 0.01, "4 Oct 1957  -  Sputnik 1, the first artificial satellite"),
    (1961.30, "1961  -  Yuri Gagarin, first human in orbit"),
    (1969.55, "1969  -  Apollo 11"),
    (1998.90, "1998  -  ISS assembly begins"),
    (2019.40, "2019  -  first Starlink batch: the mega-constellation era"),
    (2026.00, "Debris fragments are not included in this count"),
]


def total_frames() -> int:
    return int(round((INTRO_S + GROW1_S + GROW2_S + HOLD_S) * FPS))


def calendar_time(sec: float) -> float:
    """Video second -> calendar year (fractional)."""
    if sec < INTRO_S:
        return 1957.70 + (1958.0 - 1957.70) * sec / INTRO_S
    sec -= INTRO_S
    if sec < GROW1_S:
        return 1958.0 + (2015.0 - 1958.0) * sec / GROW1_S
    sec -= GROW1_S
    if sec < GROW2_S:
        return 2015.0 + (2026.0 - 2015.0) * sec / GROW2_S
    return 2026.0


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def camera_distance(cal: float) -> float:
    """Close on the lonely Earth first, pull back as MEO/GEO fill in."""
    d = 5.2 + (13.0 - 5.2) * float(smoothstep(1961.3, 1965.5, cal))
    return d - 0.9 * float(smoothstep(1990.0, 2026.0, cal))


# --------------------------------------------------------------------- data
REGIMES = ["LEO", "MEO", "GEO", "HEO"]


def load_counts(path: Path):
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    years = rows["year"].astype(float)
    counts = {r: rows[r].astype(float) for r in REGIMES}
    return years, counts, rows


def regime_count(years, counts, reg, cal):
    """Value for year Y is treated as the population at the END of year Y."""
    return np.interp(cal, years + 1.0, counts[reg], left=0.0, right=counts[reg][-1])


# ------------------------------------------------------------ orbit physics
def compress(r_re):
    """Display radius (Earth radii). Monotonic, ~identity in LEO."""
    return 1.0 + np.log1p((r_re - 1.0) / 1.6) * 1.55


def kepler_E(M, e, iters=7):
    E = M + e * np.sin(M)
    for _ in range(iters):
        E -= (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return E


class Swarm:
    """Fixed orbital elements for every object that will ever be drawn."""

    def __init__(self, n_per_regime, appear_cal, rng):
        a, e, inc, raan, argp, m0, reg = [], [], [], [], [], [], []
        for ri, name in enumerate(REGIMES):
            n = n_per_regime[name]
            yrs = appear_cal[name]
            if name == "LEO":
                alt, i, O = self._leo(n, yrs, rng)
                ecc = np.where(rng.random(n) < 0.12, rng.uniform(0.002, 0.03, n), 0.0)
                w = rng.uniform(0, 2 * np.pi, n)
            elif name == "MEO":
                alt, i, O = self._meo(n, yrs, rng)
                ecc = rng.uniform(0, 0.01, n)
                w = rng.uniform(0, 2 * np.pi, n)
            elif name == "GEO":
                alt = rng.normal(35786, 60, n)
                i = np.radians(np.where(yrs < 1985, rng.uniform(0, 12, n), rng.exponential(0.8, n)))
                O = rng.uniform(0, 2 * np.pi, n)
                ecc = rng.uniform(0, 0.002, n)
                w = rng.uniform(0, 2 * np.pi, n)
            else:  # HEO: Molniya/Tundra-type ellipses + supersync graveyard
                molniya = rng.random(n) < 0.55
                per = rng.uniform(500, 1500, n)
                apo = np.where(molniya, rng.uniform(38000, 40500, n), 0)
                circ = rng.uniform(36400, 48000, n)
                ra = np.where(molniya, apo + RE, circ + RE)
                rp = np.where(molniya, per + RE, circ + RE - rng.uniform(0, 300, n))
                sma = (ra + rp) / 2
                ecc = (ra - rp) / (ra + rp)
                alt = sma - RE
                i = np.radians(np.where(molniya, rng.normal(63.4, 1.0, n), rng.uniform(0, 15, n)))
                O = rng.uniform(0, 2 * np.pi, n)
                w = np.where(molniya, np.radians(270.0) + rng.normal(0, 0.05, n),
                             rng.uniform(0, 2 * np.pi, n))
            sma = alt + RE
            a.append(sma); e.append(ecc); inc.append(i); raan.append(O)
            argp.append(w); m0.append(rng.uniform(0, 2 * np.pi, n)); reg.append(np.full(n, ri))
        cat = lambda L: np.concatenate(L)
        self.a, self.e, self.i = cat(a), cat(e), cat(inc)
        self.raan, self.argp, self.m0 = cat(raan), cat(argp), cat(m0)
        self.regime = cat(reg)
        self.n = np.sqrt(MU / self.a ** 3)                      # Kepler's third law
        self._basis()

    @staticmethod
    def _leo(n, yrs, rng):
        alt = np.empty(n); inc = np.empty(n); raan = rng.uniform(0, 2 * np.pi, n)
        u = rng.random(n)
        early = yrs < 1990
        mid = (yrs >= 1990) & (yrs < 2019.4)
        late = yrs >= 2019.4
        # early space age: scattered low orbits, Soviet/US favourite inclinations
        k = np.where(early)[0]
        alt[k] = np.clip(rng.lognormal(np.log(750), 0.45, k.size), 250, 1990)
        inc[k] = rng.choice([28.5, 34, 48.4, 50.7, 65, 74, 81.2, 82.5, 98.5], k.size) \
            + rng.normal(0, 1.2, k.size)
        # 1990s-2010s: ISS, sun-synchronous imagers, Iridium, Globalstar
        k = np.where(mid)[0]
        pick = rng.choice(5, k.size, p=[0.10, 0.40, 0.12, 0.08, 0.30])
        alt[k] = np.select([pick == 0, pick == 1, pick == 2, pick == 3],
                           [rng.normal(415, 20, k.size), rng.normal(680, 90, k.size),
                            rng.normal(780, 8, k.size), rng.normal(1414, 10, k.size)],
                           rng.uniform(350, 1500, k.size))
        inc[k] = np.select([pick == 0, pick == 1, pick == 2, pick == 3],
                           [51.6, 98.0, 86.4, 52.0],
                           rng.choice([28.5, 45, 65, 74, 82, 97.5], k.size)) \
            + rng.normal(0, 0.6, k.size)
        # mega-constellation era: Starlink shells (Walker planes), OneWeb, SSO rideshares
        k = np.where(late)[0]
        pick = rng.choice(6, k.size, p=[0.38, 0.14, 0.10, 0.10, 0.10, 0.18])
        shell_alt = np.array([550, 540, 570, 560, 1200, 0])
        shell_inc = np.array([53.0, 53.2, 70.0, 97.6, 87.9, 0])
        shell_planes = np.array([72, 72, 36, 12, 18, 0])
        alt[k] = shell_alt[pick] + rng.normal(0, 3, k.size)
        inc[k] = shell_inc[pick] + rng.normal(0, 0.05, k.size)
        planes = shell_planes[pick]
        walker = planes > 0
        kw = k[walker]
        raan[kw] = (rng.integers(0, 1 << 30, kw.size) % planes[walker]) * 2 * np.pi / planes[walker] \
            + pick[walker] * 0.37
        ko = k[~walker]
        alt[ko] = rng.normal(530, 60, ko.size)
        inc[ko] = rng.choice([97.5, 45.0, 51.6], ko.size, p=[0.7, 0.15, 0.15])
        return alt, np.radians(inc), raan

    @staticmethod
    def _meo(n, yrs, rng):
        nav = (yrs > 1978) & (rng.random(n) < 0.75)
        pick = rng.choice(4, n, p=[0.40, 0.25, 0.18, 0.17])
        alt = np.where(nav, np.array([20180, 19130, 23222, 21528])[pick],
                       np.clip(rng.lognormal(np.log(6000), 0.6, n), 2000, 20000))
        inc = np.where(nav, np.array([55.0, 64.8, 56.0, 55.0])[pick], rng.uniform(0, 90, n))
        raan = np.where(nav, rng.integers(0, 6, n) * np.pi / 3 + pick * 0.5, rng.uniform(0, 2 * np.pi, n))
        return alt + rng.normal(0, 30, n), np.radians(inc), raan

    def _basis(self):
        cO, sO = np.cos(self.raan), np.sin(self.raan)
        cw, sw = np.cos(self.argp), np.sin(self.argp)
        ci, si = np.cos(self.i), np.sin(self.i)
        self.P = np.stack([cO * cw - sO * sw * ci, sO * cw + cO * sw * ci, sw * si], 1)
        self.Q = np.stack([-cO * sw - sO * cw * ci, -sO * sw + cO * cw * ci, cw * si], 1)

    def positions(self, t_sec):
        """Earth-centred inertial positions, display-compressed, Earth radii."""
        M = self.m0 + self.n * t_sec
        E = kepler_E(np.mod(M, 2 * np.pi), self.e)
        xp = self.a * (np.cos(E) - self.e)
        yp = self.a * np.sqrt(1 - self.e ** 2) * np.sin(E)
        r = (xp[:, None] * self.P + yp[:, None] * self.Q) / RE
        rr = np.linalg.norm(r, axis=1)
        return r * (compress(rr) / rr)[:, None]


# --------------------------------------------------------------- the scene
class Scene:
    def __init__(self, size, rng):
        self.W = self.H = size
        self.fov = math.radians(34.0)
        self.F = (size / 2) / math.tan(self.fov / 2)
        el, self.roll = math.radians(21.0), math.radians(-18.0)
        self.dir = np.array([math.cos(el), 0.0, math.sin(el)])
        f = -self.dir
        right = np.cross(f, [0, 0, 1.0]); right /= np.linalg.norm(right)
        up = np.cross(right, f)
        cr, sr = math.cos(self.roll), math.sin(self.roll)
        self.fw, self.rt, self.up = f, right * cr + up * sr, up * cr - right * sr
        sun = np.array([math.cos(math.radians(62)), math.sin(math.radians(62)), 0.28])
        self.sun = sun / np.linalg.norm(sun)
        self.theta0 = math.radians(-58.0)       # Baikonur region faces us at launch
        self._load_textures()
        self.stars = self._starfield(rng)

    def _load_textures(self):
        tw = 2048
        day = Image.open(ROOT / "assets/earth_texture.jpg").convert("RGB").resize((tw, tw // 2), Image.LANCZOS)
        self.day = (np.asarray(day, np.float32) / 255.0) ** 2.2
        night = Image.open(ROOT / "assets/night_lights.jpg").convert("L").resize((tw, tw // 2), Image.LANCZOS)
        n = (np.asarray(night, np.float32) / 255.0) ** 2.0
        self.night = n[..., None] * np.array([1.0, 0.72, 0.38], np.float32)

    def _starfield(self, rng):
        W, H = self.W, self.H
        img = np.zeros((H, W, 3), np.float32)
        n = int(1900 * (W / 1080) ** 2)
        x, y = rng.integers(0, W, n), rng.integers(0, H, n)
        b = 0.05 + 0.9 * rng.random(n) ** 9
        tint = np.where(rng.random(n)[:, None] < 0.5,
                        [0.75, 0.85, 1.0], [1.0, 0.9, 0.78]).astype(np.float32)
        np.add.at(img, (y, x), b[:, None] * tint)
        img = cv2.GaussianBlur(img, (0, 0), 0.55 * W / 1080) * 3.0
        return np.clip(img, 0, 0.8)

    def cam_pos(self, D):
        return self.dir * D

    def project(self, P, C):
        v = P - C
        z = v @ self.fw
        x = v @ self.rt
        y = v @ self.up
        return self.W / 2 + self.F * x / z, self.H / 2 - self.F * y / z, z

    def draw_earth(self, img, D, theta):
        W, H, F = self.W, self.H, self.F
        C = self.cam_pos(D)
        R_px = F * math.tan(math.asin(1.0 / D)) * 1.08 + 8
        cx = cy = W / 2
        x0, x1 = int(max(0, cx - R_px)), int(min(W, cx + R_px))
        y0, y1 = int(max(0, cy - R_px)), int(min(H, cy + R_px))
        xs, ys = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        d = (self.fw[None, None] + ((xs - cx) / F)[..., None] * self.rt
             - ((ys - cy) / F)[..., None] * self.up)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        b = d @ C
        h2 = C @ C - b * b                               # squared impact parameter
        hit = h2 < 1.0
        patch = img[y0:y1, x0:x1]

        # ---- atmosphere halo (outside the disc)
        h = np.sqrt(np.maximum(h2, 0))
        pc = C[None, None] - b[..., None] * d
        pc /= np.linalg.norm(pc, axis=-1, keepdims=True) + 1e-9
        lit = smoothstep(-0.35, 0.6, pc @ self.sun)
        halo = np.exp(-np.maximum(h - 1.0, 0) / 0.022) * (0.08 + 0.55 * lit)
        halo = np.where(hit, 0, halo)
        patch += halo[..., None] * np.array([0.25, 0.5, 1.0], np.float32)

        # ---- the globe
        dh, bh = d[hit], b[hit]
        t = -bh - np.sqrt(1.0 - h2[hit])
        nrm = C[None] + t[:, None] * dh
        lon = np.arctan2(nrm[:, 1], nrm[:, 0]) - theta
        lat = np.arcsin(np.clip(nrm[:, 2], -1, 1))
        th, tw = self.day.shape[:2]
        u = (((lon + np.pi) % (2 * np.pi)) / (2 * np.pi) * tw).astype(int) % tw
        v = np.clip(((np.pi / 2 - lat) / np.pi * th).astype(int), 0, th - 1)
        ndl = nrm @ self.sun
        day_w = smoothstep(-0.10, 0.22, ndl)
        lam = np.clip(ndl, 0, 1) ** 0.75
        col = self.day[v, u] * (0.015 + 0.95 * lam)[:, None]
        col += self.night[v, u] * (1.0 - day_w)[:, None] * 1.3
        mu = np.clip(-(dh * nrm).sum(1), 0, 1)
        rim = (1 - mu) ** 3.0 * (0.06 + 0.6 * day_w)
        col += rim[:, None] * np.array([0.25, 0.5, 1.0], np.float32)
        col *= 0.85
        patch[hit] = col

    def occluded(self, P, C):
        d = P - C
        t = -(d @ C) / np.einsum("ij,ij->i", d, d)
        cl = C[None] + t[:, None] * d
        return (t > 0) & (t < 1) & (np.einsum("ij,ij->i", cl, cl) < 1.0)


# -------------------------------------------------------- firefly shading
REGIME_COLOR = np.array([[1.00, 0.93, 0.42],   # LEO  yellow firefly
                         [0.72, 1.00, 0.42],   # MEO  green firefly
                         [1.00, 0.72, 0.28],   # GEO  amber
                         [1.00, 0.58, 0.32]],  # HEO  orange
                        np.float32)
REGIME_GAIN = np.array([0.55, 0.95, 1.10, 1.00], np.float32)
HALO_TINT = np.array([0.80, 1.00, 0.50], np.float32)


def splat(W, H, x, y, w_rgb):
    """Bilinear additive splat of N coloured points into an HxWx3 buffer."""
    x0, y0 = np.floor(x - 0.5).astype(int), np.floor(y - 0.5).astype(int)
    fx, fy = x - 0.5 - x0, y - 0.5 - y0
    out = np.zeros((H * W, 3), np.float32)
    for dx, dy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                       (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = x0 + dx, y0 + dy
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        idx = yi[ok] * W + xi[ok]
        for c in range(3):
            out[:, c] += np.bincount(idx, weights=w_rgb[ok, c] * wt[ok], minlength=H * W)
    return out.reshape(H, W, 3)


def bloom(S, W):
    k = W / 1080
    core = cv2.GaussianBlur(S, (0, 0), 0.65 * k) * 2.4
    halo = cv2.GaussianBlur(S, (0, 0), 2.4 * k) * 1.5
    small = cv2.resize(S, (W // 4, W // 4), interpolation=cv2.INTER_AREA)
    wide = cv2.resize(cv2.GaussianBlur(small, (0, 0), 3.5 * k), (W, W),
                      interpolation=cv2.INTER_LINEAR) * 0.9
    return core + (halo + wide) * HALO_TINT


# ---------------------------------------------------------------------- HUD
def font(size, light=False):
    for p in (["/usr/share/fonts/truetype/dejavu/DejaVuSans-ExtraLight.ttf"] if light else []) + [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf", "arial.ttf"]:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


class HUD:
    def __init__(self, W, frame_cal):
        s = W / 1080
        self.s = s
        self.f_year = font(int(84 * s), light=True)
        self.f_count = font(int(46 * s))
        self.f_label = font(int(19 * s))
        self.f_cap = font(int(25 * s))
        self.f_small = font(int(15 * s))
        # frame index at which each caption first becomes true
        self.caps = [(int(np.searchsorted(frame_cal, c)), txt) for c, txt in CAPTIONS]

    def draw(self, im: Image.Image, frame, cal, count, sput_xy=None):
        s, W = self.s, im.width
        d = ImageDraw.Draw(im)
        year = int(math.floor(cal - 1e-6))
        m = int(40 * s)
        d.text((m, m - 12 * s), f"{year}", font=self.f_year, fill=(235, 238, 225))
        d.text((m + 3 * s, m + 92 * s), f"{count:,}", font=self.f_count, fill=(255, 236, 140))
        d.text((m + 4 * s, m + 146 * s), "OBJECTS IN ORBIT", font=self.f_label, fill=(170, 175, 160))

        # progress line 1957 -> 2025
        y = W - int(78 * s)
        x0, x1 = m, W - m
        frac = np.clip((cal - 1957.7) / (2026.0 - 1957.7), 0, 1)
        d.line([(x0, y), (x1, y)], fill=(60, 64, 60), width=max(1, int(2 * s)))
        d.line([(x0, y), (x0 + (x1 - x0) * frac, y)], fill=(230, 220, 120), width=max(1, int(2 * s)))
        d.text((x0, y + 6 * s), "1957", font=self.f_small, fill=(120, 124, 118))
        d.text((x1 - 34 * s, y + 6 * s), "2025", font=self.f_small, fill=(120, 124, 118))

        # captions: 2.8 s each, fading in/out; the final one stays
        for k, (f0, txt) in enumerate(self.caps):
            last = k == len(self.caps) - 1
            age = (frame - f0) / FPS
            dur = 99 if last else 2.8
            if 0 <= age < dur:
                a = min(1.0, age / 0.4, (dur - age) / 0.5)
                tw = d.textlength(txt, font=self.f_cap)
                c = int(230 * a)
                d.text(((W - tw) / 2, y - 52 * s), txt, font=self.f_cap, fill=(c, c, int(c * 0.9)))

        cred = ("Counts: Our World in Data / U.S. Space Force Space-Track (payloads + rocket bodies)"
                "   |   orbits simulated, altitudes compressed")
        tw = d.textlength(cred, font=self.f_small)
        d.text(((W - tw) / 2, W - 30 * s), cred, font=self.f_small, fill=(95, 98, 92))

        if sput_xy is not None:
            (sx, sy), a = sput_xy
            c = int(220 * a)
            d.line([(sx + 8 * s, sy - 8 * s), (sx + 30 * s, sy - 30 * s)], fill=(c, c, int(c * .6)), width=1)
            d.text((sx + 34 * s, sy - 44 * s), "SPUTNIK 1", font=self.f_label, fill=(c, c, int(c * .6)))


# -------------------------------------------------------------------- main
def build(args):
    rng = np.random.default_rng(1957)
    W = args.size
    years, counts, _ = load_counts(ROOT / "data/satellite_history_clean.csv")
    nf = total_frames()
    frame_cal = np.array([calendar_time(f / FPS) for f in range(nf)])

    # per-frame, per-regime population straight from the data
    N = {r: regime_count(years, counts, r, frame_cal) for r in REGIMES}
    n_max = {r: int(counts[r].max()) for r in REGIMES}
    # object k of a regime is "born" the first time the running max count exceeds k
    appear_frame, appear_cal = {}, {}
    for r in REGIMES:
        run = np.maximum.accumulate(N[r])
        k = np.arange(n_max[r])
        fi = np.minimum(np.searchsorted(run, k + 1e-9, side="right"), nf - 1)
        appear_frame[r] = fi
        # calendar year each object appears (drives which orbit family it gets)
        appear_cal[r] = np.interp(k + 0.5, np.concatenate([[0], counts[r]]),
                                  np.concatenate([[1958.0], years + 1.0]))
    swarm = Swarm(n_max, appear_cal, rng)
    reg = swarm.regime
    kidx = np.concatenate([np.arange(n_max[r]) for r in REGIMES])
    born = np.concatenate([appear_frame[r] for r in REGIMES])
    freq = rng.uniform(0.22, 0.75, reg.size) * 2 * np.pi
    phase = rng.uniform(0, 2 * np.pi, reg.size)
    base_rgb = REGIME_COLOR[reg] * REGIME_GAIN[reg][:, None]

    scene = Scene(W, rng)
    hud = HUD(W, frame_cal)

    # Sputnik 1: 215 x 939 km, i = 65.1 deg, launched from Baikonur (45.9 N, 63.3 E)
    sp_f0 = int(np.searchsorted(frame_cal, T_SPUTNIK))
    sp_f1 = int(np.searchsorted(frame_cal, T_SPUTNIK_DECAY))
    sp = Swarm.__new__(Swarm)
    rp, ra = RE + 215, RE + 939
    sp.a, sp.e, sp.i = np.array([(rp + ra) / 2]), np.array([(ra - rp) / (ra + rp)]), np.radians([65.1])
    t_launch = sp_f0 / FPS * ORBIT_TIME_SCALE
    lat, lon = math.radians(45.9), math.radians(63.3)
    lon_eci = lon + scene.theta0 + 2 * math.pi * t_launch / SIDEREAL_DAY
    u = math.asin(math.sin(lat) / math.sin(sp.i[0]))
    sp.raan = np.array([lon_eci - math.atan2(math.cos(sp.i[0]) * math.sin(u), math.cos(u))])
    sp.argp = np.array([0.0])
    E = 2 * math.atan(math.sqrt((1 - sp.e[0]) / (1 + sp.e[0])) * math.tan(u / 2))
    sp.n = np.sqrt(MU / sp.a ** 3)
    sp.m0 = np.array([E - sp.e[0] * math.sin(E) - sp.n[0] * t_launch])
    sp._basis()

    end = nf if args.end is None else min(args.end, nf)
    frames = [int(s) for s in args.stills.split(",")] if args.stills else range(args.start, end)
    enc = None
    if not args.stills:
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        enc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{W}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
             "-preset", args.preset, "-crf", str(args.crf), "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)

    for f in frames:
        sec = f / FPS
        cal = frame_cal[f]
        D = camera_distance(cal)
        C = scene.cam_pos(D)
        t_orb = sec * ORBIT_TIME_SCALE
        theta = scene.theta0 + 2 * math.pi * t_orb / SIDEREAL_DAY

        img = scene.stars.copy()
        scene.draw_earth(img, D, theta)

        # which objects exist now, with a soft fade for the fractional one
        cur = np.concatenate([np.full(n_max[r], N[r][f]) for r in REGIMES])
        fade = np.clip(cur - kidx, 0, 1)
        live = fade > 0
        P = swarm.positions(t_orb)[live]
        age = (f - born[live]) / FPS
        flash = 1.0 + 3.0 * np.exp(-np.maximum(age, 0) / 0.25)
        pulse = (0.5 + 0.5 * np.sin(freq[live] * sec + phase[live])) ** 3
        w = fade[live] * flash * (0.30 + 0.70 * pulse)
        rgb = base_rgb[live] * w[:, None]
        tot = int(round(sum(N[r][f] for r in REGIMES)))

        sput_xy = None
        if sp_f0 <= f < sp_f1:
            Ps = sp.positions(t_orb)
            age_s = (f - sp_f0) / FPS
            ws = min(1.0, age_s / 0.3) * (1.0 + 3.0 * math.exp(-age_s / 0.4)) * 2.2
            P = np.vstack([P, Ps])
            rgb = np.vstack([rgb, np.array([[1.0, 0.95, 0.55]]) * ws])
            tot = max(tot, 1)

        vis = ~scene.occluded(P, C)
        x, y, z = scene.project(P[vis], C)
        depth = np.clip(D / np.maximum(z, 0.1), 0.6, 1.6) ** 1.2
        S = splat(W, W, x, y, rgb[vis] * depth[:, None])

        if sp_f0 <= f < sp_f1 and vis[-1]:
            a_lab = min(1.0, (f - sp_f0) / FPS / 0.6, max(0.0, (sp_f1 - f) / FPS / 0.6))
            sput_xy = ((x[-1], y[-1]), a_lab)

        light = bloom(S, W)
        img += 1.0 - np.exp(-1.35 * light)
        img = np.clip(img, 0, 1) ** (1 / 2.2)
        im = Image.fromarray((img * 255 + 0.5).astype(np.uint8))
        hud.draw(im, f, cal, tot, sput_xy)

        if enc:
            enc.stdin.write(im.tobytes())
            if f % 60 == 0:
                print(f"frame {f}/{nf}  year {int(cal - 1e-6)}  objects {tot:,}", flush=True)
        else:
            p = Path(args.stills_dir) / f"still_{f:04d}_{int(cal - 1e-6)}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            im.save(p)
            print("wrote", p, tot)

    if enc:
        enc.stdin.close()
        enc.wait()
        print("done ->", args.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=1080, help="square frame size in px")
    ap.add_argument("--out", default=str(ROOT / "output/space_fireflies_1957_2025_1x1.mp4"))
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--stills", default="", help="comma list of frame indices -> PNGs instead of video")
    ap.add_argument("--start", type=int, default=0, help="first frame (for rendering in chunks)")
    ap.add_argument("--end", type=int, default=None, help="stop before this frame")
    ap.add_argument("--stills-dir", default=str(ROOT / "output/stills"))
    build(ap.parse_args())
