#!/usr/bin/env python3
"""Draw the two full-width results figures, one question per panel.

Optional. It needs the aggregate JSON from completed P0 and P1 runs, and matplotlib with the Liberation fonts.

  results_primary.pdf      A: the two frozen PS-S contrasts.   B: PS-S within each learner family.
  results_sensitivity.pdf  C: other inductive arms, and the transductive arm in its own band.
                           D: cohort predicted from the features, on its own 0-to-1 axis.

Panels A, B and C share one physical scale, so a length means the same effect anywhere. Exact intervals are drawn and
not printed beside the marks: only the point is labelled, to three decimals. Status never depends on colour:
  * frozen decision contrast: filled circle, heavy line;
  * exploratory inductive: open circle, thin line;
  * transductive: open diamond, tinted band and its own heading. Orange is reserved for this deployment caution.

Reads only the two aggregate JSON files, so every printed value and every drawn position comes from the same number.
Reads no participant data and fits nothing. Writes both figures as PDF and SVG to --out-dir. The output is
byte-deterministic.
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42,  # embed TrueType, never Type 3
    "svg.fonttype": "none",
    "svg.hashsalt": "results-figures",
    "path.simplify": False,
})
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Polygon, Rectangle

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--p0-json", required=True, type=Path, help="p0.json written by scripts/care_pd/run_p0.py")
    parser.add_argument("--p1-json", required=True, type=Path, help="p1_strong_models.json written by scripts/care_pd/run_p1.py")
    parser.add_argument("--out-dir", required=True, type=Path, help="folder for the PDF and SVG figures")
    return parser.parse_args()


ARGS = parse_args()
OUTPUT = ARGS.out_dir
P0 = json.loads(ARGS.p0_json.read_text())
P1 = json.loads(ARGS.p1_json.read_text())

POL = P1["per_policy_exploratory"]
X = P1["contrasts_exploratory_inductive"]
GATE = P0["thresholds"]["C1_min"]
SID = P1["site_identifiability"]["balanced_accuracy_5fold_subject_grouped"]
CHANCE = P1["site_identifiability"]["chance"]

# A contrast token is (text, subscript). A row is (descriptor, second line or None, contrast tokens, JSON contrast).
# Panels A and B each hold one contrast, PS - S, named once in the panel title.
PS_S = [("PS", None), ("−", None), ("S", None)]
PANEL_A = [
    ("P0 fixed logistic", "frozen before any score", None, P0["paired_deltas"]["PS_minus_S"]["M"]),
    ("P1 source-selected", "frozen after P0, same people", None, P1["primary"]["contrast"]["M"]),
]
PANEL_B = [
    ("Logistic (LR)", None, None, POL["LR"]["PS_minus_S"]["M"]),
    ("Forest (RF)", None, None, POL["RF"]["PS_minus_S"]["M"]),
    ("Boosting (HGB)", None, None, POL["HGB"]["PS_minus_S"]["M"]),
]
PANEL_C_INDUCTIVE = [
    ("Kinematics", None, [("KS", None), ("−", None), ("S", None)], X["KS_minus_S"]["M"]),
    ("Duration", None, [("S", "D"), ("−", None), ("S", None)], X["S_D_minus_S"]["M"]),
    ("Alt. speed", None, [("PS", "alt"), ("−", None), ("S", "alt")], X["PS_alt_minus_S_alt"]["M"]),
]
PANEL_C_TRANSDUCTIVE = [
    ("Scaled", None, [("P", "z"), ("S", "z"), ("−", None), ("S", "z")], P1["contrasts_transductive"]["PzSz_minus_Sz"]["M"]),
]
PANEL_D = [("S", "speed"), ("D", "duration"), ("K", "kinematics"), ("P", "pose")]
ALL_DELTA_ROWS = PANEL_A + PANEL_B + PANEL_C_INDUCTIVE + PANEL_C_TRANSDUCTIVE
assert len(ALL_DELTA_ROWS) == 9 and GATE == 0.05 and CHANCE == 0.25

# One shared effect scale, derived from the data and the margin.
XMIN = math.floor(min(r[3]["ci95"][0] for r in ALL_DELTA_ROWS) * 10) / 10
XMAX = math.ceil(max(GATE, *(r[3]["ci95"][1] for r in ALL_DELTA_ROWS)) * 10) / 10
TICKS = [round(XMIN + i / 10, 1) for i in range(int(round((XMAX - XMIN) * 10)) + 1)]

PRINT_WIDTH = 6.875  # the printed text width in inches, so type below is its true printed point size
GUTTER = 0.34
PANEL_W = (PRINT_WIDTH - GUTTER) / 2
LABEL_W = 1.26  # descriptor and contrast
VALUE_W = 0.40  # the point estimate, right of the plot
PLOT_W = PANEL_W - LABEL_W - VALUE_W

TITLE_H = 0.17
GUIDE_H = 0.15  # direction and margin labels
AXIS_H = 0.18
TITLE = 9.0
BODY = 8.0
SMALL = 7.5
MATH = 8.6
SUB = 7.0

INK = "#16191D"
SOFT = "#4A5159"
ZERO = "#8A9097"
HAIR = "#D5D8DC"
BAND = "#FBEFE3"
BLUE = "#0F3D5E"
SLATE = "#7F9BB0"  # panel D's bars, lighter than the effect marks because they answer a separate question
ORANGE = "#A84E08"
SANS = "Liberation Sans"
SERIF = "Liberation Serif"
for family in (SANS, SERIF):
    font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False)


class Canvas:
    """A figure drawn in printed inches, so sizes and positions are the ones on the page."""

    def __init__(self, height):
        self.h = height
        self.fig = plt.figure(figsize=(PRINT_WIDTH, height))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, PRINT_WIDTH)
        self.ax.set_ylim(0, height)
        self.ax.axis("off")
        self.renderer = self.fig.canvas.get_renderer()

    def text(self, x, y, s, size=BODY, color=INK, ha="left", va="center", weight="normal", family=SANS, style="normal", z=10):
        return self.ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, family=family, style=style, zorder=z)

    def width(self, artist):
        return artist.get_window_extent(self.renderer).width / self.fig.dpi

    def line(self, xs, ys, color=INK, lw=0.7, z=3, **kw):
        self.ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="butt", zorder=z, **kw)

    def title(self, x, y, letter, heading, tokens=None):
        a = self.text(x, y, letter, size=TITLE, weight="bold")
        x += self.width(a) + 0.07
        b = self.text(x, y, heading, size=TITLE, weight="bold")
        x += self.width(b) + 0.05
        if tokens:
            self.contrast(x, y, tokens)

    def contrast(self, x, y, tokens, right=None):
        """Set a contrast as math, from x or ending at right: serif italic symbols, upright minus, subscripts."""
        def lay(x0, draw):
            cur = x0
            for symbol, subscript in tokens:
                if symbol == "−":
                    cur += 0.03
                    a = self.text(cur, y - 0.002, "−", size=MATH, family=SERIF)
                    cur += self.width(a) + 0.03
                else:
                    a = self.text(cur, y, symbol, size=MATH, family=SERIF, style="italic")
                    cur += self.width(a)
                    if subscript:
                        b = self.text(cur, y - 0.03, subscript, size=SUB, family=SERIF, style="italic" if subscript != "alt" else "normal")
                        cur += self.width(b)
                if not draw:
                    a.remove()
                    if symbol != "−" and subscript:
                        b.remove()
            return cur - x0
        lay(x if right is None else right - lay(0, False), True)

    def save(self, stem):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(OUTPUT / f"{stem}.pdf", format="pdf", facecolor="white",
                         metadata={"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None})
        self.fig.savefig(OUTPUT / f"{stem}.svg", format="svg", facecolor="white",
                         metadata={"Creator": None, "Date": None, "Format": None, "Type": None})
        plt.close(self.fig)
        print(f"RESULTS_VECTOR_READY={OUTPUT / (stem + '.pdf')}")


def signed(v):
    return f"{v:+.3f}".replace("-", "−")


class Scale:
    """The shared effect axis, placed at one panel's plot area."""

    def __init__(self, panel_x):
        self.panel = panel_x
        self.l = panel_x + LABEL_W
        self.r = self.l + PLOT_W

    def x(self, v):
        return self.l + (v - XMIN) / (XMAX - XMIN) * (self.r - self.l)


def effect_frame(c, s, top, bottom):
    """Reference lines, direction and margin labels above the plot, ticks below. Zero is solid and the margin dashed."""
    c.line([s.x(0), s.x(0)], [bottom, top], color=ZERO, lw=0.8, z=2)
    c.line([s.x(GATE), s.x(GATE)], [bottom, top], color=INK, lw=0.7, z=2, dashes=(2.6, 1.8))
    c.line([s.l, s.r], [bottom, bottom], color=SOFT, lw=0.6, z=2)
    for v in TICKS:
        c.line([s.x(v), s.x(v)], [bottom, bottom - 0.035], color=INK, lw=0.6, z=2)
        lab = "0" if v == 0 else f"{v:+.1f}".replace("-", "−")
        c.text(s.x(v), bottom - 0.105, lab, size=SMALL, ha="center", color=SOFT)
    c.line([s.x(GATE), s.x(GATE)], [bottom, bottom - 0.035], color=INK, lw=0.6, z=2)
    c.text(s.panel, bottom - 0.105, "Δ macro-F1", size=SMALL, color=SOFT)
    y = top + GUIDE_H / 2
    c.text(s.x(0) - 0.05, y, "← speed alone better", size=SMALL, color=SOFT, ha="right")
    c.text(s.r + VALUE_W, y, f"+{GATE:.2f} margin", size=SMALL, color=INK, ha="right")  # stays inside the panel


def mark(c, s, y, d, kind):
    point, (lo, hi) = d["point"], d["ci95"]
    if kind == "frozen":
        c.line([s.x(lo), s.x(hi)], [y, y], color=BLUE, lw=1.7, z=6)
        c.ax.add_patch(Circle((s.x(point), y), 0.05, facecolor=BLUE, edgecolor="white", linewidth=0.6, zorder=7))
    elif kind == "transductive":
        c.line([s.x(lo), s.x(hi)], [y, y], color=ORANGE, lw=1.0, z=6)
        r = 0.052
        px = s.x(point)
        c.ax.add_patch(Polygon([(px, y + r), (px + r, y), (px, y - r), (px - r, y)], closed=True, facecolor=BAND, edgecolor=ORANGE, linewidth=1.0, zorder=7))
    else:
        c.line([s.x(lo), s.x(hi)], [y, y], color=INK, lw=1.0, z=6)
        c.ax.add_patch(Circle((s.x(point), y), 0.04, facecolor="white", edgecolor=INK, linewidth=1.0, zorder=7))
    c.text(s.r + VALUE_W, y, signed(point), ha="right", weight="bold" if kind == "frozen" else "normal")


def effect_row(c, s, y, row, kind):
    label, second, tokens, d = row
    if second:
        c.text(s.panel, y + 0.056, label, weight="bold")
        c.text(s.panel, y - 0.056, second, size=SMALL, color=SOFT)
    else:
        c.text(s.panel, y, label)
        if tokens:
            c.contrast(0, y, tokens, s.l - 0.09)
    mark(c, s, y, d, kind)


def primary():
    row = 0.17
    plot_h = 3 * row
    h = TITLE_H + GUIDE_H + plot_h + AXIS_H
    c = Canvas(h)
    top = h - TITLE_H - GUIDE_H
    bottom = top - plot_h
    a, b = Scale(0.0), Scale(PANEL_W + GUTTER)
    c.title(a.panel, h - TITLE_H / 2, "A", "Frozen contrasts:", PS_S)
    c.title(b.panel, h - TITLE_H / 2, "B", "Learner families, exploratory:", PS_S)
    effect_frame(c, a, top, bottom)
    effect_frame(c, b, top, bottom)
    # Panel A's two rows take the same height as B's three.
    step = plot_h / len(PANEL_A)
    for i, r in enumerate(PANEL_A):
        effect_row(c, a, top - step * (i + 0.5), r, "frozen")
    for i, r in enumerate(PANEL_B):
        effect_row(c, b, top - row * (i + 0.5), r, "explore")
    c.save("results_primary")


def sensitivity():
    row = 0.17
    head = 0.15
    gap = 0.04
    plot_h = 3 * row + gap + head + row
    h = TITLE_H + GUIDE_H + plot_h + AXIS_H
    c = Canvas(h)
    top = h - TITLE_H - GUIDE_H
    bottom = top - plot_h
    s = Scale(0.0)
    c.title(s.panel, h - TITLE_H / 2, "C", "Sensitivity, exploratory, P1 selected")
    effect_frame(c, s, top, bottom)
    y = top
    c.text(s.panel, top + GUIDE_H / 2, "Inductive", size=SMALL, weight="bold", color=SOFT)
    for r in PANEL_C_INDUCTIVE:
        y -= row
        effect_row(c, s, y + row / 2, r, "explore")
    y -= gap
    c.ax.add_patch(Rectangle((s.panel, bottom), PANEL_W, y - bottom, facecolor=BAND, edgecolor="none", zorder=0.5))
    y -= head
    c.text(s.panel + 0.04, y + head / 2, "Transductive: unlabelled target walks used", size=SMALL, weight="bold", color=ORANGE).set_bbox(dict(boxstyle="square,pad=0.08", facecolor=BAND, edgecolor="none"))
    for r in PANEL_C_TRANSDUCTIVE:
        y -= row
        label, second, tokens, d = r
        c.text(s.panel + 0.04, y + row / 2, label)
        c.contrast(0, y + row / 2, tokens, s.l - 0.09)
        mark(c, s, y + row / 2, d, "transductive")

    # Panel D: another quantity, so another axis, 0 to 1. Point estimates; the JSON carries no interval, so none is drawn.
    px = PANEL_W + GUTTER
    d_label = 0.92
    d_value = 0.40
    l, r_end = px + d_label, px + PANEL_W - d_value
    x_of = lambda v: l + v * (r_end - l)
    c.title(px, h - TITLE_H / 2, "D", "Cohort predicted from features, no severity label")
    c.line([x_of(CHANCE), x_of(CHANCE)], [bottom, top], color=INK, lw=0.7, z=6, dashes=(2.6, 1.8))
    c.text(x_of(CHANCE), top + GUIDE_H / 2, "four-cohort chance", size=SMALL, color=INK, ha="center")
    c.text(r_end + d_value, top + GUIDE_H / 2, "person-grouped 5-fold CV", size=SMALL, color=SOFT, ha="right")
    c.line([l, r_end], [bottom, bottom], color=SOFT, lw=0.6, z=2)
    c.line([l, l], [bottom, top], color=SOFT, lw=0.6, z=2)
    for v in (0.0, CHANCE, 0.5, 0.75, 1.0):
        c.line([x_of(v), x_of(v)], [bottom, bottom - 0.035], color=INK, lw=0.6, z=2)
        c.text(x_of(v), bottom - 0.105, f"{v:.2f}" if v != 0 else "0", size=SMALL, ha="center", color=SOFT)
    c.text(px, bottom - 0.105, "Balanced acc.", size=SMALL, color=SOFT)
    d_row = plot_h / len(PANEL_D)
    for i, (key, word) in enumerate(PANEL_D):
        y = top - d_row * (i + 0.5)
        a = c.text(px, y, key, size=MATH, family=SERIF, style="italic")
        c.text(px + c.width(a) + 0.05, y, word)
        c.ax.add_patch(Rectangle((l, y - 0.04), x_of(SID[key]) - l, 0.08, facecolor=SLATE, edgecolor="none", zorder=5))
        c.text(r_end + d_value, y, f"{SID[key]:.3f}", ha="right")
    c.save("results_sensitivity")


primary()
sensitivity()
