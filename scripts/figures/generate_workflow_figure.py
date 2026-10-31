#!/usr/bin/env python3
"""Draw the protocol figure as a transit-line diagram of a controlled experiment.

Optional. It needs the aggregate JSON from completed P0 and P1 runs, and matplotlib with the Liberation fonts.

Design grammar:
  * each arm is a line; PS is literally the S strand plus one added pose strand;
  * a capsule crossed by both lines is a step applied identically to both arms;
  * a hatched wall separates source cohorts from the held-out cohort, and hatching always means held out.

Reads only the two aggregate JSON files, so every printed count is derived rather than typed. Reads no
participant data and fits nothing. The walker glyph is schematic artwork, not a recorded pose.
Writes workflow_protocol.{pdf,svg} to --out-dir. The output is byte-deterministic.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42,  # embed TrueType, never Type 3
    "svg.fonttype": "none",
    "svg.hashsalt": "workflow-figure",
    "path.simplify": False,
})
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch

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

COHORTS = ["3DGait", "BMCLab", "PD-GaM", "T-SDU-PD"]
SHOWN_FOLD = "T-SDU-PD"  # the fold magnified in panel b
CANDIDATES = [c["name"] for c in P1["candidate_configs"]]
FAMILIES = [c["family"] for c in P1["candidate_configs"]]
GATE = P0["thresholds"]["C1_min"]
PEOPLE = P0["totals"]["subjects_eligible"]
WALKS = P0["totals"]["walks_eligible_012"]
assert list(P0["cohorts"]) == COHORTS and FAMILIES == ["LR", "LR", "RF", "RF", "HGB", "HGB"]
assert GATE == 0.05
PICK = {arm: CANDIDATES.index(P1["selected_configs"][arm][SHOWN_FOLD]["all"]) for arm in ("S", "PS")}

W, H = 7.0, 2.42  # layout units; visible height is H - YMIN
YMIN = 0.20
PRINT_WIDTH = 6.875  # the printed text width in inches, so type below is its true printed point size
SCALE = PRINT_WIDTH / W

INK = "#16191D"
SOFT = "#555D66"
RULE = "#6F767E"
HAIR = "#B9BEC4"
SOURCE = "#8E969F"
WORLD = "#F1F1EC"
ZOOM = "#ECEEF0"
SPEED = "#0F3D5E"
POSE = "#E0761A"
POSE_TEXT = "#B4560C"  # the pose accent, darkened until small type holds its contrast

SANS = "Liberation Sans"
SERIF = "Liberation Serif"
for family in (SANS, SERIF):
    font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False)

fig = plt.figure(figsize=(PRINT_WIDTH, (H - YMIN) * SCALE))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(YMIN, H)
ax.set_aspect("equal")
ax.axis("off")


def text(x, y, s, size=6.6, color=INK, ha="left", va="center", weight="normal", z=10):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, family=SANS, zorder=z)


def math(x, y, s, size=10.0, color=INK, ha="left", style="italic", z=10):
    """Arm symbols are set in serif italic."""
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va="center", family=SERIF, style=style, zorder=z)


def hatch_rect(x, y, w, h, gap=0.045, lw=0.55, color=INK, z=3, face="white", edge=True):
    box = Rectangle((x, y), w, h, facecolor=face, edgecolor=color if edge else "none", linewidth=0.7, zorder=z)
    ax.add_patch(box)
    t = -h
    while t < w:
        line, = ax.plot([x + t, x + t + h], [y, y + h], color=color, lw=lw, zorder=z + 0.1, solid_capstyle="butt")
        line.set_clip_path(box)
        t += gap


def line(points, color, lw=2.0, z=5):
    """A transit line. "L" points are straight; a pair of "C" points is a rounded corner."""
    codes = [MPath.MOVETO] + [MPath.LINETO if p[2] == "L" else MPath.CURVE3 for p in points[1:]]
    ax.add_patch(PathPatch(MPath([(p[0], p[1]) for p in points], codes), facecolor="none", edgecolor=color,
                           linewidth=lw, capstyle="butt", joinstyle="round", zorder=z))


def capsule(x0, x1):
    ax.add_patch(FancyBboxPatch((x0, CAP_B), x1 - x0, CAP_T - CAP_B, boxstyle="round,pad=0,rounding_size=0.08",
                                facecolor="white", edgecolor=RULE, linewidth=0.7, zorder=2))


def between(x, first, second):
    """Station wording sits in the gap between the two lines."""
    text(x, MID + 0.065, first, ha="center")
    text(x, MID - 0.06, second, ha="center")


def bead(x, y, color=None):
    if color:
        ax.add_patch(Circle((x, y), 0.05, facecolor=color, edgecolor="white", linewidth=0.6, zorder=8))
        ax.add_patch(Circle((x, y), 0.073, facecolor="white", edgecolor=color, linewidth=0.9, zorder=7))
    else:
        ax.add_patch(Circle((x, y), 0.034, facecolor="white", edgecolor=SOFT, linewidth=0.6, zorder=7))


def arrow(x0, y0, x1, y1, color=INK, lw=0.7, z=6, head=0.045):
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw, zorder=z, solid_capstyle="butt")
    dx, dy = x1 - x0, y1 - y0
    n = (dx * dx + dy * dy) ** 0.5
    ux, uy = dx / n, dy / n
    ax.add_patch(Polygon([(x1, y1), (x1 - ux * head - uy * head * 0.62, y1 - uy * head + ux * head * 0.62),
                          (x1 - ux * head + uy * head * 0.62, y1 - uy * head - ux * head * 0.62)],
                         closed=True, facecolor=color, edgecolor="none", zorder=z))


def heading(x, y, letter, title):
    text(x, y, letter, size=9.0, weight="bold")
    text(x + 0.135, y, title, size=8.2, weight="bold")


# Layout constants ------------------------------------------------------------------------------
TOP = H - 0.11          # headings
Y_S, Y_PS = 1.50, 0.90  # the two lines
MID = (Y_S + Y_PS) / 2
STRAND = 0.03           # half the separation of the two strands of PS
CAP_T, CAP_B = 1.74, 0.66
STAGE_L = 1.57
STAGE_T = TOP - 0.13
STAGE_B = YMIN + 0.02
WALL_L, WALL_R = 4.10, 4.24
WORLD_R = 5.42

# a  Rotation matrix: four cohorts, four folds ----------------------------------------------------
heading(0, TOP, "a", "Hold one cohort out")
GX, GW = 0.60, 0.205
RT, RH = 1.99, 0.365
for k in range(4):
    text(GX + GW * (k + 0.5), RT + 0.075, str(k + 1), color=SOFT, ha="center")
text(GX - 0.05, RT + 0.075, "fold", color=SOFT, ha="right")
for r, cohort in enumerate(COHORTS):
    y = RT - RH * (r + 1)
    text(0, y + 0.235, cohort, size=7.0, weight="bold" if cohort == SHOWN_FOLD else "normal")
    c = P0["cohorts"][cohort]
    text(0, y + 0.105, f"{c['subjects_eligible']} · {c['walks_eligible']}", size=6.5, color=SOFT)
    for k in range(4):
        tx, ty = GX + GW * k + 0.012, y + 0.035
        if k == r:
            hatch_rect(tx, ty, GW - 0.024, RH - 0.07, gap=0.04, lw=0.5)
        else:
            ax.add_patch(Rectangle((tx, ty), GW - 0.024, RH - 0.07, facecolor=SOURCE, edgecolor="none", zorder=3))
text(0, RT - RH * 4 - 0.10, f"people · walks (all: {PEOPLE} · {WALKS})", size=6.5, color=SOFT)
ax.add_patch(Rectangle((0, 0.285), 0.13, 0.085, facecolor=SOURCE, edgecolor="none", zorder=3))
text(0.165, 0.328, "source", size=6.5, color=SOFT)
hatch_rect(0.60, 0.285, 0.13, 0.085, gap=0.04, lw=0.5)
text(0.765, 0.328, "held out", size=6.5, color=SOFT)

# The magnified fold and its zoom into panel b
ZX0, ZX1 = GX + GW * 3, GX + GW * 4
ax.add_patch(Rectangle((ZX0 - 0.004, RT - RH * 4 + 0.012), ZX1 - ZX0 + 0.008, RH * 4 - 0.024, facecolor="none",
                       edgecolor=INK, linewidth=1.0, zorder=5))
ax.add_patch(Polygon([(ZX1 + 0.004, RT - 0.012), (STAGE_L, STAGE_T), (STAGE_L, STAGE_B),
                      (ZX1 + 0.004, RT - RH * 4 + 0.012)], closed=True, facecolor=ZOOM, edgecolor="none", zorder=1))
ax.plot([STAGE_L, STAGE_L], [STAGE_B, STAGE_T], color=HAIR, lw=0.6, zorder=2)

# b  One fold, source side ------------------------------------------------------------------------
heading(STAGE_L + 0.04, TOP, "b", "Inside one fold: same pipeline, one added input")
ax.add_patch(Rectangle((WALL_R, STAGE_B), WORLD_R - WALL_R, STAGE_T - STAGE_B, facecolor=WORLD, edgecolor="none", zorder=0))
text(WALL_L - 0.05, STAGE_T - 0.09, "SOURCE COHORTS ONLY", size=6.5, color=SOFT, ha="right", weight="bold")
text(WALL_R + 0.06, STAGE_T - 0.09, "HELD-OUT COHORT", size=6.5, ha="left", weight="bold")
hatch_rect(WALL_L, STAGE_B, WALL_R - WALL_L, STAGE_T - STAGE_B, gap=0.06, lw=0.9, z=4)

# Schematic walker: the pelvis is the root (S); the limbs carry the joint angles (P).
PELVIS = (1.93, 1.20)
NECK = (1.95, 1.56)
KNEE = (2.03, 0.97)
limbs = [[PELVIS, KNEE, (2.00, 0.74), (2.09, 0.73)],
         [PELVIS, (1.89, 0.95), (1.73, 0.79), (1.77, 0.73)],
         [NECK, (2.05, 1.41), (2.15, 1.31)],
         [NECK, (1.83, 1.41), (1.81, 1.27)],
         [PELVIS, NECK]]
for limb in limbs:
    ax.plot(*zip(*limb), color=POSE, lw=1.5, solid_capstyle="round", solid_joinstyle="round", zorder=6)
for limb in limbs[:4]:
    for joint in limb[1:-1]:
        ax.add_patch(Circle(joint, 0.024, facecolor="white", edgecolor=POSE, linewidth=0.8, zorder=7))
ax.add_patch(Circle((1.96, 1.645), 0.058, facecolor="white", edgecolor=POSE, linewidth=1.1, zorder=7))
ax.add_patch(Circle(PELVIS, 0.05, facecolor=SPEED, edgecolor="white", linewidth=0.6, zorder=8))
text(1.865, 1.135, "root path", size=6.5, color=SPEED, ha="right")
text(1.625, 0.605, "joint angles", size=6.5, color=POSE_TEXT)

# The speed strand leaves the pelvis and feeds both arms; the pose strand joins PS only.
BR = 2.36
line([(PELVIS[0], 1.20, "M"), (BR - 0.09, 1.20, "L"), (BR, 1.20, "C"), (BR, 1.29, "C"), (BR, Y_S - 0.09, "L"),
      (BR, Y_S, "C"), (BR + 0.09, Y_S, "C"), (WALL_L, Y_S, "L")], SPEED)
line([(BR - 0.09, 1.20, "M"), (BR, 1.20, "C"), (BR, 1.11, "C"), (BR, Y_PS + STRAND + 0.09, "L"),
      (BR, Y_PS + STRAND, "C"), (BR + 0.09, Y_PS + STRAND, "C"), (WALL_L, Y_PS + STRAND, "L")], SPEED)
line([(KNEE[0], KNEE[1], "M"), (KNEE[0] + 0.07, Y_PS - STRAND, "C"), (KNEE[0] + 0.24, Y_PS - STRAND, "C"),
      (WALL_L, Y_PS - STRAND, "L")], POSE)

# Arm names, the largest type on the lines
AX = 2.185
math(AX, Y_S + 0.235, "S", size=11, color=SPEED)
text(AX, Y_S + 0.09, "speed only", size=6.5, color=SPEED)
text(AX, Y_PS - 0.135, "pose + speed", size=6.5, color=INK)
math(AX, Y_PS - 0.29, "P", size=11, color=POSE_TEXT)
math(AX + 0.085, Y_PS - 0.29, "S", size=11, color=SPEED)

# The six candidates, strung on both lines
CX0, CX1 = 2.76, 3.58
capsule(CX0, CX1)
text((CX0 + CX1) / 2, CAP_T + 0.095, "Same 6 candidates", size=7.4, ha="center", weight="bold")
PIP = lambda i: CX0 + 0.105 + i * (CX1 - CX0 - 0.21) / 5
for fam, a, b in (("LR", 0, 1), ("RF", 2, 3), ("HGB", 4, 5)):
    text((PIP(a) + PIP(b)) / 2, CAP_T - 0.085, fam, size=6.5, color=SOFT, ha="center")
for i in range(6):
    bead(PIP(i), Y_S, SPEED if PICK["S"] == i else None)
    bead(PIP(i), Y_PS, POSE if PICK["PS"] == i else None)
between((CX0 + CX1) / 2, "best on inner", "LOCO of sources")

# P0 is one column of the grid; P1 is the whole grid.
ax.plot([PIP(0) - 0.05, PIP(0) + 0.05], [CAP_B - 0.075] * 2, color=INK, lw=1.5, solid_capstyle="butt", zorder=5)
text(PIP(0) + 0.085, CAP_B - 0.075, "P0: fixed logistic", size=6.5)
ax.plot([PIP(0) - 0.05, PIP(5) + 0.05], [CAP_B - 0.195] * 2, color=INK, lw=1.5, solid_capstyle="butt", zorder=5)
text(PIP(0) - 0.05, CAP_B - 0.285, "P1: one chosen per arm, fold", size=6.5)

# Refit
capsule(3.64, 4.03)
between(3.835, "Refit", "on all 3")

# Through the wall, one way
for y in (Y_S, Y_PS):
    ax.add_patch(Polygon([(WALL_L - 0.03, y + 0.08), (WALL_L + 0.10, y), (WALL_L - 0.03, y - 0.08)], closed=True,
                         facecolor=INK, edgecolor="white", linewidth=0.6, zorder=9))

# b  One fold, held-out side ----------------------------------------------------------------------
PX0, PX1, QX0, QX1 = 4.37, 4.81, 4.88, 5.32
capsule(PX0, PX1)
capsule(QX0, QX1)
between((PX0 + PX1) / 2, "Predict", "once")
between((QX0 + QX1) / 2, "Score", "macro-F1")
hatch_rect(PX0, 1.90, QX1 - PX0, 0.085, gap=0.04, lw=0.5)
arrow((PX0 + PX1) / 2, 1.90, (PX0 + PX1) / 2, CAP_T + 0.005)
arrow((QX0 + QX1) / 2, 1.90, (QX0 + QX1) / 2, CAP_T + 0.005)
text((PX0 + PX1) / 2 + 0.035, 1.82, "walks", size=6.5, color=SOFT)
text((QX0 + QX1) / 2 + 0.035, 1.82, "labels", size=6.5, color=SOFT)

# Nothing returns
BY = 0.555
ax.plot([4.66, WALL_R + 0.035], [BY, BY], color=INK, lw=0.8, ls=(0, (2.2, 1.6)), zorder=6)
ax.plot([WALL_R + 0.035] * 2, [BY - 0.07, BY + 0.07], color=INK, lw=1.5, zorder=6, solid_capstyle="butt")
text(WALL_R + 0.09, BY - 0.135, "no labels, features or", size=6.5)
text(WALL_R + 0.09, BY - 0.245, "scaling flow back", size=6.5)

# The lines run on, then meet at the contrast
MX = 5.58
line([(WALL_R, Y_S, "M"), (MX - 0.09, Y_S, "L"), (MX, Y_S, "C"), (MX, Y_S - 0.09, "C"), (MX, MID, "L")], SPEED)
line([(WALL_R, Y_PS + STRAND, "M"), (MX - 0.09 - STRAND, Y_PS + STRAND, "L"), (MX - STRAND, Y_PS + STRAND, "C"),
      (MX - STRAND, Y_PS + STRAND + 0.09, "C"), (MX - STRAND, MID, "L")], SPEED)
line([(WALL_R, Y_PS - STRAND, "M"), (MX - 0.09 + STRAND, Y_PS - STRAND, "L"), (MX + STRAND, Y_PS - STRAND, "C"),
      (MX + STRAND, Y_PS - STRAND + 0.09, "C"), (MX + STRAND, MID, "L")], POSE)
ax.add_patch(Circle((MX, MID), 0.062, facecolor=INK, edgecolor="white", linewidth=0.7, zorder=9))

# c  The contrast and the frozen gate -------------------------------------------------------------
CXL = 5.70
heading(WORLD_R + 0.1, TOP, "c", "Paired contrast")
math(CXL, MID, "Δ", size=14, style="normal")
math(CXL + 0.155, MID, "=", size=13, style="normal")
math(CXL + 0.315, MID, "P", size=14, color=POSE_TEXT)
math(CXL + 0.425, MID, "S", size=14, color=SPEED)
math(CXL + 0.575, MID, "−", size=13, style="normal")
math(CXL + 0.73, MID, "S", size=14, color=SPEED)
text(CXL, MID - 0.195, "same walks, 4-cohort mean", size=6.5, color=SOFT)
text(CXL, MID - 0.305, "paired person bootstrap", size=6.5, color=SOFT)

RX0, RX1, R0, RG = CXL, 6.98, 6.03, 6.42
RY = 0.45
text(RX0, RY + 0.265, "Frozen gate on Δ", size=7.4, weight="bold")
hatch_rect(RG, RY, RX1 - RG, 0.16, gap=0.05, lw=0.5, edge=False, color=POSE)
ax.plot([RX0, RX1], [RY, RY], color=INK, lw=0.8, zorder=6, solid_capstyle="butt")
for x, lab in ((R0, "0"), (RG, f"+{GATE:.2f}")):
    ax.plot([x, x], [RY - 0.035, RY + 0.035 if x == R0 else RY + 0.16], color=INK, lw=0.8, zorder=6)
    text(x, RY - 0.095, lab, size=6.5, ha="center")
ax.text((RG + RX1) / 2, RY + 0.08, "gain", fontsize=6.8, fontweight="bold", family=SANS, ha="center", va="center", zorder=11,
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none"))
text((RX0 + RG) / 2, RY + 0.08, "no gain", size=6.5, ha="center", color=SOFT)
text(RX1, RY - 0.205, "and lower 95% CI above 0", size=6.5, color=SOFT, ha="right")

OUTPUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT / "workflow_protocol.pdf", format="pdf", facecolor="white",
            metadata={"Creator": None, "Producer": None, "CreationDate": None, "ModDate": None})
fig.savefig(OUTPUT / "workflow_protocol.svg", format="svg", facecolor="white",
            metadata={"Creator": None, "Date": None, "Format": None, "Type": None})
plt.close(fig)
print(f"WORKFLOW_VECTOR_READY={OUTPUT / 'workflow_protocol.pdf'}")
