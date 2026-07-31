"""
make_questions.py — turn labeled cloth images into ManipBench-style MCQs.

For each gradeable image, draws:
  - a light reference grid,
  - lettered candidate grasp points (your labeled corners + a center distractor),
and writes questions.json mapping each image to its candidates and correct
answer(s), for the VLM reasoning evaluation.

No GPU / no API needed. Run:
    python make_questions.py \
        --labels data/labels.json \
        --images data/images \
        --out_images data/annotated \
        --out_json data/questions.json
"""

import argparse, json, os
from PIL import Image, ImageDraw, ImageFont

# an MCQ needs enough options to be answerable; images with fewer identifiable
# candidate points are excluded from the MCQ eval (too occluded to pose a choice)
MIN_CANDIDATES = 3


FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",   # Linux / Colab
    "/Library/Fonts/Arial Bold.ttf",                          # macOS
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",      # macOS
    "C:/Windows/Fonts/arialbd.ttf",                           # Windows
    "arialbd.ttf",
    "DejaVuSans-Bold.ttf",
]

_font_reported = False


def load_font(px):
    """Load a scalable font at the requested pixel size.

    Tries real TrueType files first; if none load, uses PIL's default font
    WITH a size (Pillow >= 10.1), which unlike the bare default respects size.
    Prints once which font was chosen so silent tiny-text fallback is visible.
    """
    global _font_reported
    for p in FONT_PATHS:
        try:
            f = ImageFont.truetype(p, px)
            if not _font_reported:
                print(f"[font] using TrueType: {p} @ {px}px")
                _font_reported = True
            return f
        except Exception:
            continue
    try:
        f = ImageFont.load_default(size=px)          # scalable default (Pillow >=10.1)
        if not _font_reported:
            print(f"[font] no TrueType found; using sized default @ {px}px")
            _font_reported = True
        return f
    except TypeError:
        if not _font_reported:
            print("[font] WARNING: old Pillow, default font is fixed tiny size. "
                  "Run: pip install -U Pillow")
            _font_reported = True
        return ImageFont.load_default()


def build_candidates(entry):
    """Return (candidates dict {letter:[x,y]}, correct list[letter]).

    Candidates = all named corner points that are real locations, plus a
    'center' distractor (mean of the corner points -> the classic 'grab the
    middle' wrong answer). Duplicate coordinates are merged.
    """
    corners = entry["corners"]
    grasp = set(entry["grasp_points"])

    # collect unique points, remembering which names map to each coordinate
    pts = []                       # list of (coord_tuple, set_of_names)
    for name, xy in corners.items():
        xy = tuple(xy)
        hit = next((p for p in pts if p[0] == xy), None)
        if hit:
            hit[1].add(name)
        else:
            pts.append((xy, {name}))

    # center distractor from the corner cloud (skip if it lands on a corner)
    xs = [c[0][0] for c in pts]
    ys = [c[0][1] for c in pts]
    center = (round(sum(xs) / len(xs)), round(sum(ys) / len(ys)))
    if all(center != c[0] for c in pts) and len(pts) >= 3:
        pts.append((center, {"__center__"}))

    candidates, correct = {}, []
    for i, (xy, names) in enumerate(pts):
        letter = chr(ord("A") + i)
        candidates[letter] = list(xy)
        if names & grasp:                    # this point is a correct grasp
            correct.append(letter)
    return candidates, correct


def annotate(img, candidates, correct, grid=5):
    W, H = img.size
    d = ImageDraw.Draw(img, "RGBA")

    # reference grid (subtle)
    for i in range(1, grid):
        x = W * i // grid
        y = H * i // grid
        d.line([(x, 0), (x, H)], fill=(255, 255, 255, 90), width=max(2, W // 800))
        d.line([(0, y), (W, y)], fill=(255, 255, 255, 90), width=max(2, W // 800))

    r = max(30, W // 45)                       # original marker radius
    font = load_font(max(60, W // 34))         # letters ~half of previous
    for letter, (x, y) in candidates.items():
        # all candidates drawn identically so the image gives no answer away
        d.ellipse([x - r, y - r, x + r, y + r],
                  fill=(0, 0, 0, 170), outline=(255, 255, 255, 255),
                  width=max(3, W // 500))
        tb = d.textbbox((0, 0), letter, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        d.text((x - tw / 2, y - th / 2 - tb[1]), letter, fill=(255, 255, 255, 255), font=font)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/labels.json")
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--out_images", default="data/annotated")
    ap.add_argument("--out_json", default="data/questions.json")
    args = ap.parse_args()

    labels = json.load(open(args.labels))
    os.makedirs(args.out_images, exist_ok=True)
    questions = {}
    excluded = {}

    for fname, entry in labels.items():
        if not entry.get("gradeable", False):
            print(f"skip (not gradeable): {fname}")
            continue
        candidates, correct = build_candidates(entry)
        if not correct:
            print(f"skip (no correct candidate resolved): {fname}")
            continue
        if len(candidates) < MIN_CANDIDATES:
            reason = (f"only {len(candidates)} identifiable candidate(s); too "
                      f"occluded ({entry['cloth_state']}) to pose a multiple-choice question")
            excluded[fname] = {"cloth_state": entry["cloth_state"], "reason": reason}
            print(f"EXCLUDE {fname}: {reason}")
            continue

        src = os.path.join(args.images, fname)
        out_name = fname.replace(".jpg", "_annotated.jpg")
        out_path = os.path.join(args.out_images, out_name)
        if os.path.exists(src):
            img = Image.open(src).convert("RGB")
            annotate(img, candidates, correct).save(out_path, quality=90)
            drew = True
        else:
            print(f"  (image file not found, wrote question only): {src}")
            drew = False

        questions[fname] = {
            "instruction": entry["instruction"],
            "cloth_state": entry["cloth_state"],
            "candidates": candidates,
            "correct": correct,
            "annotated_image": out_path if drew else None,
        }
        print(f"{fname}: {len(candidates)} candidates, correct={correct}")

    out = {"questions": questions, "excluded": excluded}
    json.dump(out, open(args.out_json, "w"), indent=2)
    print(f"\nwrote {len(questions)} questions ({len(excluded)} excluded) -> {args.out_json}")
    if excluded:
        print("excluded:", ", ".join(excluded.keys()))


if __name__ == "__main__":
    main()