"""
run_vlm.py — run a VLM over the cloth MCQs and score against ground truth.

Reads questions.json (produced by make_questions.py), asks the model to pick a
grasp letter for each annotated image, scores against the correct set, and
reports overall accuracy plus a breakdown by cloth_state (the difficulty
gradient). Writes results.json.

Setup:  pip install google-genai
Usage:  python run_vlm.py --questions data/questions.json \
                          --images data/annotated \
                          --model gemini-3.5-flash --seeds 1
"""

import argparse, json, os, re, time
from PIL import Image
from google import genai


def parse_letter(text, valid):
    """Extract a single candidate letter from a possibly-chatty response.

    Priority: (1) the whole reply IS one letter, (2) a standalone letter token
    (surrounded by non-letters, e.g. 'B.' or 'answer: C'), (3) give up. We do
    NOT scan for any letter inside words, since that matches the 'A' in 'grasp'.
    """
    if not text:
        return None
    up = text.strip().upper()

    # (1) clean single-letter reply, optionally with trailing punctuation
    m = re.fullmatch(r"([A-Z])[\.\)\:]?", up)
    if m and m.group(1) in valid:
        return m.group(1)

    # (2) a letter that stands alone as its own token (word boundaries on both
    #     sides so it isn't part of a word like GRASP or ANSWER)
    for tok in re.findall(r"(?<![A-Z])([A-Z])(?![A-Z])", up):
        if tok in valid:
            return tok

    return None


def ask(client, model, prompt, img, retries=4):
    for i in range(retries):
        try:
            r = client.models.generate_content(model=model, contents=[prompt, img])
            return r.text
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "503" in msg:
                wait = 5 * (i + 1)
                print(f"    rate/limit hit, waiting {wait}s...")
                time.sleep(wait)
                continue
            raise
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/questions.json")
    ap.add_argument("--images", default="data/annotated")
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--seeds", type=int, default=1, help="repeats per image")
    ap.add_argument("--out", default="data/results.json")
    args = ap.parse_args()

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or input("API key: "))
    data = json.load(open(args.questions))
    questions = data["questions"]

    results, correct_count, total = {}, 0, 0
    by_state = {}   # cloth_state -> [n_correct, n_total]

    for fname, q in questions.items():
        letters = list(q["candidates"].keys())
        img_path = q.get("annotated_image") or os.path.join(args.images,
                     fname.replace(".jpg", "_annotated.jpg"))
        img = Image.open(img_path)
        prompt = (
            "You are a robot manipulation assistant. The image shows a piece of "
            "cloth on a table with candidate grasp points labeled with letters.\n\n"
            f"Task: {q['instruction']}\n\n"
            "Which single labeled point should the robot grasp to perform this task? "
            "If the task needs two hands, name the first point you would grasp.\n"
            f"Answer with ONLY the letter (one of: {', '.join(letters)}). No explanation."
        )

        picks, hits = [], []
        for s in range(args.seeds):
            raw = ask(client, args.model, prompt, img)
            pick = parse_letter(raw, set(letters))
            is_correct = pick in q["correct"]
            picks.append(pick)
            hits.append(is_correct)
            time.sleep(1)   # gentle on the free tier

        # majority-correct across seeds (or single if seeds=1)
        img_correct = sum(hits) > len(hits) / 2
        correct_count += img_correct
        total += 1
        st = q["cloth_state"]
        by_state.setdefault(st, [0, 0])
        by_state[st][0] += img_correct
        by_state[st][1] += 1

        results[fname] = {
            "cloth_state": q["cloth_state"],
            "instruction": q["instruction"],
            "picks": picks,
            "correct_set": q["correct"],
            "is_correct": bool(img_correct),
        }
        print(f"{fname:22s} [{q['cloth_state']:18s}] picked {picks} "
              f"correct={q['correct']} -> {'OK' if img_correct else 'MISS'}")

    acc = correct_count / total if total else 0.0
    summary = {
        "model": args.model,
        "seeds": args.seeds,
        "n_questions": total,
        "accuracy": round(acc, 3),
        "by_cloth_state": {k: {"correct": v[0], "total": v[1]} for k, v in by_state.items()},
        "excluded": data.get("excluded", {}),
    }
    out = {"summary": summary, "per_image": results}
    json.dump(out, open(args.out, "w"), indent=2)

    print("\n--- SUMMARY ---")
    print(f"model: {args.model} | overall accuracy: {acc:.3f}  ({correct_count}/{total})")
    print("by cloth state:")
    for k, v in by_state.items():
        print(f"  {k:20s} {v[0]}/{v[1]}")
    if summary["excluded"]:
        print("excluded from MCQ:", ", ".join(summary["excluded"].keys()))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()