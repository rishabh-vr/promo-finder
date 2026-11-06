"""Quality experiment on saved outputs:  blind pairwise LLM-vs-rules, by a strong model.

  python evaluate.py --data <show folder> --out out/        (expects out/promos.json and
                                                             out/baseline_judged/promos.json)
"""
import argparse
import json
import os

from promo.evaluate import pairwise, write
from promo.llm import LLM
from promo.parse import load_show


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", help="comparison model chain (default: the audit chain)")
    a = ap.parse_args()
    eps = {e.ep: e for e in load_show(a.data)}
    llm_rows = json.load(open(os.path.join(a.out, "promos.json"), encoding="utf-8"))
    base_rows = json.load(open(os.path.join(a.out, "baseline_judged", "promos.json"), encoding="utf-8"))
    judge = LLM(role="audit", models=[m.strip() for m in a.models.split(",")] if a.models else None)
    res = pairwise(llm_rows, base_rows, eps, judge)
    write(a.out, res)
    print(json.dumps({k: v for k, v in res.items() if k != "pairs"}, indent=1))
    print("usage:", json.dumps(judge.stats))


if __name__ == "__main__":
    main()
