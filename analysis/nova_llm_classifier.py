"""
nova_llm_classifier.py  —  UPF-NeuroCog v2.0
=============================================
LLM-powered NOVA 1–4 food classifier.

Supports three backends (choose via --backend):
  openai   : GPT-4o via OpenAI API (set OPENAI_API_KEY)
  hf       : Mistral-7B-Instruct via HuggingFace transformers
  rule     : Deterministic keyword fallback (no API needed)

Outputs a CSV with columns [food_description, nova_group, confidence, backend]

Usage
-----
  python nova_llm_classifier.py --input data/nhanes_food_items.csv \
                                 --output data/nova_classified.csv  \
                                 --backend openai

Real dataset URLs embedded in nhanes_pipeline.py.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd

# ─────────────────────────────────────────────────────────────
# NOVA classification reference (Monteiro et al., 2019)
# ─────────────────────────────────────────────────────────────
NOVA_DEFINITIONS = {
    1: "Unprocessed or minimally processed foods: fresh/frozen/dried fruits, "
       "vegetables, plain meats, eggs, milk, plain grains, legumes, nuts, seeds.",
    2: "Processed culinary ingredients: oils, butter, lard, sugar, salt, flour, "
       "starches, vinegar, honey — used in cooking but not eaten alone.",
    3: "Processed foods: canned/bottled vegetables & legumes, cured/smoked meats, "
       "salted nuts, cheese, freshly baked bread, beer, wine, cider.",
    4: "Ultra-processed foods (UPF): carbonated drinks, packaged snacks, "
       "reconstituted meat products, instant noodles, mass-produced bread/pastries, "
       "breakfast cereals with sugar, flavoured yoghurts, margarine, energy drinks, "
       "ready meals, fast-food items.",
}

FEW_SHOT_EXAMPLES = [
    ("Fresh apple",                       1),
    ("Olive oil",                         2),
    ("White sugar",                       2),
    ("Canned black beans in water",       3),
    ("Smoked salmon",                     3),
    ("Cheddar cheese",                    3),
    ("Coca-Cola regular",                 4),
    ("Lay's potato chips",                4),
    ("Oreo cookies",                      4),
    ("Instant ramen noodles",             4),
    ("Plain whole milk",                  1),
    ("Frozen mixed vegetables no sauce",  1),
    ("Wheat flour",                       2),
    ("Sourdough bread artisan bakery",    3),
    ("Nature Valley granola bar",         4),
]

SYSTEM_PROMPT = """You are a precise nutritional scientist specialising in the NOVA food classification system.
Classify each food item into exactly one NOVA group (1, 2, 3, or 4).

NOVA groups:
{defs}

Return ONLY a JSON object: {{"nova_group": <int 1-4>, "confidence": <float 0-1>, "rationale": "<one sentence>"}}
No other text.""".format(defs="\n".join(f"  Group {k}: {v}" for k, v in NOVA_DEFINITIONS.items()))

# ─────────────────────────────────────────────────────────────
# Rule-based fallback (no API)
# ─────────────────────────────────────────────────────────────
_NOVA4_KEYWORDS = re.compile(
    r"chip|crisp|soda|cola|pepsi|sprite|fanta|energy drink|red bull|"
    r"oreo|dorito|cheeto|pringle|instant|ready.meal|fast.food|nugget|"
    r"hot dog|frankfurter|margarine|flavou?r(ed)?|artificial|"
    r"breakfast cereal|granola bar|protein bar|candy|gummy|lollipop|"
    r"ice cream|gelato|frozen yogurt|milkshake|whipped cream spray|"
    r"packaged bread|wonder bread|bimbo|wonder|hostess|"
    r"ramen|cup.noodle|flavored|sweetened|diet soda|diet coke",
    re.I,
)
_NOVA3_KEYWORDS = re.compile(
    r"canned|smoked|cured|salted nut|cheese|beer|wine|cider|"
    r"artisan bread|sourdough|preserved|pickled|fermented|"
    r"deli meat|ham|bacon|sausage|salami|pepperoni",
    re.I,
)
_NOVA2_KEYWORDS = re.compile(
    r"\boil\b|butter|lard|sugar|salt|flour|starch|vinegar|"
    r"honey|syrup|cream|condensed milk",
    re.I,
)


def rule_based_classify(food: str) -> dict:
    """Deterministic keyword-based NOVA classifier (no API)."""
    f = food.lower()
    if _NOVA4_KEYWORDS.search(f):
        return {"nova_group": 4, "confidence": 0.80, "rationale": "Keyword match: UPF marker detected."}
    if _NOVA3_KEYWORDS.search(f):
        return {"nova_group": 3, "confidence": 0.75, "rationale": "Keyword match: processed food marker."}
    if _NOVA2_KEYWORDS.search(f):
        return {"nova_group": 2, "confidence": 0.78, "rationale": "Keyword match: culinary ingredient."}
    return {"nova_group": 1, "confidence": 0.70, "rationale": "No processing marker found; assumed minimally processed."}


# ─────────────────────────────────────────────────────────────
# OpenAI backend
# ─────────────────────────────────────────────────────────────
def openai_classify(food: str, client) -> dict:
    """Classify a single food item using OpenAI GPT-4o."""
    examples_text = "\n".join(
        f'  Food: "{f}" → {{"nova_group": {g}, "confidence": 0.95, "rationale": "example"}}'
        for f, g in FEW_SHOT_EXAMPLES[:6]
    )
    user_msg = f"Examples:\n{examples_text}\n\nNow classify:\nFood: \"{food}\""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0,
        max_tokens=120,
    )
    raw = resp.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', raw, re.S)
        return json.loads(m.group()) if m else rule_based_classify(food)


# ─────────────────────────────────────────────────────────────
# HuggingFace backend (Mistral-7B-Instruct)
# ─────────────────────────────────────────────────────────────
def hf_classify(food: str, pipeline_fn) -> dict:
    """Classify a single food item using local HuggingFace model."""
    prompt = (
        f"[INST] {SYSTEM_PROMPT}\n\n"
        f"Food: \"{food}\" [/INST]"
    )
    out = pipeline_fn(prompt, max_new_tokens=120, do_sample=False)[0]["generated_text"]
    generated = out[len(prompt):].strip()
    try:
        m = re.search(r'\{.*\}', generated, re.S)
        return json.loads(m.group()) if m else rule_based_classify(food)
    except Exception:
        return rule_based_classify(food)


# ─────────────────────────────────────────────────────────────
# Batch classification
# ─────────────────────────────────────────────────────────────
def classify_batch(
    foods: list[str],
    backend: str = "rule",
    batch_size: int = 50,
    sleep_between: float = 0.5,
) -> pd.DataFrame:
    """
    Classify a list of food descriptions.

    Parameters
    ----------
    foods        : list of food name strings
    backend      : 'openai' | 'hf' | 'rule'
    batch_size   : API calls per chunk (openai/hf only)
    sleep_between: seconds between batches to avoid rate limits

    Returns
    -------
    DataFrame with columns [food_description, nova_group, confidence, rationale, backend]
    """
    classify_fn = None

    if backend == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        classify_fn = lambda f: openai_classify(f, client)

    elif backend == "hf":
        from transformers import pipeline as hf_pipeline
        pipe = hf_pipeline(
            "text-generation",
            model="mistralai/Mistral-7B-Instruct-v0.2",
            device_map="auto",
        )
        classify_fn = lambda f: hf_classify(f, pipe)

    else:  # rule-based fallback
        classify_fn = rule_based_classify

    results = []
    for i, food in enumerate(foods):
        if not isinstance(food, str) or not food.strip():
            results.append({"nova_group": None, "confidence": None, "rationale": "empty"})
            continue
        try:
            res = classify_fn(food.strip())
        except Exception as e:
            res = {"nova_group": None, "confidence": None, "rationale": f"error: {e}"}
        res["food_description"] = food
        res["backend"] = backend
        results.append(res)

        if backend in ("openai", "hf") and (i + 1) % batch_size == 0:
            time.sleep(sleep_between)

    df = pd.DataFrame(results)
    return df[["food_description", "nova_group", "confidence", "rationale", "backend"]]


# ─────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────
def evaluate_classifier(pred_df: pd.DataFrame, ground_truth_csv: str) -> dict:
    """
    Compare classifier output against a manual gold-standard CSV.

    Expected gold-standard CSV columns: food_description, nova_group
    """
    gt = pd.read_csv(ground_truth_csv).rename(columns={"nova_group": "nova_true"})
    merged = pred_df.merge(gt, on="food_description", how="inner")
    acc = (merged["nova_group"] == merged["nova_true"]).mean()
    upa_acc = (
        (merged["nova_group"].isin([3, 4]) & merged["nova_true"].isin([3, 4]))
        | (merged["nova_group"].isin([1, 2]) & merged["nova_true"].isin([1, 2]))
    ).mean()
    print(f"Overall accuracy : {acc:.3f}")
    print(f"UPF/non-UPF acc  : {upa_acc:.3f}")
    print(merged.groupby(["nova_true", "nova_group"]).size().unstack(fill_value=0))
    return {"accuracy": acc, "upf_accuracy": upa_acc}


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="NOVA LLM Food Classifier — UPF-NeuroCog v2.0")
    parser.add_argument("--input",   required=True,  help="CSV with a 'food_description' column")
    parser.add_argument("--output",  required=True,  help="Output CSV path")
    parser.add_argument("--backend", default="rule", choices=["openai", "hf", "rule"])
    parser.add_argument("--eval",    default=None,   help="Optional gold-standard CSV for evaluation")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    assert "food_description" in df.columns, "Input CSV must have a 'food_description' column"

    print(f"[nova_llm_classifier] Classifying {len(df):,} items via '{args.backend}' backend…")
    results = classify_batch(df["food_description"].tolist(), backend=args.backend)

    results.to_csv(args.output, index=False)
    print(f"[nova_llm_classifier] Saved → {args.output}")
    print(results["nova_group"].value_counts().sort_index().rename("count").to_string())

    if args.eval:
        evaluate_classifier(results, args.eval)


if __name__ == "__main__":
    main()
