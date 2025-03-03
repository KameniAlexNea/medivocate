import json
import os
from collections import defaultdict
from glob import glob


def compute_evaluation(folder="data/evaluation_results"):
    files = glob(os.path.join(folder, "*.json"))
    results = defaultdict(int)
    for file in files:
        raw: str = json.load(open(file))["evaluation"]
        try:
            result = json.loads(raw.replace("```", "").replace("json", "").strip())
            results[result["evaluation"]] += 1
        except:
            pass
    return results


results = compute_evaluation()
print(results)
total = sum(results.values())
for k, v in results.items():
    print(k, round(v * 100 / total, 1))
