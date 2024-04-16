import json
import sys

file_to_inspect = sys.argv[1]

results_jsonl = []
with open(file_to_inspect, "r") as f:
    results_jsonl.extend(json.loads(line) for line in f)

# %%
score = 0
for result in results_jsonl:
    if result["passed"] is True:
        score += 1
print(f"Score: {score}/{len(results_jsonl)}={score/len(results_jsonl):.3f}")
