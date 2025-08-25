import os
import sys
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    output_dir = "./dtw_metrics"
    all_json = [os.path.join(output_dir, x) for x in os.listdir(output_dir)]

    score_predicted = []
    score_predicted_1 = []
    score_predicted_3 = []
    for filename in all_json:
        with open(filename, 'r') as jfile:
            data = json.load(jfile)
        for pathway in data:
            score_predicted.append(pathway['score'])
            score_predicted_1.append(pathway['score_sp1'])
            score_predicted_3.append(pathway['score_sp3'])

    df = pd.DataFrame({
        "value": score_predicted + score_predicted_1 + score_predicted_3,
        "group": (
            ["Predicted Path"] * len(score_predicted) +
            ["Predicted Path + 1 Spike In Node"] * len(score_predicted_1) +
            ["Predicted Path + 3 Spike In Nodes"] * len(score_predicted_3)
        )
    })

    # Create directory if it doesn't exist
    os.makedirs("su_figures", exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.kdeplot(data=df, x="value", hue="group", common_norm=False, fill=True)
    plt.xlabel("")
    plt.ylabel("Distance From Ground Truth Pathway")
    plt.tight_layout()
    plt.savefig("su_figures/dtw_distributions.png", dpi=300)

if __name__ == "__main__":
    main()
