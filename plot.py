import argparse
import re
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_scores_from_files(folder_path):
    scores = []

    for file_path in folder_path.glob("*.txt"):
        pidx, _ = file_path.name.split("_")
        with file_path.open("r") as f:
            content = f.read()

            # Dataset: cifar10; Time limit: 10s
            dataset = re.search(r"Dataset: (.*?);", content).group(1)
            time_limit = re.search(r"Time limit: (\d+)s", content).group(1)

            score = re.search(r"Score: (\d+\.\d+)", content).group(1)
            score = float(score)
            if score > 0:
                scores.append((int(pidx), score))

    scores.sort()
    scores = [score for _, score in scores]

    if scores:
        sns.set_style("darkgrid")
        plt.figure(figsize=(12, 8))
        sns.lineplot(x=range(len(scores)), y=scores, marker="o", markersize=8, linewidth=2, color="#1f77b4")
        plt.xlabel("Program Index", fontsize=14)
        plt.ylabel("Accuracy", fontsize=14)
        plt.title(f"Automatic Program Generation for {dataset}. Accuracy after {time_limit}s training.", fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlim(-0.5, len(scores) - 0.5)
        plt.ylim(min(scores) - 0.05, max(scores) + 0.05)
        plt.tight_layout()

        plt.show()
    else:
        print("No score files found in the specified folder.")


def main():
    parser = argparse.ArgumentParser(description="Plot scores from files in a folder.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing the score files.")
    args = parser.parse_args()

    folder_path = Path(args.folder_path)

    if folder_path.is_dir():
        plot_scores_from_files(folder_path)
    else:
        print("Invalid folder path. Please provide a valid folder path.")


if __name__ == "__main__":
    main()
