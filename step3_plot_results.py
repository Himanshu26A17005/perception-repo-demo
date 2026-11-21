import matplotlib.pyplot as plt
import numpy as np

pipelines = ["Pure OpenCV\n(Canny + Template)", "Hybrid\n(OpenCV + HOG + SVM)", "Deep ML\n(Raw-Pixel MLP)"]
colors = ["#94a3b8", "#2563eb", "#16a34a"]

fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), dpi=150)

# 1. Overall Accuracy (270 Unseen-Font Stress Test)
ax = axes[0]
overall_acc = [11.48, 84.07, 61.85]
bars = ax.bar(pipelines, overall_acc, color=colors, width=0.52)
ax.set_ylim(0, 112)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Overall Stress-Test Accuracy (n=270)")
ax.grid(axis="y", linestyle="--", alpha=0.4)
for b, v in zip(bars, overall_acc):
    ax.text(b.get_x() + b.get_width() / 2, v + 2.5, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=9)

# 2. Accuracy by Lighting Condition
ax = axes[1]
lights = ["Glare\n(n=54)", "Normal\n(n=108)", "Shadow\n(n=54)", "Low Light\n(n=54)"]
x = np.arange(len(lights))
w = 0.25
ax.bar(x - w, [9.3, 13.9, 3.7, 16.7], width=w, label="Pure OpenCV", color=colors[0])
ax.bar(x,     [96.3, 88.9, 85.2, 61.1], width=w, label="Hybrid (HOG+SVM)", color=colors[1])
ax.bar(x + w, [74.1, 63.9, 61.1, 46.3], width=w, label="Deep ML", color=colors[2])
ax.set_xticks(x)
ax.set_xticklabels(lights, fontsize=8.5)
ax.set_ylim(0, 115)
ax.set_title("Accuracy by Lighting Condition")
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(loc="upper right", fontsize=8)

# 3. Accuracy by Viewing Angle
ax = axes[2]
angles = ["Frontal\n0°–20° (n=45)", "Moderate\n20°–40° (n=40)", "Extreme\n40°+ (n=185)"]
x = np.arange(len(angles))
ax.bar(x - w, [13.3, 0.0, 13.5], width=w, label="Pure OpenCV", color=colors[0])
ax.bar(x,     [62.2, 80.0, 90.3], width=w, label="Hybrid (HOG+SVM)", color=colors[1])
ax.bar(x + w, [51.1, 35.0, 70.3], width=w, label="Deep ML", color=colors[2])
ax.set_xticks(x)
ax.set_xticklabels(angles, fontsize=8.5)
ax.set_ylim(0, 115)
ax.set_title("Accuracy by Viewing Angle")
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("benchmark_270_results.png")
print("Saved benchmark_270_results.png successfully.")