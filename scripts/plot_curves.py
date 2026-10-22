#!/usr/bin/env python3
"""Plot the REAL training curves from the actual run logs."""
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT = "/home/z/my-project/nexus-500m"

steps, losses = [], []
with open(f"{ROOT}/training/runs/poc/train_log.csv") as f:
    next(f)
    for line in f:
        parts = line.strip().split(",")
        if parts[0].isdigit():
            steps.append(int(parts[0]))
            losses.append(float(parts[1]))

vsteps, vlosses = [], []
with open(f"{ROOT}/training/runs/poc/val_log.csv") as f:
    for line in f:
        if line.strip():
            a, b = line.strip().split(",")
            vsteps.append(int(a))
            vlosses.append(float(b))

# instruct phase (starts at step 3500)
isteps, ilosses = [], []
with open(f"{ROOT}/training/runs/instruct/train_log.csv") as f:
    next(f)
    for line in f:
        parts = line.strip().split(",")
        if parts[0].isdigit():
            isteps.append(int(parts[0]))
            ilosses.append(float(parts[1]))
ivsteps, ivlosses = [], []
with open(f"{ROOT}/training/runs/instruct/val_log.csv") as f:
    for line in f:
        if line.strip():
            a, b = line.strip().split(",")
            ivsteps.append(int(a))
            ivlosses.append(float(b))

fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
ax.plot(steps, losses, lw=1.0, alpha=0.5, color="#3b82f6", label="pretrain train loss")
ax.plot(vsteps, vlosses, "o-", lw=1.6, color="#1d4ed8", ms=4, label="pretrain val loss")
ax.plot(isteps, ilosses, lw=1.0, alpha=0.6, color="#f97316", label="instruct-tune train loss")
ax.plot(ivsteps, ivlosses, "s-", lw=1.6, color="#c2410c", ms=4, label="instruct-tune val loss")
ax.set_xlabel("training step (batch 8 × ctx 256 = 2048 tokens/step)")
ax.set_ylabel("cross-entropy loss (nats)")
ax.set_title("NEXUS-PoC (12.57M params) - real training run, CPU only (2 cores)")
ax.axvline(3500, color="#94a3b8", ls="--", lw=1)
ax.text(3500, ax.get_ylim()[0] + 0.2, " instruction tuning begins", fontsize=8, color="#64748b")
ax.grid(alpha=0.25)
ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=9)
fig.savefig(f"{ROOT}/docs/training_curves.png", dpi=150)
print("saved docs/training_curves.png")
print(f"steps plotted: pretrain={len(steps)} val pts={len(vsteps)} instruct={len(isteps)}")
