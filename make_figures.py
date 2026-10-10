"""
make_figures.py: draws the chapter's figures from the files saved by world_model.py.
Run after world_model.py:  python make_figures.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import world_model as wm

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "axes.titlesize": 9.5, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6})

log = json.load(open("runs/log.json"))
cfg = wm.Config(**log["config"])
vae = wm.ConvVAE(cfg.z_dim).to(wm.DEVICE)
vae.load_state_dict(torch.load("runs/vae.pt", map_location=wm.DEVICE))
rnn = wm.MDNRNN(cfg.z_dim, 1, cfg.hidden, cfg.n_mix).to(wm.DEVICE)
rnn.load_state_dict(torch.load("runs/mdnrnn.pt", map_location=wm.DEVICE))
params = torch.load("runs/controller.pt", map_location=wm.DEVICE)
torch.manual_seed(3)

# ---- Figure: training curves of V, M and C ------------------------------------------
smooth = lambda y, k: np.convolve(y, np.ones(k) / k, mode="valid")
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.0))
ax = ax.ravel()
rec = smooth(np.array(log["vae"])[:, 0], 50)
ax[0].plot(np.arange(50, 50 + len(rec)), rec, color=BLUE)
ax[0].set(title="(a) V: reconstruction loss", xlabel="training step", yscale="log")
m = np.array(log["mdnrnn"])
ax[1].plot(np.arange(1, len(m) + 1), m[:, 0], color=BLUE)
ax[1].set(title="(b) M: NLL of next z (per dim)", xlabel="epoch")
ax[2].plot(np.arange(1, len(m) + 1), m[:, 1], color=BLUE)
ax[2].set(title="(c) M: reward MSE", xlabel="epoch", yscale="log")
es = np.array(log["cmaes"])
gens = np.arange(1, len(es) + 1)
ax[3].plot(gens, es[:, 1], color=ORANGE, label="best candidate")
ax[3].plot(gens, es[:, 0], color=BLUE, label="population mean")
ax[3].axhline(5, color=MUTED, lw=0.8, ls="--")
ax[3].text(len(es), 5.07, "maximum (5 balls)", ha="right", va="bottom", color=MUTED,
           fontsize=8)
ax[3].set(title="(d) C: return inside the dream", xlabel="generation", ylim=(0, 5.6))
ax[3].legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig("fig_training.png", dpi=220)

# ---- Figure: reality vs. what V, M and C "see" --------------------------------------
T = 32
env, agent = wm.CatchEnv(7), wm.WorldModelAgent(vae, rnn, params)
agent.reset()
frames, actions = [env.reset()], []
for t in range(T):                                      # C plays in the real environment
    actions.append(agent(frames[-1]))
    frames.append(env.step(actions[-1])[0])

with torch.no_grad():
    mu, _ = vae.encode(wm.to_tensor(np.stack(frames)))
    recon = vae.decode(mu)                              # what the agent "sees" through V
    z, state, predicted = mu[:1], None, [vae.decode(mu[:1])]
    for t in range(T):                         # M imagines the result of the same actions
        a = torch.tensor([[[actions[t]]]], device=wm.DEVICE)
        logpi, mmu, logsigma, _, state = rnn(z[:, None], a, state)
        z = wm.sample_mdn(logpi[:, 0], mmu[:, 0], logsigma[:, 0])
        predicted.append(vae.decode(z))
    z, state, dream = mu[:1], None, [vae.decode(mu[:1])]
    h = torch.zeros(1, cfg.hidden, device=wm.DEVICE)
    for t in range(T):                         # C plays inside the dream (closed loop)
        a = wm.controller_action(params, z[None], h[None])[0]
        logpi, mmu, logsigma, _, state = rnn(z[:, None], a[:, None], state)
        z, h = wm.sample_mdn(logpi[:, 0], mmu[:, 0], logsigma[:, 0]), state[0][0]
        dream.append(vae.decode(z))

to_numpy = lambda x: x.permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy()   # [N, 64, 64, 3]
rows = [("Reality", np.stack(frames) / 255.0),
        ("V's\nreconstruction", to_numpy(recon)),
        ("M's prediction\n(same actions)", to_numpy(torch.cat(predicted))),
        ("Dream\n(C in control)", to_numpy(torch.cat(dream)))]
times = [0, 5, 10, 15, 16, 21, 26, 31]
fig, ax = plt.subplots(len(rows), len(times), figsize=(7.2, 4.1))
for r, (label, images) in enumerate(rows):
    for c, t in enumerate(times):
        ax[r, c].imshow(images[t], interpolation="nearest")
        ax[r, c].set_xticks([])
        ax[r, c].set_yticks([])
        ax[r, c].grid(False)
        if r == 0:
            ax[r, c].set_title(f"t = {t}", color=INK)
    ax[r, 0].set_ylabel(label, rotation=0, ha="right", va="center", color=INK, labelpad=8)
fig.tight_layout()
fig.savefig("fig_dream.png", dpi=220)
print("Saved fig_training.png and fig_dream.png")
