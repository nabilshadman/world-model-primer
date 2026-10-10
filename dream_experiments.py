"""
dream_experiments.py: when does a controller trained in the dream work in reality?
Reuses V and M saved by world_model.py and runs two small experiments:
  A. world-model quality: evolve C inside weaker world models (less data / less training)
  B. dream temperature: evolve C inside the full world model at several temperatures
Each controller is scored inside the dream (500 dreams) and in the real game
(100 episodes).

Run after world_model.py:  python dream_experiments.py   (about 12 minutes on 2 CPU cores)
"""
import json
from dataclasses import replace

import torch

import world_model as wm

cfg = wm.Config(**json.load(open("runs/log.json"))["config"])
vae = wm.ConvVAE(cfg.z_dim).to(wm.DEVICE)
vae.load_state_dict(torch.load("runs/vae.pt", map_location=wm.DEVICE))
full_m = wm.MDNRNN(cfg.z_dim, 1, cfg.hidden, cfg.n_mix).to(wm.DEVICE)
full_m.load_state_dict(torch.load("runs/mdnrnn.pt", map_location=wm.DEVICE))

# The same random-policy data as in world_model.py (same seed), encoded by the saved V
obs, act, rew = wm.collect_episodes(cfg.n_episodes, cfg.seed)
mu, logvar = wm.encode_episodes(vae, obs)
act, rew = torch.from_numpy(act), torch.from_numpy(rew)


def dream_vs_reality(rnn, tau):
    """Evolve C inside `rnn` at temperature tau; return its dream and real scores."""
    torch.manual_seed(cfg.seed)
    run_cfg = replace(cfg, temperature=tau)
    params, _ = wm.train_controller_in_dream(rnn, mu[:, 0], logvar[:, 0], run_cfg)
    run_cfg.dream_rollouts = 500
    dream = wm.dream_returns(rnn, params.cpu().numpy(), mu[:, 0], logvar[:, 0], run_cfg)
    dream = float(dream[0])
    real, real_std = wm.evaluate(wm.WorldModelAgent(vae, rnn, params), cfg.eval_episodes)
    return {"dream": dream, "real": real, "real_std": real_std}


results = {"A": [], "B": []}
print("A. World-model quality (temperature 1.15)")
for n_episodes, epochs in [(20, 60), (300, 3), (300, 60)]:
    if (n_episodes, epochs) == (cfg.n_episodes, cfg.rnn_epochs):
        rnn = full_m                            # the model trained by world_model.py
    else:
        torch.manual_seed(cfg.seed)
        weak_cfg = replace(cfg, rnn_epochs=epochs)
        rnn, _ = wm.train_mdnrnn(mu[:n_episodes], logvar[:n_episodes], act[:n_episodes],
                                 rew[:n_episodes], weak_cfg)
    r = {"episodes": n_episodes, "epochs": epochs,
         **dream_vs_reality(rnn, cfg.temperature)}
    results["A"].append(r)
    print(f"   M from {n_episodes:3d} episodes, {epochs:2d} epochs: "
          f"dream {r['dream']:.2f}, "
          f"reality {r['real']:.2f} +/- {r['real_std']:.2f}")

print("B. Dream temperature (full world model)")
for tau in [0.1, 0.5, 1.0, 1.15, 1.5]:
    r = results["A"][-1] if tau == cfg.temperature else dream_vs_reality(full_m, tau)
    results["B"].append({"temperature": tau, "dream": r["dream"], "real": r["real"],
                         "real_std": r["real_std"]})
    print(f"   tau = {tau:4.2f}: dream {r['dream']:.2f}, "
          f"reality {r['real']:.2f} +/- {r['real_std']:.2f}")

with open("runs/dream_experiments.json", "w") as f:
    json.dump(results, f, indent=1)
