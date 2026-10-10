"""
world_model.py: a complete, minimal World Models agent in PyTorch.

It follows the V-M-C design of Ha & Schmidhuber (2018):
  V  a convolutional VAE that compresses each 64x64 RGB frame into a vector z (32 numbers)
  M  an MDN-RNN (LSTM + mixture density output) that predicts the next z and the reward
  C  a tiny linear controller, trained with CMA-ES entirely inside M's "dream"

main() runs the whole pipeline:
  1. collect episodes with a random policy in the real environment
  2. train V on the frames, then encode every frame into (mu, logvar)
  3. train M on the latent sequences
  4. evolve C inside the dream (the real environment is not used in this step)
  5. test C in the real environment and compare it with baselines, including
     a planner that reuses the same world model (model-predictive control with CEM)

Install:  pip install torch numpy cma
Run:      python world_model.py
"""
import json
import math
import os
import time
from dataclasses import asdict, dataclass

import cma  # CMA-ES optimiser
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 0. Settings
# ---------------------------------------------------------------------------
@dataclass
class Config:
    seed: int = 0
    out_dir: str = "runs"
    # data
    n_episodes: int = 300        # random-policy episodes (300 x 80 = 24,000 steps)
    # V: convolutional VAE
    z_dim: int = 32
    vae_epochs: int = 3
    vae_batch: int = 32
    vae_lr: float = 1e-3
    kl_tolerance: float = 0.5    # "free bits": KL below 0.5 nats per dim is not penalised
    # M: MDN-RNN
    hidden: int = 256
    n_mix: int = 5
    rnn_epochs: int = 60
    rnn_batch: int = 16
    rnn_lr: float = 1e-3
    # C: controller evolved inside the dream
    temperature: float = 1.15    # > 1 makes the dream noisier, so it is harder to exploit
    popsize: int = 32
    dream_rollouts: int = 16     # dreams per candidate; fitness = average dream return
    generations: int = 80
    sigma0: float = 0.1
    # evaluation and planning
    eval_episodes: int = 100
    plan_episodes: int = 20
    plan_horizon: int = 16       # imagine 16 steps ahead (one full ball drop)
    plan_candidates: int = 100
    plan_elites: int = 10
    plan_iters: int = 2


# ---------------------------------------------------------------------------
# 1. The environment: a tiny "Catch" game, observed only as 64x64 RGB frames
# ---------------------------------------------------------------------------
class CatchEnv:
    """A ball falls from the top; move the paddle (action in [-1, 1]) to catch it.
    Reward is +1 per catch. Each episode has 5 balls, i.e. 80 steps."""
    SIZE = 64
    PADDLE_W, PADDLE_H, PADDLE_Y, PADDLE_SPEED = 16, 4, 58, 4.0
    BALL_R, BALL_Y0, BALL_SPEED = 4, 6.0, 3.0
    PADDLE_RGB, BALL_RGB = (60, 160, 255), (255, 140, 40)    # blue paddle, orange ball
    N_BALLS = 5
    STEPS_PER_BALL = math.ceil((PADDLE_Y - BALL_R - BALL_Y0) / BALL_SPEED)  # 16
    EPISODE_LEN = N_BALLS * STEPS_PER_BALL                                   # 80

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.ys, self.xs = np.mgrid[0:self.SIZE, 0:self.SIZE]

    def reset(self):
        half = self.PADDLE_W / 2
        self.paddle_x = self.rng.uniform(half, self.SIZE - half)
        self.balls_left = self.N_BALLS
        self._new_ball()
        return self.render()

    def _new_ball(self):
        self.ball_x = self.rng.uniform(self.BALL_R, self.SIZE - self.BALL_R)
        self.ball_y = self.BALL_Y0

    def step(self, action):
        half = self.PADDLE_W / 2
        move = float(np.clip(action, -1, 1)) * self.PADDLE_SPEED
        self.paddle_x = float(np.clip(self.paddle_x + move, half, self.SIZE - half))
        self.ball_y += self.BALL_SPEED
        reward = 0.0
        if self.ball_y >= self.PADDLE_Y - self.BALL_R:     # ball reached the paddle row
            reward = float(abs(self.ball_x - self.paddle_x) <= half + self.BALL_R / 2)
            self.balls_left -= 1
            self._new_ball()                                # the next ball: random x
        return self.render(), reward, self.balls_left == 0

    def render(self):
        img = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
        x0, y0 = int(round(self.paddle_x - self.PADDLE_W / 2)), self.PADDLE_Y
        img[y0:y0 + self.PADDLE_H, x0:x0 + self.PADDLE_W] = self.PADDLE_RGB
        dist2 = (self.xs - self.ball_x) ** 2 + (self.ys - self.ball_y) ** 2
        img[dist2 <= self.BALL_R ** 2] = self.BALL_RGB
        return img


class RandomPolicy:
    """Random actions, each held for 1-8 steps (explores more widely than pure noise)."""
    def __init__(self, rng):
        self.rng, self.action, self.steps_left = rng, 0.0, 0

    def __call__(self, obs=None):
        if self.steps_left == 0:
            self.action = self.rng.uniform(-1, 1)
            self.steps_left = int(self.rng.integers(1, 9))
        self.steps_left -= 1
        return self.action


def collect_episodes(n, seed):
    """Step 1: play n random-policy episodes; record frames, actions and rewards."""
    env, rng, T = CatchEnv(seed), np.random.default_rng(seed + 1), CatchEnv.EPISODE_LEN
    obs = np.zeros((n, T + 1, 64, 64, 3), np.uint8)   # frames  x_0 ... x_T
    act = np.zeros((n, T, 1), np.float32)             # actions a_0 ... a_{T-1}
    rew = np.zeros((n, T), np.float32)                # rewards r_0 ... r_{T-1}
    for i in range(n):
        policy = RandomPolicy(rng)
        obs[i, 0] = env.reset()
        for t in range(T):
            act[i, t, 0] = policy()
            obs[i, t + 1], rew[i, t], _ = env.step(act[i, t, 0])
    return obs, act, rew


def to_tensor(frames):
    """uint8 frames [B, 64, 64, 3] -> float tensor [B, 3, 64, 64], values in [0, 1]."""
    return torch.from_numpy(frames).to(DEVICE).permute(0, 3, 1, 2).float() / 255.0


# ---------------------------------------------------------------------------
# 2. V: the convolutional VAE (same layer sizes as in the World Models paper)
# ---------------------------------------------------------------------------
class ConvVAE(nn.Module):
    def __init__(self, z_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),       # 64x64 -> 31x31
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),      # -> 14x14
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),     # -> 6x6
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU(),    # -> 2x2
            nn.Flatten())                                   # -> 1024
        self.fc_mu = nn.Linear(1024, z_dim)
        self.fc_logvar = nn.Linear(1024, z_dim)
        self.fc_dec = nn.Linear(z_dim, 1024)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, 5, stride=2), nn.ReLU(),  # 1x1 -> 5x5
            nn.ConvTranspose2d(128, 64, 5, stride=2), nn.ReLU(),    # -> 13x13
            nn.ConvTranspose2d(64, 32, 6, stride=2), nn.ReLU(),     # -> 30x30
            nn.ConvTranspose2d(32, 3, 6, stride=2))                 # -> 64x64 (logits)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode_logits(self, z):
        return self.decoder(self.fc_dec(z).view(-1, 1024, 1, 1))

    def decode(self, z):
        return torch.sigmoid(self.decode_logits(z))        # image with values in [0, 1]

    def forward(self, x):
        mu, logvar = self.encode(x)
        # reparameterisation trick: z = mu + sigma * eps with eps ~ N(0, I)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode_logits(z), mu, logvar


def vae_loss(x, logits, mu, logvar, kl_tolerance):
    """Reconstruction (binary cross-entropy per pixel) + KL divergence, both summed per
    image and averaged over the batch. As in the original code, the KL term is floored
    at kl_tolerance * z_dim ("free bits")."""
    bce = F.binary_cross_entropy_with_logits(logits, x, reduction="none")
    rec = bce.sum(dim=(1, 2, 3)).mean()
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
    kl = torch.clamp(kl, min=kl_tolerance * mu.shape[1]).mean()
    return rec + kl, rec, kl


def train_vae(frames, cfg):
    """Step 2a: learn to compress single frames (unsupervised: no actions, no rewards)."""
    vae = ConvVAE(cfg.z_dim).to(DEVICE)
    opt = torch.optim.Adam(vae.parameters(), lr=cfg.vae_lr)
    log, steps_per_epoch = [], math.ceil(len(frames) / cfg.vae_batch)
    for epoch in range(cfg.vae_epochs):
        perm = np.random.permutation(len(frames))
        for i in range(0, len(frames), cfg.vae_batch):
            x = to_tensor(frames[perm[i:i + cfg.vae_batch]])
            logits, mu, logvar = vae(x)
            loss, rec, kl = vae_loss(x, logits, mu, logvar, cfg.kl_tolerance)
            opt.zero_grad()
            loss.backward()
            opt.step()
            log.append([rec.item(), kl.item()])              # losses of every step
        rec_avg, kl_avg = np.mean(log[-steps_per_epoch:], axis=0)
        print(f"   VAE epoch {epoch + 1}/{cfg.vae_epochs}: "
              f"reconstruction {rec_avg:6.1f}  KL {kl_avg:5.1f}")
    return vae, log


@torch.no_grad()
def encode_episodes(vae, obs, batch=1000):
    """Step 2b: turn every frame into (mu, logvar). M trains on these, never on pixels."""
    flat = obs.reshape(-1, 64, 64, 3)
    mus, logvars = [], []
    for i in range(0, len(flat), batch):
        mu, logvar = vae.encode(to_tensor(flat[i:i + batch]))
        mus.append(mu.cpu())
        logvars.append(logvar.cpu())
    shape = obs.shape[:2] + (-1,)                     # [episodes, T + 1, z_dim]
    return torch.cat(mus).view(shape), torch.cat(logvars).view(shape)


# ---------------------------------------------------------------------------
# 3. M: the MDN-RNN, which models P(z_{t+1} | a_t, z_t, h_t) and predicts r_t
# ---------------------------------------------------------------------------
class MDNRNN(nn.Module):
    def __init__(self, z_dim=32, action_dim=1, hidden=256, n_mix=5):
        super().__init__()
        self.z_dim, self.n_mix, self.hidden = z_dim, n_mix, hidden
        self.lstm = nn.LSTM(z_dim + action_dim, hidden, batch_first=True)
        # for every z-dimension and every mixture component: log pi, mu and log sigma
        self.mdn = nn.Linear(hidden, 3 * n_mix * z_dim)
        self.reward = nn.Linear(hidden, 1)

    def forward(self, z, a, state=None):
        """z: [B, T, z_dim], a: [B, T, action_dim]. For every step t this returns the
        mixture over z_{t+1}, the predicted reward r_t and the LSTM state (h, c) after
        seeing (z_t, a_t)."""
        out, state = self.lstm(torch.cat([z, a], dim=-1), state)
        B, T, _ = out.shape
        p = self.mdn(out).view(B, T, self.z_dim, 3 * self.n_mix)
        logpi, mu, logsigma = p.split(self.n_mix, dim=-1)  # each [B, T, z_dim, n_mix]
        logpi = F.log_softmax(logpi, dim=-1)               # mixture weights sum to 1
        return logpi, mu, logsigma, self.reward(out).squeeze(-1), state


def mdn_nll(logpi, mu, logsigma, target):
    """Negative log-likelihood of target [B, T, z_dim] under a factored mixture of
    Gaussians, averaged over batch, time and z-dimensions (as in the original code)."""
    target = target.unsqueeze(-1)
    # log N(target; mu, sigma^2) for every component
    log_gauss = (-0.5 * ((target - mu) / logsigma.exp()) ** 2
                 - logsigma - 0.5 * math.log(2 * math.pi))
    return -torch.logsumexp(logpi + log_gauss, dim=-1).mean()


def sample_mdn(logpi, mu, logsigma, tau=1.0):
    """Sample z_{t+1}: pick a component per dimension, then sample from that Gaussian.
    Temperature tau > 1 flattens the mixture weights and widens each Gaussian
    (as in SketchRNN)."""
    k = torch.distributions.Categorical(logits=logpi / tau).sample().unsqueeze(-1)
    mu_k = mu.gather(-1, k).squeeze(-1)
    sigma_k = logsigma.gather(-1, k).squeeze(-1).exp()
    return mu_k + sigma_k * math.sqrt(tau) * torch.randn_like(mu_k)


def train_mdnrnn(mu, logvar, act, rew, cfg):
    """Step 3: teach M to predict the next latent vector and the reward."""
    rnn = MDNRNN(cfg.z_dim, act.shape[-1], cfg.hidden, cfg.n_mix).to(DEVICE)
    opt = torch.optim.Adam(rnn.parameters(), lr=cfg.rnn_lr)
    log, n = [], len(mu)
    for epoch in range(cfg.rnn_epochs):
        perm, sums = torch.randperm(n), np.zeros(2)
        for i in range(0, n, cfg.rnn_batch):
            idx = perm[i:i + cfg.rnn_batch]
            m, lv = mu[idx].to(DEVICE), logvar[idx].to(DEVICE)
            # a fresh sample z ~ N(mu, sigma^2) for every batch (data augmentation)
            z = m + torch.exp(0.5 * lv) * torch.randn_like(m)
            a, r = act[idx].to(DEVICE), rew[idx].to(DEVICE)
            # teacher forcing: inputs are the true z_0 ... z_{T-1}, targets z_1 ... z_T
            logpi, mmu, logsigma, r_pred, _ = rnn(z[:, :-1], a)
            loss_z = mdn_nll(logpi, mmu, logsigma, z[:, 1:])
            loss_r = F.mse_loss(r_pred, r)
            opt.zero_grad()
            (loss_z + loss_r).backward()
            nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)
            opt.step()
            sums += [loss_z.item() * len(idx), loss_r.item() * len(idx)]
        log.append((sums / n).tolist())
        if (epoch + 1) % 10 == 0:
            print(f"   MDN-RNN epoch {epoch + 1:3d}/{cfg.rnn_epochs}: "
                  f"z NLL {log[-1][0]:6.3f}  reward MSE {log[-1][1]:.4f}")
    return rnn, log


# ---------------------------------------------------------------------------
# 4. C: a linear controller, evolved with CMA-ES inside the dream
# ---------------------------------------------------------------------------
def controller_action(params, z, h):
    """a = tanh(W [z, h] + b), for many controllers at once.
    params: [P, z_dim + hidden + 1], one row per controller;
    z: [P, R, z_dim] and h: [P, R, hidden], for R rollouts per controller."""
    W, b = params[:, :-1], params[:, -1:]
    x = torch.cat([z, h], dim=-1)
    return torch.tanh(torch.einsum("prf,pf->pr", x, W) + b).unsqueeze(-1)  # [P, R, 1]


@torch.no_grad()
def dream_returns(rnn, params, z0_mu, z0_logvar, cfg):
    """Run P controllers x R dreams in parallel inside M and return the average predicted
    return of each controller. No pixels and no real environment are involved."""
    P, R = len(params), cfg.dream_rollouts
    params = torch.as_tensor(np.asarray(params), dtype=torch.float32, device=DEVICE)
    idx = torch.randint(len(z0_mu), (R,)).repeat(P)    # same R start frames for all P
    m, lv = z0_mu[idx].to(DEVICE), z0_logvar[idx].to(DEVICE)
    z = m + torch.exp(0.5 * lv) * torch.randn_like(m)       # z_0 from real first frames
    h, state = torch.zeros(P * R, cfg.hidden, device=DEVICE), None
    total = torch.zeros(P * R, device=DEVICE)
    for t in range(CatchEnv.EPISODE_LEN):
        a = controller_action(params, z.view(P, R, -1), h.view(P, R, -1)).view(P * R, 1)
        logpi, mu, logsigma, r, state = rnn(z[:, None], a[:, None], state)
        total += r[:, 0]                                     # predicted reward r_t
        # the dream: z_{t+1} is sampled from M (at temperature tau) instead of observed
        z = sample_mdn(logpi[:, 0], mu[:, 0], logsigma[:, 0], cfg.temperature)
        h = state[0][0]                                      # h_{t+1}
    return total.view(P, R).mean(dim=1).cpu().numpy()


def train_controller_in_dream(rnn, z0_mu, z0_logvar, cfg):
    """Step 4: CMA-ES proposes controllers, the dream scores them, CMA-ES adapts."""
    n_params = cfg.z_dim + cfg.hidden + 1                   # 32 + 256 + 1 = 289
    options = {"popsize": cfg.popsize, "seed": cfg.seed + 1, "verbose": -9}
    es = cma.CMAEvolutionStrategy(np.zeros(n_params), cfg.sigma0, options)
    log = []
    for gen in range(cfg.generations):
        candidates = es.ask()
        fitness = dream_returns(rnn, candidates, z0_mu, z0_logvar, cfg)
        es.tell(candidates, (-fitness).tolist())             # CMA-ES minimises, so negate
        log.append([float(fitness.mean()), float(fitness.max())])
        if (gen + 1) % 10 == 0:
            print(f"   generation {gen + 1:3d}: "
                  f"dream return mean {log[-1][0]:.2f}  best {log[-1][1]:.2f}")
    mean = torch.tensor(es.result.xfavorite, dtype=torch.float32, device=DEVICE)
    return mean[None], log             # the mean of the final search distribution


# ---------------------------------------------------------------------------
# 5. The agent in the real world, and a planning alternative to C
# ---------------------------------------------------------------------------
@torch.no_grad()
def cem_plan(rnn, z, state, cfg):
    """Model-predictive control: imagine many action sequences inside M, keep the best
    ones (cross-entropy method), and return only the first action of the final plan."""
    H, N, E = cfg.plan_horizon, cfg.plan_candidates, cfg.plan_elites
    mean, std = torch.zeros(H, device=DEVICE), torch.ones(H, device=DEVICE)
    for _ in range(cfg.plan_iters):
        actions = (mean + std * torch.randn(N, H, device=DEVICE)).clamp(-1, 1)   # [N, H]
        s = None if state is None else tuple(x.repeat(1, N, 1) for x in state)
        zz, ret = z.repeat(N, 1), torch.zeros(N, device=DEVICE)
        for k in range(H):
            logpi, mu, logsigma, r, s = rnn(zz[:, None], actions[:, k, None, None], s)
            ret += r[:, 0]
            zz = sample_mdn(logpi[:, 0], mu[:, 0], logsigma[:, 0])
        elite = actions[ret.topk(E).indices]                # the E best imagined futures
        mean, std = elite.mean(dim=0), elite.std(dim=0) + 0.05
    return mean[0].item()


class WorldModelAgent:
    """The full agent, exactly like the paper's rollout() pseudocode:
         z = V(obs);  a = C([z, h]);  obs, r, done = env.step(a);  h = M(z, a, h)
    With plan=True, the action comes from CEM planning inside M instead of from C."""
    def __init__(self, vae, rnn, params=None, cfg=None, plan=False):
        self.vae, self.rnn, self.params, self.cfg, self.plan = vae, rnn, params, cfg, plan

    def reset(self):
        self.state, self.h = None, torch.zeros(1, self.rnn.hidden, device=DEVICE)

    @torch.no_grad()
    def __call__(self, obs):
        z = self.vae.encode(to_tensor(obs[None]))[0]       # z_t = mean of q(z | frame)
        if self.plan:
            a_plan = cem_plan(self.rnn, z, self.state, self.cfg)
            a = torch.tensor([[a_plan]], device=DEVICE)
        else:
            a = controller_action(self.params, z[None], self.h[None])[0]
        *_, self.state = self.rnn(z[:, None], a[:, None], self.state)   # memory update
        self.h = self.state[0][0]
        return a.item()


def handcoded_policy(obs):
    """Baseline: find the orange ball and blue paddle by colour; move toward the ball."""
    ball_x = np.argwhere(obs[..., 0] > 200)[:, 1].mean()
    paddle_x = np.argwhere(obs[..., 2] > 200)[:, 1].mean()
    return float(np.clip((ball_x - paddle_x) / CatchEnv.PADDLE_SPEED, -1, 1))


def evaluate(policy, n_episodes, seed=12345):
    """Balls caught per episode (mean, std; max 5). All policies see the same balls."""
    env, scores = CatchEnv(seed), []
    for _ in range(n_episodes):
        obs, done, total = env.reset(), False, 0.0
        if hasattr(policy, "reset"):
            policy.reset()
        while not done:
            obs, r, done = env.step(policy(obs))
            total += r
        scores.append(total)
    return float(np.mean(scores)), float(np.std(scores))


# ---------------------------------------------------------------------------
# 6. Put it all together
# ---------------------------------------------------------------------------
def main(cfg=None):
    cfg = cfg or Config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    start, timings = time.time(), {}

    print("1) Collecting random-policy episodes")
    obs, act, rew = collect_episodes(cfg.n_episodes, cfg.seed)
    catches = rew.mean() * CatchEnv.EPISODE_LEN
    print(f"   {act.shape[0] * act.shape[1]} steps, {catches:.2f} catches/episode")
    timings["collect"] = time.time() - start

    print("2) Training V (convolutional VAE)")
    vae, vae_log = train_vae(obs.reshape(-1, 64, 64, 3), cfg)
    mu, logvar = encode_episodes(vae, obs)
    timings["vae"] = time.time() - start

    print("3) Training M (MDN-RNN)")
    act, rew = torch.from_numpy(act), torch.from_numpy(rew)
    rnn, rnn_log = train_mdnrnn(mu, logvar, act, rew, cfg)
    timings["rnn"] = time.time() - start

    print("4) Evolving C inside the dream (CMA-ES)")
    params, es_log = train_controller_in_dream(rnn, mu[:, 0], logvar[:, 0], cfg)
    timings["controller"] = time.time() - start
    torch.save(vae.state_dict(), os.path.join(cfg.out_dir, "vae.pt"))
    torch.save(rnn.state_dict(), os.path.join(cfg.out_dir, "mdnrnn.pt"))
    torch.save(params.cpu(), os.path.join(cfg.out_dir, "controller.pt"))

    print("5) Testing in the real environment")
    random_policy = RandomPolicy(np.random.default_rng(1))
    agent = WorldModelAgent(vae, rnn, params)                    # V + M + dream-trained C
    planner = WorldModelAgent(vae, rnn, cfg=cfg, plan=True)      # V + M + CEM planning
    results = {
        "random policy": evaluate(random_policy, cfg.eval_episodes),
        "world model + dream-trained C": evaluate(agent, cfg.eval_episodes),
        "world model + CEM planning": evaluate(planner, cfg.plan_episodes),
        "hand-coded policy": evaluate(handcoded_policy, cfg.eval_episodes),
    }
    for name, (mean, std) in results.items():
        print(f"   {name:30s} {mean:.2f} +/- {std:.2f} balls caught (out of 5)")
    timings["total"] = time.time() - start
    log = {"config": asdict(cfg), "vae": vae_log, "mdnrnn": rnn_log, "cmaes": es_log,
           "results": results, "timings_s": timings}
    with open(os.path.join(cfg.out_dir, "log.json"), "w") as f:
        json.dump(log, f, indent=1)
    print(f"Finished in {timings['total'] / 60:.1f} minutes. "
          f"Models and logs saved to '{cfg.out_dir}/'.")


if __name__ == "__main__":
    main()
