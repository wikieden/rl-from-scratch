"""第 15 章 · TD(0)：走一步就更新 V(S) ← V(S) + α[R + γV(S') − V(S)]（式 6.2）；中括号里是 TD 误差 δ（式 6.5）。"""

import numpy as np

from ch14_monte_carlo import mc_prediction, rms
from rl_envs import random_walk_episode


def td_update(V, s, r, s_next, alpha, gamma=1.0):
    target = r + (0.0 if s_next is None else gamma * V[s_next])
    delta = target - V[s]
    V[s] += alpha * delta
    return delta


def td_prediction(n_episodes, alpha=0.1, seed=0, init=0.5):
    rng = np.random.default_rng(seed)
    V = np.full(5, init)
    for _ in range(n_episodes):
        for s, r, ns in random_walk_episode(rng):
            td_update(V, s, r, ns, alpha)
    return V


def compare(n_episodes=100, runs=100):
    td = np.mean([rms(td_prediction(n_episodes, 0.1, seed=k)) for k in range(runs)])
    mc = np.mean([rms(mc_prediction(n_episodes, 0.02, seed=k)) for k in range(runs)])
    return td, mc


def main():
    V = np.full(5, 0.5)
    d = td_update(V, 4, 1.0, None, 0.1)  # 在 E 往右走出去，拿到 +1
    print(f"E 走出右边：δ = {d:.2f}，V(E) 0.5 → {V[4]:.2f}")
    d = td_update(V, 3, 0.0, 4, 0.1)  # D → E，奖励 0
    print(f"D → E：δ = {d:.3f}，V(D) 0.5 → {V[3]:.3f}")
    td, mc = compare()
    print(f"100 局后平均 RMS 误差：TD(α=0.1) {td:.3f}，MC(α=0.02) {mc:.3f}")


if __name__ == "__main__":
    main()
