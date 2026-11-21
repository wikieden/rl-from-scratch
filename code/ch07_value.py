"""第 7 章 · 价值函数：v_π(s) = E_π[G_t | S_t = s]（式 3.12），q_π(s, a)（式 3.13）。
这里先用"跑很多次、把回报取平均"来估计，第 8 章再用 Bellman 方程精确算。"""

import numpy as np

from ch06_policy import sample_action
from rl_envs import ROBOT_PI, ROBOT_STATES, robot_mdp

GAMMA = 0.9


def step(P, s, a, rng):
    outs = P[s][a]
    p, ns, r, done = outs[rng.choice(len(outs), p=[o[0] for o in outs])]
    return ns, r


def mc_value(P, pi, s0, n_episodes=2000, horizon=100, gamma=GAMMA, seed=0, first_action=None):
    """从 s0 出发跑 n 次（每次截到 horizon 步，γ^100 ≈ 3e-5 可忽略），把折扣回报取平均。"""
    rng, total = np.random.default_rng(seed), 0.0
    for _ in range(n_episodes):
        s, G, disc = s0, 0.0, 1.0
        for t in range(horizon):
            a = first_action if (t == 0 and first_action is not None) else sample_action(pi, s, rng)
            s, r = step(P, s, a, rng)
            G += disc * r
            disc *= gamma
        total += G
    return total / n_episodes


def main():
    P, pi = robot_mdp(), ROBOT_PI
    for s in pi:
        print(f"v_π({ROBOT_STATES[s]}) ≈", round(mc_value(P, pi, s), 2))
    print("q_π(低电, 充电) ≈", round(mc_value(P, pi, 1, first_action=2), 2))
    print("q_π(低电, 搜索) ≈", round(mc_value(P, pi, 1, first_action=0), 2))


if __name__ == "__main__":
    main()
