"""第 12 章 · 价值迭代：v_{k+1}(s) = max_a Σ p [r + γ v_k(s')]（式 4.10），评估只做一轮就改进。"""

import numpy as np

from ch08_bellman import q_from_v
from ch11_policy_iteration import greedy_policy, show_policy
from rl_envs import gridworld_4x4, gridworld_5x5


def value_iteration(P, gamma, theta=1e-6):
    v, sweeps = np.zeros(len(P)), 0
    while True:
        new = np.array([max(q_from_v(P, v, gamma, s, a) for a in P[s]) for s in P])
        sweeps += 1
        if np.max(np.abs(new - v)) < theta:
            return new, sweeps
        v = new


def main():
    P = gridworld_4x4()
    v, n = value_iteration(P, 1.0)
    print(f"4×4 价值迭代 {n} 轮：\n{np.round(v.reshape(4, 4), 1)}")
    print(show_policy(greedy_policy(P, v, 1.0), 4))
    G = gridworld_5x5()
    vg, ng = value_iteration(G, 0.9)
    print(f"5×5 价值迭代 {ng} 轮，v*(A) = {vg[1]:.1f}")


if __name__ == "__main__":
    main()
