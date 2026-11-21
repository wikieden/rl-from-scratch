"""第 9 章 · 最优价值与 Bellman 最优方程：v*(s) = max_a Σ p(s', r|s, a)[r + γ v*(s')]（式 3.19），
q*(s, a) = Σ p [r + γ max_a' q*(s', a')]（式 3.20）。"""

import numpy as np

from ch08_bellman import q_from_v
from rl_envs import MOVE_NAMES, ROBOT_ACTIONS, ROBOT_STATES, gridworld_5x5, robot_mdp


def optimal_v(P, gamma, tol=1e-10):
    """反复套最优方程直到不再变（这就是第 12 章的价值迭代）。"""
    v = np.zeros(len(P))
    while True:
        new = np.array([max(q_from_v(P, v, gamma, s, a) for a in P[s]) for s in P])
        if np.max(np.abs(new - v)) < tol:
            return new
        v = new


def greedy_actions(P, v, gamma, s, eps=1e-6):
    qs = {a: q_from_v(P, v, gamma, s, a) for a in P[s]}
    best = max(qs.values())
    return [a for a, q in qs.items() if q > best - eps]


def main():
    P = robot_mdp()
    v = optimal_v(P, 0.9)
    for s in P:
        acts = [ROBOT_ACTIONS[a] for a in greedy_actions(P, v, 0.9, s)]
        print(f"v*({ROBOT_STATES[s]}) = {v[s]:.3f}，最优动作 {acts}")
    G = gridworld_5x5()
    vg = optimal_v(G, 0.9)
    print("5×5 网格 v*（图 3.5 中）：")
    print(np.round(vg.reshape(5, 5), 1))
    print("右下角 (4,4) 的最优动作：", [MOVE_NAMES[a] for a in greedy_actions(G, vg, 0.9, 24)])


if __name__ == "__main__":
    main()
