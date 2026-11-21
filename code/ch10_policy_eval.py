"""第 10 章 · 策略评估：把 Bellman 期望方程当成更新规则反复套（式 4.5），v_0 = 0 → v_1 → v_2 → ... → v_π。"""

import numpy as np

from ch08_bellman import q_from_v
from rl_envs import gridworld_4x4, uniform_policy


def sweep(P, pi, v, gamma):
    """一轮：每个状态都用旧的 v 算一遍右边（同步更新，书里叫 two-array 版本）。"""
    return np.array([sum(pa * q_from_v(P, v, gamma, s, a) for a, pa in pi[s].items()) for s in P])


def policy_evaluation(P, pi, gamma=1.0, theta=1e-6, history=False):
    v, hist = np.zeros(len(P)), []
    while True:
        hist.append(v)
        new = sweep(P, pi, v, gamma)
        if np.max(np.abs(new - v)) < theta:
            return (new, hist) if history else new
        v = new


def main():
    P = gridworld_4x4()
    v, hist = policy_evaluation(P, uniform_policy(P), history=True)
    for k in [0, 1, 2, 3, 10]:
        print(f"k = {k}：\n{np.round(hist[k].reshape(4, 4), 1)}")
    print(f"收敛（{len(hist)} 轮）：\n{np.round(v.reshape(4, 4), 1)}")


if __name__ == "__main__":
    main()
