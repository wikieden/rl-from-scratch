"""第 11 章 · 策略改进与策略迭代：按 q_π 挑最好的动作得到更好的策略（式 4.9），评估 → 改进 → 评估 …（§4.3）。"""

import numpy as np

from ch08_bellman import q_from_v, solve_v
from ch10_policy_eval import policy_evaluation
from rl_envs import MOVE_NAMES, gridworld_4x4, gridworld_5x5, uniform_policy


def greedy_policy(P, v, gamma, eps=1e-9):
    """每个状态在 q 最大的动作之间平分概率（并列时都留着，方便看）。"""
    pi = {}
    for s in P:
        qs = {a: q_from_v(P, v, gamma, s, a) for a in P[s]}
        best = max(qs.values())
        top = [a for a, q in qs.items() if q > best - eps]
        pi[s] = {a: (1 / len(top) if a in top else 0.0) for a in P[s]}
    return pi


def policy_iteration(P, gamma, evaluate):
    pi, rounds = uniform_policy(P), 0
    while True:
        v = evaluate(P, pi, gamma)
        new = greedy_policy(P, v, gamma)
        rounds += 1
        if new == pi:
            return pi, v, rounds
        pi = new


def show_policy(pi, n):
    rows = []
    for r in range(n):
        rows.append(" ".join("".join(MOVE_NAMES[a] for a, p in pi[r * n + c].items() if p > 0) or "·"
                             for c in range(n)))
    return "\n".join(rows)


def main():
    P = gridworld_4x4()
    v_rand = policy_evaluation(P, uniform_policy(P))
    pi1 = greedy_policy(P, v_rand, 1.0)
    print("4×4：对随机策略的价值贪心一次：\n" + show_policy(pi1, 4))
    pi, v, rounds = policy_iteration(P, 1.0, lambda P, pi, g: policy_evaluation(P, pi, g))
    print(f"策略迭代 {rounds} 轮停下，价值：\n{np.round(v.reshape(4, 4), 1)}")
    G = gridworld_5x5()
    pi5, v5, r5 = policy_iteration(G, 0.9, solve_v)
    print(f"5×5 网格策略迭代 {r5} 轮，v*(A) = {v5[1]:.1f}")


if __name__ == "__main__":
    main()
