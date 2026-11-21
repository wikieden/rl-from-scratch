"""第 8 章 · Bellman 期望方程（式 3.14）：v_π(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a) [r + γ v_π(s')]。
有限 MDP 时它是一组线性方程，可以一次解出来。"""

import numpy as np

from rl_envs import ROBOT_PI, ROBOT_STATES, gridworld_5x5, robot_mdp, uniform_policy


def policy_matrices(P, pi):
    """把 (P, π) 压成 r_π[s] = 期望即时奖励、P_π[s, s'] = 一步转移概率（终点之后不再有价值）。"""
    n = len(P)
    r_pi, P_pi = np.zeros(n), np.zeros((n, n))
    for s in P:
        for a, pa in pi[s].items():
            for p, ns, r, done in P[s][a]:
                r_pi[s] += pa * p * r
                if not done:
                    P_pi[s, ns] += pa * p
    return r_pi, P_pi


def solve_v(P, pi, gamma):
    """v = r_π + γ P_π v  ⇒  (I − γ P_π) v = r_π。"""
    r_pi, P_pi = policy_matrices(P, pi)
    return np.linalg.solve(np.eye(len(P)) - gamma * P_pi, r_pi)


def q_from_v(P, v, gamma, s, a):
    """q_π(s, a) = Σ p(s', r|s, a) [r + γ v_π(s')]。"""
    return sum(p * (r + (0 if done else gamma * v[ns])) for p, ns, r, done in P[s][a])


def bellman_rhs(P, pi, v, gamma, s):
    return sum(pa * q_from_v(P, v, gamma, s, a) for a, pa in pi[s].items())


def main():
    P, pi = robot_mdp(), ROBOT_PI
    v = solve_v(P, pi, 0.9)
    for s in P:
        print(f"v_π({ROBOT_STATES[s]}) = {v[s]:.3f}；Bellman 右边 = {bellman_rhs(P, pi, v, 0.9, s):.3f}")
    G = gridworld_5x5()
    vg = solve_v(G, uniform_policy(G), 0.9)
    print("5×5 网格、随机策略（图 3.2 右）：")
    print(np.round(vg.reshape(5, 5), 1))


if __name__ == "__main__":
    main()
