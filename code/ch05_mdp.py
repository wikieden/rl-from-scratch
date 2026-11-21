"""第 5 章 · 马尔可夫决策过程 MDP：动力学 p(s', r | s, a)（式 3.2），每组 (s, a) 加起来 = 1（式 3.3），
由它算出 p(s' | s, a)（式 3.4）和 r(s, a)（式 3.5）。"""

from rl_envs import ROBOT_ACTIONS, ROBOT_STATES, robot_mdp


def check_sums(P):
    return {(s, a): sum(p for p, *_ in P[s][a]) for s in P for a in P[s]}


def transition_prob(P, s, a, s_next):
    return sum(p for p, ns, _, _ in P[s][a] if ns == s_next)


def expected_reward(P, s, a):
    return sum(p * r for p, _, r, _ in P[s][a])


def main():
    P = robot_mdp()
    for s in P:
        for a in P[s]:
            rows = "；".join(f"→{ROBOT_STATES[ns]} 奖励 {r:+g} 概率 {p:g}" for p, ns, r, _ in P[s][a])
            print(f"{ROBOT_STATES[s]} + {ROBOT_ACTIONS[a]}：{rows}")
    print("每组概率之和：", set(round(v, 9) for v in check_sums(P).values()))
    print("p(高电 | 低电, 搜索) =", transition_prob(P, 1, 0, 0))
    print("r(低电, 搜索) =", round(expected_reward(P, 1, 0), 4), "；r(高电, 搜索) =", expected_reward(P, 0, 0))


if __name__ == "__main__":
    main()
