"""第 6 章 · 策略 π(a|s)：每个状态下各动作的概率。"""

import numpy as np

from ch05_mdp import expected_reward
from rl_envs import ROBOT_ACTIONS, ROBOT_PI, ROBOT_STATES, robot_mdp


def sample_action(pi, s, rng):
    acts = list(pi[s])
    return acts[rng.choice(len(acts), p=[pi[s][a] for a in acts])]


def expected_immediate_reward(P, pi, s):
    """E_π[R_{t+1} | S_t = s] = Σ_a π(a|s) r(s, a)。"""
    return sum(pi[s][a] * expected_reward(P, s, a) for a in pi[s])


def main():
    P, pi, rng = robot_mdp(), ROBOT_PI, np.random.default_rng(0)
    for s in pi:
        print(ROBOT_STATES[s], {ROBOT_ACTIONS[a]: p for a, p in pi[s].items()})
    picks = [sample_action(pi, 1, rng) for _ in range(1000)]
    print("低电时抽 1000 次：", {ROBOT_ACTIONS[a]: picks.count(a) for a in pi[1]})
    for s in pi:
        print(f"E[下一步奖励 | {ROBOT_STATES[s]}] =", round(expected_immediate_reward(P, pi, s), 4))


if __name__ == "__main__":
    main()
