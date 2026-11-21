"""第 16 章 · SARSA（式 6.7）与 Q-learning（式 6.8）：悬崖行走（Example 6.6）。"""

import numpy as np

from rl_envs import CLIFF_COLS, CLIFF_GOAL, CLIFF_ROWS, CLIFF_START, MOVE_NAMES, cliff_step


def eps_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(4))
    q = Q[s]
    return int(rng.choice(np.flatnonzero(q == q.max())))


def run(method, episodes=500, alpha=0.5, eps=0.1, gamma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((CLIFF_ROWS, CLIFF_COLS, 4))
    sums = []
    for _ in range(episodes):
        s, total = CLIFF_START, 0.0
        a = eps_greedy(Q, s, eps, rng)
        for _ in range(10000):
            ns, r, done = cliff_step(s, a)
            total += r
            na = eps_greedy(Q, ns, eps, rng)
            if method == "sarsa":
                target = r + (0 if done else gamma * Q[ns][na])  # 用真正要做的下一个动作
            else:
                target = r + (0 if done else gamma * Q[ns].max())  # 用下一格最好的动作
            Q[s][a] += alpha * (target - Q[s][a])
            s, a = ns, na
            if done:
                break
        sums.append(total)
    return Q, np.array(sums)


def greedy_path(Q):
    s, path = CLIFF_START, [CLIFF_START]
    for _ in range(60):
        s, _, done = cliff_step(s, int(np.argmax(Q[s])))
        path.append(s)
        if done:
            break
    return path


def main():
    Q = np.zeros((CLIFF_ROWS, CLIFF_COLS, 4))
    Q[2, 0] = [-5, -12, -4, -6]
    s, a, r, ns, na = (3, 0), 0, -1.0, (2, 0), 0
    sarsa_t = r + Q[ns][na]
    q_t = r + Q[ns].max()
    print(f"手算一步：SARSA 目标 = {sarsa_t}（下一步真做 ↑），Q-learning 目标 = {q_t}（下一格最好 →）")
    for m in ["sarsa", "q"]:
        res = [run(m, seed=k) for k in range(30)]
        tail = np.mean([r[1][-100:].mean() for r in res])
        rows = {p[0] for p in greedy_path(res[0][0])[1:-1]}
        print(f"{m}: 最后 100 局平均每局奖励 {tail:.1f}；贪心路线经过的行 {sorted(rows)}")
    print("动作编号：", MOVE_NAMES, "终点", CLIFF_GOAL)


if __name__ == "__main__":
    main()
