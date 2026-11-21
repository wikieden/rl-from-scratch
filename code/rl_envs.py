"""讲义里反复用到的几个小环境：扫地机器人（第 5–9 章）、网格世界（第 8–12 章）、
随机游走（第 14–15 章）、悬崖（第 16 章）、走廊（第 17–18 章）。

动力学统一写成 P[s][a] = [(概率, 下一个状态, 奖励, 是否结束), ...]，
也就是 Sutton & Barto 式 3.2 的 p(s', r | s, a)。
"""

import numpy as np

# ---------- 扫地机器人（S&B Example 3.3 回收机器人，参数是讲义自己定的）----------
ALPHA, BETA = 0.8, 0.7  # 高电量搜索后仍是高电量的概率；低电量搜索后仍是低电量的概率
R_SEARCH, R_WAIT, R_RESCUE = 2.0, 1.0, -3.0
ROBOT_STATES = ["高电", "低电"]
ROBOT_ACTIONS = ["搜索", "等待", "充电"]


def robot_mdp():
    """返回 P[s][a]；状态 0 = 高电、1 = 低电；动作 0 = 搜索、1 = 等待、2 = 充电（只有低电能充电）。"""
    hi, lo = 0, 1
    P = {
        hi: {
            0: [(ALPHA, hi, R_SEARCH, False), (1 - ALPHA, lo, R_SEARCH, False)],
            1: [(1.0, hi, R_WAIT, False)],
        },
        lo: {
            0: [(BETA, lo, R_SEARCH, False), (1 - BETA, hi, R_RESCUE, False)],
            1: [(1.0, lo, R_WAIT, False)],
            2: [(1.0, hi, 0.0, False)],
        },
    }
    return P


ROBOT_PI = {0: {0: 0.9, 1: 0.1}, 1: {0: 0.2, 1: 0.3, 2: 0.5}}  # 第 6 章的策略 π(a|s)


# ---------- 网格世界 ----------
MOVES = [(-1, 0), (1, 0), (0, 1), (0, -1)]  # 北、南、东、西
MOVE_NAMES = ["↑", "↓", "→", "←"]


def gridworld_5x5():
    """S&B Example 3.5（图 3.2）：5×5，A=(0,1) 任何动作 +10 跳到 A'=(4,1)，B=(0,3) +5 跳到 B'=(2,3)，
    撞墙原地不动 −1，其余 0。连续任务，不结束。状态编号 = 行 × 5 + 列。"""
    n = 5
    P = {}
    for r in range(n):
        for c in range(n):
            s = r * n + c
            P[s] = {}
            for a, (dr, dc) in enumerate(MOVES):
                if (r, c) == (0, 1):
                    P[s][a] = [(1.0, 4 * n + 1, 10.0, False)]
                elif (r, c) == (0, 3):
                    P[s][a] = [(1.0, 2 * n + 3, 5.0, False)]
                else:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n:
                        P[s][a] = [(1.0, nr * n + nc, 0.0, False)]
                    else:
                        P[s][a] = [(1.0, s, -1.0, False)]
    return P


def gridworld_4x4():
    """S&B Example 4.1：4×4，左上 (0) 和右下 (15) 是终点，每走一步 −1，撞墙原地不动。"""
    n = 4
    P = {}
    for s in range(n * n):
        r, c = divmod(s, n)
        P[s] = {}
        for a, (dr, dc) in enumerate(MOVES):
            if s in (0, 15):
                P[s][a] = [(1.0, s, 0.0, True)]
                continue
            nr, nc = r + dr, c + dc
            ns = nr * n + nc if (0 <= nr < n and 0 <= nc < n) else s
            P[s][a] = [(1.0, ns, -1.0, ns in (0, 15))]
    return P


def uniform_policy(P):
    return {s: {a: 1.0 / len(P[s]) for a in P[s]} for s in P}


# ---------- 随机游走（S&B Example 6.2）----------
WALK_NAMES = ["A", "B", "C", "D", "E"]
WALK_TRUE = np.array([1, 2, 3, 4, 5]) / 6  # 真实价值 1/6 … 5/6


def random_walk_episode(rng):
    """从 C 出发，每步等概率往左或往右；走出左边奖励 0，走出右边奖励 +1。
    返回 [(状态, 奖励, 下一个状态或 None)]，状态 0–4 对应 A–E。"""
    s, steps = 2, []
    while True:
        ns = s + (1 if rng.random() < 0.5 else -1)
        if ns < 0:
            steps.append((s, 0.0, None))
            return steps
        if ns > 4:
            steps.append((s, 1.0, None))
            return steps
        steps.append((s, 0.0, ns))
        s = ns


# ---------- 悬崖（S&B Example 6.6）----------
CLIFF_ROWS, CLIFF_COLS = 4, 12
CLIFF_START, CLIFF_GOAL = (3, 0), (3, 11)


def cliff_step(state, a):
    """每步 −1；掉下悬崖（最下一行中间 10 格）−100 并回到起点；到达终点结束。"""
    r, c = state
    dr, dc = MOVES[a]
    nr, nc = min(max(r + dr, 0), CLIFF_ROWS - 1), min(max(c + dc, 0), CLIFF_COLS - 1)
    if nr == 3 and 1 <= nc <= 10:
        return CLIFF_START, -100.0, False
    return (nr, nc), -1.0, (nr, nc) == CLIFF_GOAL


# ---------- 走廊（第 17–18 章）----------
CORRIDOR_N = 6  # 格子 0–5，从 0 出发，走到 5 拿 +1 结束；动作 0 = 左、1 = 右


def corridor_step(s, a):
    ns = max(s - 1, 0) if a == 0 else s + 1
    if ns == CORRIDOR_N - 1:
        return ns, 1.0, True
    return ns, 0.0, False
