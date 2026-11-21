"""第 1 章 · 智能体与环境：看状态 → 做动作 → 拿奖励和新状态，串成轨迹 S0, A0, R1, S1, ..."""


class DogEnv:
    """讲义的训练小狗：主人举手时坐下得 1 块零食，其余 0。主人举手 / 放手轮流出现。"""

    def __init__(self):
        self.state = "主人举手"

    def step(self, action):
        reward = 1 if (self.state == "主人举手" and action == "坐下") else 0
        self.state = "主人放下手" if self.state == "主人举手" else "主人举手"
        return reward, self.state


def rollout(env, actions):
    """按给定动作走几步，返回轨迹列表 [("S0", ...), ("A0", ...), ("R1", ...), ...]。"""
    traj = [("S0", env.state)]
    for t, a in enumerate(actions):
        traj.append((f"A{t}", a))
        r, s = env.step(a)
        traj.append((f"R{t + 1}", r))  # 奖励的下标比动作大 1
        traj.append((f"S{t + 1}", s))
    return traj


def main():
    traj = rollout(DogEnv(), ["坐下", "乱跑"])
    print("轨迹：", ", ".join(f"{k}={v}" for k, v in traj))


if __name__ == "__main__":
    main()
