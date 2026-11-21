"""第 2 章 · 奖励与回报：G_t = R_{t+1} + R_{t+2} + ... + R_T（S&B 式 3.7）。"""


def episode_return(rewards, t=0):
    """rewards[0] 是 R_1。G_t 从 R_{t+1} 加到最后。"""
    return sum(rewards[t:])


def all_returns(rewards):
    return [episode_return(rewards, t) for t in range(len(rewards) + 1)]  # 最后一个是 G_T = 0


def main():
    dog = [1, 0, 1, 0]  # 小狗 4 步的奖励 R1..R4
    print("小狗 G0..G4 =", all_returns(dog))
    fast, slow = [-1] * 5, [-1] * 8  # 迷宫：每走一步 −1
    print("迷宫 5 步出去 G0 =", episode_return(fast), "；8 步出去 G0 =", episode_return(slow))


if __name__ == "__main__":
    main()
