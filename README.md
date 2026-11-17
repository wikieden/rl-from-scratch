# 从零学强化学习

**在线阅读：<https://wikieden.github.io/rl-from-scratch/>** · [讲义目录](https://wikieden.github.io/rl-from-scratch/lessons/index.html) · [全部动画](https://wikieden.github.io/rl-from-scratch/animations/index.html) · [复习卡片](https://wikieden.github.io/rl-from-scratch/review.html) · [全部讲义书](https://wikieden.github.io/ai-books/)

> 24 章 · 24 个动画（GIF + 可交互页面）· 42 张复习卡片 · 覆盖 Sutton & Barto 第 2–6、9、13 章与 CS 285 第 4–10、14 讲的主线

## 这本书讲什么

从"小狗看到主人举手、坐下、拿到零食"讲起，一步步走到今天训练机器人和大语言模型用的 PPO、GRPO：回报和折扣、马尔可夫决策过程、价值函数和 Bellman 方程、动态规划、蒙特卡洛和时序差分、Q-learning、DQN、策略梯度、Actor-Critic 与 GAE、TRPO / PPO，最后接到 LLM 训练里的 GRPO。

- **适合谁**：想看懂强化学习公式、能去写 CS 285 作业的人。数学只要会加减乘除；概率与期望（第 4 章）、导数和梯度（第 17 章）、log（第 20 章）都在用到的那一章现场补。看过《从零读懂 Transformer》的话，向量、点积、softmax、加权平均可以直接用上。
- **读完能做什么**：对照 Sutton & Barto 读懂第 3–6、13 章的公式；动手写 CS 285 HW2（策略梯度、baseline、GAE）和 HW3（DQN、Double DQN）；读 PPO、GRPO 论文的核心公式。
- **怎么读**：按章顺序读。每章先看讲解和动画，再自己做练习，最后用复习卡片隔几天考自己。

## 讲解的特点

- **一次一小步**：每个概念按"白话类比 → 公式（每个符号说清楚怎么读）→ 小数字亲手算一遍"讲，算例都能在纸上验证，而且全部用代码核对过（测试 22 个，全部通过）。
- **几个例子贯穿全书**：训练小狗（第 1 章）、扫地机器人（第 5–9 章，书里的回收机器人）、书上的 5×5 和 4×4 网格世界（第 8–12 章，数字和图 3.2、3.5、4.1 一模一样）、随机游走 A–E（第 14–15、17 章）、悬崖行走（第 16 章）、走廊（第 18 章）、两台老虎机（第 13、20–21 章）。
- **每章一个动画**：关键原理做成可交互页面，▶ 播放、→ 单步、↺ 从头开始，有滑块的可以拖动参数看结果怎么变；GIF 版直接嵌在正文里。
- **对照原文**：讲到教材和 slides 的地方都标了本地文本的行号（如 S&B L4309–L4344），每章末尾的延伸阅读精确到章节、式号和行号。
- **术语不设门槛**：缩写第一次出现就给英文全称并逐词解释（如 TD = Temporal Difference：时间上的 / 差分），全书缩写见 [缩写与术语表](https://wikieden.github.io/rl-from-scratch/lessons/appendix-glossary.html)。
- **每章都有代码对照**：附录最后一节把本章的数字用 NumPy / PyTorch 跑一遍（Mac 上不需要 GPU），配测试核对和讲义一致，顺带讲代码里才看得到的坑（下标差 1、`1 − 0.7` 不等于 0.3、`torch.no_grad()`、`(1 − done)`……）；只算讲过的例子，不提前给练习答案。代码在 `solutions/chapter-code/`。每章还配了一个 Jupyter notebook（`solutions/chapter-code/notebooks/`），讲解和代码放在一起，一格一格边读边跑。
- **每章收尾有口诀**：几句话记住一章，全部口诀汇总在本页末尾。

## 示例动画

<table>
<tr>
<td width="33%" valign="top"><img src="https://wikieden.github.io/rl-from-scratch/animations/l08-bellman.gif" width="100%" alt="第 8 章 · Bellman 期望方程"><br><b>第 8 章 · Bellman 期望方程</b><br>每格的价值 = 四个邻居打折的平均 + 奖励<br><a href="https://wikieden.github.io/rl-from-scratch/lessons/08-bellman.html">讲义</a> · <a href="https://wikieden.github.io/rl-from-scratch/animations/l08-bellman.html">▶ 交互</a></td>
<td width="33%" valign="top"><img src="https://wikieden.github.io/rl-from-scratch/animations/l16-sarsa-qlearning.gif" width="100%" alt="第 16 章 · SARSA 与 Q-learning"><br><b>第 16 章 · SARSA 与 Q-learning</b><br>悬崖边：一个绕远路，一个贴边走<br><a href="https://wikieden.github.io/rl-from-scratch/lessons/16-sarsa-qlearning.html">讲义</a> · <a href="https://wikieden.github.io/rl-from-scratch/animations/l16-sarsa-qlearning.html">▶ 交互</a></td>
<td width="33%" valign="top"><img src="https://wikieden.github.io/rl-from-scratch/animations/l23-ppo.gif" width="100%" alt="第 23 章 · PPO"><br><b>第 23 章 · PPO</b><br>比率涨过 1.2 就停：剪裁目标的形状<br><a href="https://wikieden.github.io/rl-from-scratch/lessons/23-ppo.html">讲义</a> · <a href="https://wikieden.github.io/rl-from-scratch/animations/l23-ppo.html">▶ 交互</a></td>
</tr>
</table>

全部动画见 [动画目录](https://wikieden.github.io/rl-from-scratch/animations/index.html)。

## 目录

每章都配一个动画：**GIF** 在讲义正文里直接播放，**▶ 交互**是可以点按钮、拖滑块的页面（本地看需要起服务，见 [动画目录](https://wikieden.github.io/rl-from-scratch/animations/index.html)"跑起来看"）。

| 章 | 主题 | 一句话精要 | 动画 | 卡片 | 状态 |
|---|---|---|---|---|---|
| [1](https://wikieden.github.io/rl-from-scratch/lessons/01-agent-environment.html) | 智能体与环境：交互循环、轨迹；和监督学习的区别；边界 | 看状态、做动作、拿奖励；奖励下标大一号；改不了的都是环境 | [GIF](https://wikieden.github.io/rl-from-scratch/animations/l01-agent-environment.gif) · [▶ 交互](https://wikieden.github.io/rl-from-scratch/animations/l01-agent-environment.html) | R01、R02 | 正文完成，练习待做 |
| [2](https://wikieden.github.io/rl-from-scratch/lessons/02-return.html) | 奖励与回报：$G_t$；分幕与持续；奖励说"要什么" | 目标是把奖励加起来最多；$G_t$ 从 $R_{t+1}$ 加到底 | [GIF](https://wikieden.github.io/rl-from-scratch/animations/l02-return.gif) · [▶ 交互](https://wikieden.github.io/rl-from-scratch/animations/l02-return.html) | R03、R04 | 正文完成，练习待做 |
| [3](https://wikieden.github.io/rl-from-scratch/lessons/03-discount.html) | 折扣 γ：几何级数；$G_t = R_{t+1} + \gamma G_{t+1}$ | 往后一步打一折；每步 +1 总数 1/(1−γ)；倒着推 | [GIF](https://wikieden.github.io/rl-from-scratch/animations/l03-discount.gif) · [▶ 交互](https://wikieden.github.io/rl-from-scratch/animations/l03-discount.html) | R05、R06 | 正文完成，练习待做 |
| [4](https://wikieden.github.io/rl-from-scratch/lessons/04-expectation.html) | 概率与期望：加权平均、大数定律、条件期望 | 期望 = 值 × 概率再相加；多试几次平均会靠近它 | [GIF](https://wikieden.github.io/rl-from-scratch/animations/l04-expectation.gif) · [▶ 交互](https://wikieden.github.io/rl-from-scratch/animations/l04-expectation.html) | R07 | 正文完成，练习待做 |
| [5](https://wikieden.github.io/rl-from-scratch/lessons/05-mdp.html) | MDP：动力学 $p(s', r \mid s, a)$；马尔可夫性质 | 环境是一张概率表；只看现在不翻历史 | [GIF](https://wikieden.github.io/rl-from-scratch/animations/l05-mdp.gif) · [▶ 交互](https://wikieden.github.io/rl-from-scratch/animations/l05-mdp.html) | R08、R09 | 正文完成，练习待做 |
| [6](https://wikieden.github.io/rl-from-scratch/lessons/06-policy.html) | 策略 $\pi(a \mid s)$：随机 / 确定性；RL 算法的三个部分 | 策略是行事手册；跑样本 → 评估 → 改策略 | [GIF](https://wikieden.github.io/rl-from-scratch/animations/l06-policy.gif) · [▶ 交互](https://wikieden.github.io/rl-from-scratch/animations/l06-policy.html) | R10 | 正文完成，练习待做 |
| [7](https://wikieden.github.io/rl-from-scratch/lessons/07-value.html) | 价值函数 $v_\pi$、$q_\pi$；两者的关系 | 价值是"房价"：从这里出发的平均回报；v 是 q 的加权平均 | [GIF](https://wikieden.github.io/rl-from-scratch/animations/l07-value.gif) · [▶ 交互](https://wikieden.github.io/rl-from-scratch/animations/l07-value.html) | R11、R12 | 正文完成，练习待做 |
| [8](https://wikieden.github.io/rl-from-scratch/lessons/08-bellman.html) | Bellman 期望方程；直接解方程；5×5 网格 | 今天 = 眼前奖励 + γ × 明天，取平均；A 格 8.8 | [GIF](https://wikieden.github.io/rl-from-scratch/animations/l08-bellman.gif) · [▶ 交互](https://wikieden.github.io/rl-from-scratch/animations/l08-bellman.html) | R13、R14 | 正文完成，练习待做 |
| [9](https://wikieden.github.io/rl-from-scratch/lessons/09-optimal.html) | 最优价值与 Bellman 最优方程；贪心即最优 | 平均换成 max；有 $q_*$ 直接挑；A 格 24.4 | [GIF](https://wikieden.github.io/rl-from-scratch/animations/l09-optimal.gif) · [▶ 交互](https://wikieden.github.io/rl-from-scratch/animations/l09-optimal.html) | R15、R16 | 正文完成，练习待做 |
| [10](https://wikieden.github.io/rl-from-scratch/lessons/10-policy-eval.html) | 策略评估：反复套期望方程；4×4 网格 | 旧值代右边、新值写左边，终点的 0 往外渗 | [GIF](https://wikieden.github.io/rl-from-scratch/animations/l10-policy-eval.gif) · [▶ 交互](https://wikieden.github.io/rl-from-scratch/animations/l10-policy-eval.html) | R17 | 正文完成，练习待做 |
| [11](https://wikieden.github.io/rl-from-scratch/lessons/11-policy-iteration.html) | 策略改进与策略迭代；改进定理 | 每格挑 q 最大，处处不输；策略不变就是最优 | [GIF](https://wikieden.github.io/rl-from-scratch/animations/l11-policy-iteration.gif) · [▶ 交互](https://wikieden.github.io/rl-from-scratch/animations/l11-policy-iteration.html) | R18、R19 | 正文完成，练习待做 |
| [12](https://wikieden.github.io/rl-from-scratch/lessons/12-value-iteration.html) | 价值迭代、异步 DP、广义策略迭代 GPI | 评估一轮就改进，合成一个 max；评估改进你推我拉 | [GIF](https://wikieden.github.io/rl-from-scratch/animations/l12-value-iteration.gif) · [▶ 交互](https://wikieden.github.io/rl-from-scratch/animations/l12-value-iteration.html) | R20、R21 | 正文完成，练习待做 |
| [13](https://wikieden.github.io/rl-from-scratch/lessons/13-bandit.html) | 探索与利用：多臂老虎机、ε-greedy、增量更新 | 九成贪心一成乱试；新 = 旧 + 步长 × (目标 − 旧) | [GIF](https://wikieden.github.io/rl-from-scratch/animations/l13-bandit.gif) · [▶ 交互](https://wikieden.github.io/rl-from-scratch/animations/l13-bandit.html) | R22、R23 | 正文完成，练习待做 |
| [14](https://wikieden.github.io/rl-from-scratch/lessons/14-monte-carlo.html) | 蒙特卡洛：first-visit、常数 α；随机游走 | 一局玩完，用真实回报的平均估价值；要等终局 | [GIF](https://wikieden.github.io/rl-from-scratch/animations/l14-monte-carlo.gif) · [▶ 交互](https://wikieden.github.io/rl-from-scratch/animations/l14-monte-carlo.html) | R24 | 正文完成，练习待做 |
| [15](https://wikieden.github.io/rl-from-scratch/lessons/15-td.html) | 时序差分 TD(0)：TD 误差、bootstrapping、TD vs MC | 走一步就用"下一步的估计"更新；方差小但有偏 | [GIF](https://wikieden.github.io/rl-from-scratch/animations/l15-td.gif) · [▶ 交互](https://wikieden.github.io/rl-from-scratch/animations/l15-td.html) | R25、R26 | 正文完成，练习待做 |
| [16](https://wikieden.github.io/rl-from-scratch/lessons/16-sarsa-qlearning.html) | SARSA 与 Q-learning；on / off-policy；悬崖行走 | 真要做的下一步 vs 下一格最好的；绕远路 vs 贴边走 | [GIF](https://wikieden.github.io/rl-from-scratch/animations/l16-sarsa-qlearning.gif) · [▶ 交互](https://wikieden.github.io/rl-from-scratch/animations/l16-sarsa-qlearning.html) | R27、R28 | 正文完成，练习待做 |
| [17](https://wikieden.github.io/rl-from-scratch/lessons/17-function-approx.html) | 函数近似与梯度下降：导数、SGD、半梯度 | 表格装不下就用函数猜；朝下坡走一小步 | [GIF](https://wikieden.github.io/rl-from-scratch/animations/l17-function-approx.gif) · [▶ 交互](https://wikieden.github.io/rl-from-scratch/animations/l17-function-approx.html) | R29、R30 | 正文完成，练习待做 |
| [18](https://wikieden.github.io/rl-from-scratch/lessons/18-dqn.html) | DQN：经验回放、目标网络、loss | Q 表换网络；随机抽旧经历；目标网络隔一阵对齐 | [GIF](https://wikieden.github.io/rl-from-scratch/animations/l18-dqn.gif) · [▶ 交互](https://wikieden.github.io/rl-from-scratch/animations/l18-dqn.html) | R31、R32 | 正文完成，练习待做 |
| [19](https://wikieden.github.io/rl-from-scratch/lessons/19-double-dqn.html) | 过估计与 Double DQN | max 挑到运气最好的；主网络挑、目标网络打分 | [GIF](https://wikieden.github.io/rl-from-scratch/animations/l19-double-dqn.gif) · [▶ 交互](https://wikieden.github.io/rl-from-scratch/animations/l19-double-dqn.html) | R33 | 正文完成，练习待做 |
| [20](https://wikieden.github.io/rl-from-scratch/lessons/20-policy-gradient.html) | 策略梯度与 REINFORCE：log 导数技巧 | $\nabla J = \mathbb{E}[G\,\nabla\log\pi]$：好的推高，坏的压低 | [GIF](https://wikieden.github.io/rl-from-scratch/animations/l20-policy-gradient.gif) · [▶ 交互](https://wikieden.github.io/rl-from-scratch/animations/l20-policy-gradient.html) | R34、R35 | 正文完成，练习待做 |
| [21](https://wikieden.github.io/rl-from-scratch/lessons/21-baseline.html) | baseline 与 reward-to-go：降方差 | 只看比平常好多少；动作只管身后事 | [GIF](https://wikieden.github.io/rl-from-scratch/animations/l21-baseline.gif) · [▶ 交互](https://wikieden.github.io/rl-from-scratch/animations/l21-baseline.html) | R36、R37 | 正文完成，练习待做 |
| [22](https://wikieden.github.io/rl-from-scratch/lessons/22-actor-critic-gae.html) | Actor-Critic、优势函数与 GAE | 演员演、评委打分；δ 当优势；λ 调偏差与方差 | [GIF](https://wikieden.github.io/rl-from-scratch/animations/l22-actor-critic-gae.gif) · [▶ 交互](https://wikieden.github.io/rl-from-scratch/animations/l22-actor-critic-gae.html) | R38、R39 | 正文完成，练习待做 |
| [23](https://wikieden.github.io/rl-from-scratch/lessons/23-ppo.html) | 从 TRPO 到 PPO：重要性采样、KL、剪裁 | 旧数据乘比率；每次别改太多；比率剪在 0.8–1.2 | [GIF](https://wikieden.github.io/rl-from-scratch/animations/l23-ppo.gif) · [▶ 交互](https://wikieden.github.io/rl-from-scratch/animations/l23-ppo.html) | R40、R41 | 正文完成，练习待做 |
| [24](https://wikieden.github.io/rl-from-scratch/lessons/24-wrap-up.html) | 收尾：全书地图、PPO → GRPO、KL 正则、下一步 | 同题答几遍互相比；拴住参考模型别学坏 | [GIF](https://wikieden.github.io/rl-from-scratch/animations/l24-wrap-up.gif) · [▶ 交互](https://wikieden.github.io/rl-from-scratch/animations/l24-wrap-up.html) | R42 | 正文完成，练习待做 |

附录：[缩写与术语表](https://wikieden.github.io/rl-from-scratch/lessons/appendix-glossary.html)（遇到缩写先查这一页；每章开头另有一行"本章缩写"）。

**状态说明**：24 章正文、动画、代码对照、notebook 都已写完；每章末尾的 3 道练习还没作答。作答时按仓库约定把原话、对错、点评记进各章"练习与作答记录"，卡住或答错的地方补成新的复习卡片。

---

本仓库由私有学习仓库里的脚本 `tools/publish-transformer/build.py` 自动生成，内容摘自讲义目录，不要直接修改这里的文件。
