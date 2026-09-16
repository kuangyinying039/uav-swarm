module.exports = [
`【中文】
各位老师、同学，大家好，我是来自华中科技大学的匡银滢。今天汇报的题目是 A Heterogeneous Graph-Attention MAPPO Algorithm for Cooperative Multi-UAV Search under Intermittent Communication Constraints。
这项工作关注一个比较实际的问题：当无人机之间不能持续通信时，每架无人机只能依赖自己的局部观测和当前能够收到的邻居信息，那么团队应该如何继续进行有效的协同搜索。围绕这个问题，我们提出了 IC-HGAT-MAPPO。下面我从问题背景、方法设计和实验结果三个方面介绍这项工作。
【English】
Good morning. I am Yinying Kuang from Huazhong University of Science and Technology. Today I will present our work entitled “A Heterogeneous Graph-Attention MAPPO Algorithm for Cooperative Multi-UAV Search under Intermittent Communication Constraints.”
We focus on a practical question: when inter-UAV communication is intermittent, each UAV can only rely on its onboard observations and currently reachable peer information. How can the team still coordinate an effective search? To address this problem, we propose IC-HGAT-MAPPO. I will introduce the motivation, the proposed method, and the experimental results.`,

`【中文】
今天的汇报分为三部分。首先介绍应用背景以及间歇通信给多无人机搜索带来的核心问题；然后介绍 IC-HGAT-MAPPO，包括局部信念维护、通信历史感知的信息选择和异构图策略；最后给出对比实验、消融实验和敏感性分析，并总结主要结论。
【English】
The presentation has three parts. I will first introduce the motivation and the main challenges caused by intermittent communication. I will then present IC-HGAT-MAPPO, including local belief maintenance, communication-history-aware information selection, and the heterogeneous graph policy. Finally, I will discuss the comparative, ablation, and sensitivity results before concluding the talk.`,

`【中文】
多无人机协同搜索可以应用于震后搜救、林火监测和海上救援等任务。相比单架无人机，多机系统能够并行覆盖更大的区域，但这种优势通常依赖于有效的信息协同。
实际环境中，通信链路可能因为距离、干扰或遮挡而中断，因此无人机无法持续获得同步的全局信息。我们关注的正是这种情况下的决策问题：每架无人机只保留自己的局部信念，并利用当前能够接收到的邻居信息完成协同搜索。由此带来的关键问题，是如何维护局部信念、如何判断邻居信息是否仍然有用，以及如何同时处理不同类型的交互关系。
【English】
Cooperative multi-UAV search is relevant to applications such as earthquake response, wildfire monitoring, and maritime rescue. A UAV team can cover a larger area in parallel, but this advantage depends strongly on effective information sharing.
In practice, communication can be interrupted by range limitations, interference, or occlusion, so synchronized global information is not continuously available. We therefore study decentralized search in which each UAV maintains its own local belief and only uses information from currently reachable peers. This leads to three questions: how to maintain useful local beliefs, how to judge the usefulness of peer information, and how to represent heterogeneous interactions.`,

`【中文】
我们把这个问题归纳为三个挑战。
第一是局部信息不完整。链路状态随时间变化，无人机无法依赖一个持续同步的全局地图。
第二是信息陈旧。一个邻居重新连上并不意味着它保存的信息仍然足够新，因此“能够通信”和“值得采用”是两个不同的问题。
第三是交互关系异构。同伴无人机提供协作信息，目标或前沿代表搜索收益，而障碍物主要体现安全约束，这些关系不适合用完全相同的方式处理。
因此，我们关注的是通信受限条件下如何做决策，而不是优化通信协议本身。
【English】
We summarize the problem into three challenges.
The first is incomplete local information. Because connectivity changes over time, the UAVs cannot rely on a continuously synchronized global map.
The second is information staleness. A peer becoming reachable again does not imply that its stored belief is still fresh, so reachability and usefulness are different concepts.
The third is heterogeneous interaction. Peers provide cooperative information, targets or frontiers represent search utility, and obstacles mainly represent safety constraints. These relations should not be treated identically.
Our focus is therefore decision-making under communication constraints rather than communication-protocol optimization.`,

`【中文】
针对这三个问题，我们设计了三个对应模块。
首先，使用 revisit-aware 的 P/E/R 记忆维护每架无人机自己的局部搜索状态，其中 P 表示目标存在概率，E 表示不确定性，R 表示区域多久没有被重新观测。
其次，在通信恢复后，并不是直接融合所有邻居信息，而是结合距离和通信历史，对每个栅格选择一个更合适的信息源。
最后，我们利用异构图注意力显式区分同伴、任务和障碍物关系。整个策略采用 MAPPO 的集中训练、分散执行框架：训练阶段 Critic 可以使用全局状态，而实际执行时 Actor 只使用本地可获得的信息。
【English】
We address these three challenges with three corresponding modules.
First, each UAV maintains revisit-aware P/E/R memory, where P represents target-existence belief, E represents uncertainty, and R represents revisit staleness.
Second, when communication becomes available, we do not simply combine all peer beliefs. Instead, distance and communication history are used to select an appropriate information source for each cell.
Third, heterogeneous graph attention explicitly distinguishes peer, task, and obstacle relations. The overall policy follows centralized training with decentralized execution: the critic can use global state during training, while the actor uses only locally available information during execution.`,

`【中文】
这里给出具体的实验场景。系统包含 6 架无人机、15 个运动目标和 10 个动态障碍物，环境被划分为 30×30 个栅格。无人机采用八方向移动加悬停的离散动作，决策间隔在 0.45 到 1 秒之间根据环境状态自适应变化。
通信只允许当前满足距离和链路状态约束的邻居进入策略，因此执行阶段不存在默认的全局信息。
评价方面，我们分别考虑目标发现率 D_T、信念决断度 C_T 和基于真实目标位置的 Brier score。前两个指标描述“发现了多少”和“信念有多明确”，BBS 则进一步判断这种信念是否正确。此外，T_80 同时用决策步数和真实物理时间表示搜索效率。
【English】
This slide summarizes the search environment. We use six UAVs, fifteen moving targets, and ten dynamic obstacles on a 30×30 grid. Each UAV selects one of eight motion directions or hovering, and the decision interval varies adaptively between 0.45 and 1 second.
Only peers satisfying the instantaneous communication constraints are available to the policy, so execution does not assume global information.
For evaluation, we report target discovery D_T, belief decisiveness C_T, and a ground-truth-based balanced Brier score. Discovery measures how many targets are found, decisiveness measures how clearly the belief map separates confident cells, and BBS evaluates whether those beliefs are correct. Search efficiency is measured by T_80 in both decision steps and physical time.`,

`【中文】
这一页给出完整的数据流。执行阶段首先获得局部观测和当前链路状态，并更新本地 P/E/R 记忆。随后根据通信历史和信息新鲜度进行信息源选择，再构建当前 UAV 所能看到的异构图。
HGAT 编码后的特征进入共享 Actor，得到九个名义动作概率，最后通过所有方法共用的安全投影得到实际执行动作。
训练阶段额外使用一个集中式 Critic，并通过 MAPPO 更新共享策略。Critic 只存在于训练阶段，因此实际部署仍然是分散执行。
【English】
This slide shows the complete information flow. During execution, each UAV first receives its local observation and the current link state and updates its local P/E/R memory. Communication history and belief freshness are then used for source selection before constructing the local heterogeneous graph.
The HGAT embedding is passed to the shared actor, which produces probabilities over nine nominal actions. A common safety projection then gives the executed action.
During training, a centralized critic is additionally used for the MAPPO update. The critic is removed during execution, so deployment remains decentralized.`,

`【中文】
先看左侧的局部信念维护。只有被实际观测到的栅格才更新 log-odds；没有被观测的区域不会凭空获得新的观测证据。与此同时，R 通道会随时间增长，并向相邻栅格平滑传播，而重新观测到的区域会被重置为零。因此，策略能够区分“刚刚确认的信息”和“虽然很确定但已经很久没有复查的信息”。
右侧解决的是通信恢复后的信息选择问题。g_{j,t} 表示节点 j 的整体 belief-age 历史摘要。对于当前可达的邻居，我们综合距离、信息历史以及 belief 强度，对每个栅格选择一个来源，而不是直接把多个相关的 log-odds 相加。这样可以降低旧信息或重复信息对决策的影响。
【English】
The left side shows local belief maintenance. Log-odds are updated only for actually sensed cells; unsensed cells do not receive artificial measurement evidence. At the same time, the revisit channel ages over time and diffuses locally, while newly sensed cells are reset to zero. This allows the policy to distinguish recently confirmed information from old but still decisive beliefs.
The right side addresses source selection after communication recovery. g_{j,t} is a node-level summary of belief age and communication history. Among currently reachable peers, we combine spatial relevance, history, and belief strength to select one source for each cell instead of directly adding correlated log-odds.`,

`【中文】
局部信念最终需要转换成动作。我们把 UAV、任务实体和障碍物建模为不同类型的关系。对于同伴关系，注意力主要反映协作与通信信息；对于任务关系，重点是搜索收益；对于障碍关系，则更多体现局部安全和可行性。
注意力中除了标准的 query-key 相似度之外，还加入了关系置信度、距离先验以及瞬时链路掩码，因此断开的同伴不会进入注意力归一化。
得到融合表示后，共享 Actor 输出动作。训练采用 MAPPO，团队奖励由发现增量、信念决断度增量和安全代价共同构成。这里的重点不是 PPO 公式本身，而是 HGAT 提供的局部表示如何进入 CTDE 学习框架。
【English】
The maintained beliefs must ultimately be converted into actions. We model peers, task entities, and obstacles as different relation types. Peer relations mainly represent cooperation and exchanged information, task relations represent search utility, and obstacle relations represent local safety constraints.
In addition to standard query-key similarity, the attention score includes relation confidence, a distance prior, and the instantaneous link mask, so disconnected peers are excluded from attention normalization.
The fused embedding is then passed to the shared actor. MAPPO is used for training, with a team reward combining discovery improvement, belief-decisiveness improvement, and a safety cost. The key point is how HGAT provides the local representation used within the CTDE framework.`,

`【中文】
所有学习方法采用相同的环境、动作空间、奖励和训练预算，并在 30 个未参与训练的场景上进行评估。
IC-HGAT-MAPPO 的信念决断度达到 62.15%，目标发现率达到 86.78%。相比启发式方法，分别提高 17.16 和 9.00 个百分点；相比 QMIX，则提高 32.15 和 14.67 个百分点。
如果采用最新的 ground-truth 指标，完整方法的终端 BBS 为 0.112，说明更高的决断度并不是以错误信念为代价获得的。达到 80% 目标发现的成功率为 82.0%，成功回合平均需要 88.63 个决策步，对应大约 63.80 秒。安全投影对所有方法相同，因此我们不把碰撞率作为本文的主要性能优势。
【English】
All learned methods use the same environment, action space, reward, and training budget and are evaluated on thirty held-out scenarios.
IC-HGAT-MAPPO achieves 62.15% belief decisiveness and 86.78% target discovery. Relative to the heuristic baseline, these are improvements of 17.16 and 9.00 percentage points; relative to QMIX, the improvements are 32.15 and 14.67 points.
With the ground-truth-based metric, the full method reaches a terminal BBS of 0.112, showing that increased decisiveness is not obtained at the cost of incorrect beliefs. The 80% discovery threshold is reached in 82.0% of evaluations, requiring 88.63 decision steps, or about 63.80 seconds, conditional on success. The safety projection is shared by all methods, so collision performance is not claimed as the main advantage.`,

`【中文】
消融结果进一步说明不同模块承担了不同作用。相对于同构图 MAPPO，完整模型的决断度提高 15.10 个百分点，说明异构关系建模能够提供更有效的策略表示。
去掉 revisit memory 后，决断度下降 16.33 个百分点，发现率下降 13.67 个百分点，是最明显的性能退化之一。去掉通信历史加权后，决断度下降 8.16 个百分点，同时条件 T_80 从 88.63 步增加到 98.19 步；但最终 discovery 的变化较小，而且统计上并不显著，因此我们主要把该模块的作用解释为改善 belief quality 和成功回合中的搜索效率。
下方敏感性实验表明，在目标速度、通信半径以及中断恢复等条件变化时，IC-HGAT-MAPPO 仍然保持较高的平均发现率。
【English】
The ablations show that the three components play different roles. Relative to Homogeneous Graph MAPPO, the full model gains 15.10 percentage points in decisiveness, supporting the value of heterogeneous relational modeling.
Removing revisit memory reduces decisiveness by 16.33 points and discovery by 13.67 points, producing one of the largest degradations. Removing communication-history weighting reduces decisiveness by 8.16 points and increases conditional T_80 from 88.63 to 98.19 steps. However, its effect on final discovery is small and statistically insignificant, so we mainly interpret this module as improving belief quality and successful-run efficiency.
The sensitivity results further show that IC-HGAT-MAPPO maintains comparatively high mean discovery as target motion, communication range, and outage-recovery conditions vary.`,

`【中文】
最后总结一下。
第一，我们把间歇通信作为策略可获得信息的约束，使无人机在执行阶段只依赖本地观测和当前可达的邻居信息。
第二，通过 revisit-aware 的 P/E/R 记忆和通信历史感知的信息源选择，策略能够处理不完整和陈旧的局部信念。
第三，通过异构图注意力显式建模同伴、任务和障碍物关系，并在 MAPPO 的 CTDE 框架下完成学习。
实验中，完整模型取得了 62.15% 的信念决断度、86.78% 的目标发现率，并获得更低的 ground-truth belief error。后续我们将重点考虑面向运动目标的 belief prediction 与 calibration、三维搜索环境，以及带真实通信链路的半实物验证。
【English】
To summarize, our work has three main points.
First, intermittent communication is modeled as a constraint on policy information, so execution relies only on onboard observations and currently reachable peers.
Second, revisit-aware P/E/R memory and communication-history-aware source selection allow the policy to handle incomplete and stale local beliefs.
Third, heterogeneous graph attention explicitly models peer, task, and obstacle relations within a MAPPO-based CTDE framework.
The full model achieves 62.15% belief decisiveness and 86.78% target discovery while also reducing ground-truth belief error. Future work will focus on motion-aware belief prediction and calibration, three-dimensional search, and hardware-in-the-loop validation with realistic asynchronous communication.`,

`【中文】
我的汇报到这里，谢谢各位老师和同学的聆听。我是匡银滢，欢迎各位老师批评指正，也欢迎提问。
【English】
That concludes my presentation. Thank you very much for your attention. I am Yinying Kuang, and I would be happy to answer your questions.`,
];
