# Nature 系列期刊导向的论文修订路线（2026-10-19）

## 定位与当前状态

正文 `scis_pursuit_draft.tex` 已按“研究问题—可检验结果—解释—可复现方法”重排；`SCIS2026` 类只是当前本地排版容器，不代表已经符合任何 Nature 子刊的最终投稿样式。用户所说的“Nature 系列”尚未指定具体刊物。Nature Communications 要求本领域的重要进展；Nature Machine Intelligence 覆盖机器人、强化学习和多智能体，但通常需要可推广的机器智能进展；Scientific Reports 更强调严谨、技术可靠的原始研究。**当前尚无正式双机物理捕获结果，不能判断稿件达到哪一刊的编辑门槛。**

从现有实现看，最稳妥的核心命题是：**当目标轨迹因遮挡而间歇失效、成功又取决于两架物理机体同时到位时，专家引导学习能否在同一执行与感知接口下形成可靠的协同捕获？** 论文应围绕此命题给出可复现证据；“首次提出图 MARL”“优于 MPC”“真实环境鲁棒”目前均没有足够依据。

Nature 官方材料： [Nature 初投与正文结构](https://www.nature.com/nature/for-authors/initial-submission)、[Nature 格式指南](https://www.nature.com/nature/for-authors/formatting-guide)、[Nature Communications 投稿指南](https://www.nature.com/ncomms/submit/how-to-submit)、[Nature Communications 范围](https://www.nature.com/ncomms/ncomms/aims)、[Nature Machine Intelligence 范围](https://www.nature.com/natmachintell/aims)、[Scientific Reports 编辑原则](https://www.nature.com/srep/about/editorial-process)。具体字数、图数、声明和版式应在选刊后以该刊最新指南核对；勿直接把 Nature 主刊限制套到全部子刊。

## 已落实的源码到正文替换

| 旧叙事或易误写之处 | 当前正文应保持的事实 | 主要依据 |
|---|---|---|
| Medium 训练、四档迁移 | Hard 训练；Nominal/Medium/Hard/Severe/Extreme 五档，冻结同一种子的 `final.pt` | `repro/run_isaac_dual_suite.sh` |
| A5、A0–A4c 是本轮主方法/消融 | MAPPO+ 与 A6 是两条主要学习路线；MAPPO-DAgger 是轻量对照；旧消融入口现在只跳过 | 同上；`repro/train_isaac_pursuit.py` |
| A6 在线混合 MPC 动作 | 默认执行 actor + 探索噪声；教师用于回放/BC/锚定；动作混合需要显式 `--residual-base` | `repro/train_isaac_pursuit.py` |
| 将 `best_capture.pt` 写成正式测试权重 | suite 评测加载 `final.pt`；验证最佳权重仅作单独诊断 | `repro/run_isaac_dual_suite.sh` |
| 四个 PhysX 飞行器或接触式捕获 | 3 架 Hummingbird 在 PhysX 中执行；逃逸目标是规则运动学球体；捕获按两架机体到目标的中心距离判定 | `repro/isaac_pursuit_backend.py` 与 `repro/quadrotor_pursuit_env.py` |
| 原生点云雷达、弱通信 | 默认几何雷达代理与全连接通信；原生 LiDAR 路径另行启用 | `repro/configs/pursuit_v2/*.json` 与环境实现 |
| 物理碰撞/安全干预统计可直接做结论 | 现有碰撞统计是几何近距代理；安全修正率在当前评价输出中需验证，不能直接当已测物理量 | `repro/evaluate_isaac_pursuit.py` 与后端输出 |
| 可见度势与全部安全系数都进入物理训练回报 | bridge 直接调用基础 pursuit step，绕过 quadrotor step 的可见度/近距/修正/平滑度奖励；论文公式必须按 bridge 基础回报、物理捕获补偿和刚体间隙势写 | `repro/isaac_pursuit_backend.py`；外部 Radar5 `isaac_pursuit_observation.py` |
| 四种难度变化可解释单一因素效应 | 五档同时改目标速度、起始距离、建筑和感知；需要单因素对照才可作机制归因 | 五档配置 JSON |

这张表描述正文必须遵守的边界，不能替代对正式运行工件的检查。外部 OmniDrones/Isaac 安装和 Radar5 桥接源不在该 Git 提交内，本机无法完成全链路物理执行核验。

## 成稿叙事：按证据组织，而非按算法名堆叠

1. **标题和摘要。** 标题点出遮挡、协同双机捕获、物理仿真即可；不要提前称高效、鲁棒或 state of the art。摘要用一段说明背景缺口、方法、共同测试、一个主效应及不确定度、结论边界。当前 `[TBD]` 是有意留白，正式投稿前必须换成真实数值和结论。
2. **引言。** 首段说明双机同时捕获相对单机追踪的科学难点；次段说明传统制导、局部观测 MARL 和专家引导各解决什么、尚缺共同物理测试；末段明确假设与贡献，不列空泛“首次”。相关工作只引用经过核对的原始研究。
3. **结果。** 优先呈现五档捕获率主图/表，给出每训练种子的结果与跨种子不确定度；再呈现 capped steps、迁移变化、预算成本。每段第一句回答一个具体问题，随后给出效应大小、区间与参照，不以训练回报替代物理成功。定性轨迹只能说明选定实例，需写明挑选准则与种子。
4. **机制证据。** 如果要宣称图结构、CI 新鲜度、可见度奖励或专家数据各自有效，补“仅改变该因素”的对照。当前主 suite 没有这些消融；没有结果就把机制语言保留为设计动机，不写成已证实。
5. **讨论。** 概括主发现及适用边界，解释 MAPPO+ 与 A6 的不同训练成本和偏差，承认脚本化目标、几何感知、全连接通信与中心距离成功判定。不要把仿真结果外推为真实追捕安全性。
6. **方法。** 正文保留影响结果解释的模型、捕获判据、观测/动作、训练与评估规则、统计单位。教师候选评分、所有超参数和详细方程可在目标刊允许时移到补充信息或扩展数据，正文交叉引用；不能为了压字数删掉复现关键参数。
7. **投稿声明。** 代码固定提交已写入正文。数据声明待正式逐局输出、配置、checkpoint 哈希和可访问归档地址；正式投稿前补完整。外部依赖版本、运行硬件和总成本也需明确。作者贡献、经费与致谢、竞争利益、图像授权及生成式 AI 使用披露应按最终目标刊的最新要求由作者填写。

## 建议的图表与最小证据集

| 图表 | 应回答的问题 | 数据及标注 |
|---|---|---|
| 图 1：任务/感知示意 | 遮挡和“两机同时进入 1 m 球”是什么？ | 明示示意图、几何尺度、目标为运动学球体 |
| 图 2：闭环与学习路线 | actor、估计、控制、教师数据如何连接？ | 教师只作训练数据/标签；A6 默认 actor 执行 |
| 主结果：五档捕获率 | MAPPO+、A6、MAPPO-DAgger 与规则基线相比如何？ | 每方法每档共同 200 场景；学习法每 seed 单独汇总，再跨 seed 报区间；清楚标 `final.pt` |
| 次结果：capped steps 与预算 | 成功率外的时间代价、在线采样与专家采样代价如何？ | 失败局计到 300 步；预算曲线只放实际快照，标明 Medium/50 |
| 机制/局限实验 | 哪个模块造成观察到的效应？ | 按主张设计受控消融；若未跑，不放虚构面板 |
| 代表性轨迹 | 在遮挡情形下策略怎样到达双机捕获？ | 预先说明选择规则、种子、目标/机体/可见性时间序列 |

当前结果表仍为空，历史单机或 Medium 试点数据保持在原归档。主图不能用这些旧数据替代。

引言中已将受限视野的搜索/探测文献与双机实体捕获区别开；参考文献已按首次引用顺序排列，删除未引用或不切题的旧条目，并补 OmniDrones、Lee 控制器、异构图注意力及 TD3+BC 的原始来源。MID-360 是 LiDAR，正文现称“几何 LiDAR 式目标量测代理”，以免把内部 Radar5 命名误当成物理雷达或原生点云。

**最高优先级实现核查：** 当前 bridge 可能在二维代理满足捕获时把内部目标标为 captured，随后停下脚本目标并改变 timeout 奖励；Isaac 外层虽重写返回的物理捕获标志，未同步清除该内部状态。正式数据进入主表前，须量化逐局代理/刚体捕获分叉，修正状态一致性并重训/重评受影响工件。现有训练结果保留为历史诊断，不能掩盖这条路径。此问题也使“可见度 shaping 改善捕获”的主张无依据，因为当前物理训练未应用该奖励项。

## 正式结果入稿前的核对

- 逐局表应能追溯：Git 提交、环境配置、方法、训练 seed、测试 seed、checkpoint SHA-256、测试档位、终止原因、是否两机同时到位。重新运行或更换 checkpoint 时用独立输出目录；评价缓存不会可靠核查工件身份。
- A6 的先验文件虽采集 200 局，success-only replay 只保留成功局 transitions；图表需记录实际成功局数与加载条数。物理数值失败会早停，当前 `mean_censored_steps` 直接平均停止步；正式 capped-steps 要从逐局数据按预定义规则重算，单列早停数量。
- 对每个训练 seed 先算 200 场景指标，再做尊重训练 seed 层级和测试场景配对的区间估计；仅一个 seed 完成时就标“单种子探索性结果”。不要把 600 局视为 600 个独立训练重复。
- 报捕获率与差值的绝对值和区间；若多档/多方法比较很多，应明确主要比较，避免事后挑最有利结果。报告未完成/失败训练和剔除规则。
- 预算比较同时计入教师示范、BC、DAgger、在线环境步、GPU 时长。A6 的成功局回放和 Q 上界裁剪应作为训练配方说明，不单凭曲线称“样本效率更高”。
- 五档属于联合难度移动。若要分辨遮挡、目标速度、探测率的贡献，需要额外单因素实验；同理，轨迹里的“角色分工”需量化指标支持。
- 投稿前核实所有参考文献的题目、作者、年份、DOI，以及是否真的支持相邻句子。将图注写成独立可读：定义 n、误差条、缩写、量纲和统计检验。

## 本地交付与下一轮工作

- 主稿：`scis_pursuit_draft.tex`；编译 PDF：`scis_pursuit_draft.pdf`。
- 实验事实速查：`PAPER_HANDOFF_EXPERIMENT_SUMMARY.md`；可执行协议：`../repro/ISAAC_DUAL_EXPERIMENTS.md` 与 `../repro/isaac_dual_experiment_manifest.json`。
- 先拿到正式五档原始结果，再补主结果与摘要；随后依据实测效应决定投 Nature Communications、Nature Machine Intelligence、Scientific Reports 或更合适的领域刊，并转换相应版式。期刊选择应由结果的普适性、机制证据与技术完整度决定。

> 历史修订与交接记录（基于旧版协议），不是当前 `physx_closed` + MPC 的实验说明。现行稿件以 `scis_pursuit_draft.tex` 为准；正式结果、checkpoint 和代码提交号须按 closed 评测核实。
