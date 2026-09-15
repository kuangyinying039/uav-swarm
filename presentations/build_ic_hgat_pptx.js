#!/usr/bin/env node
/**
 * 13-page academic talk deck: IC-HGAT-MAPPO
 * Visual language follows the uploaded formation-control template
 * (HUST logo, CAA/CAC branding, blue title blocks, 13.333 x 7.5 in).
 */
const PptxGenJS = require("pptxgenjs");
const path = require("path");

const A = "/tmp/ppt-work/assets";
const P = path.join(A, "photos");

const BLUE = "2F5597";
const BLUE_DK = "1E3F73";
const BLUE_LT = "EAF0F8";
const ORANGE = "ED7D31";
const RED = "C0392B";
const GREEN = "2E8B57";
const DARK = "1F2A44";
const MUTED = "5A6A80";
const GRAY = "F4F6FA";
const WHITE = "FFFFFF";

const pres = new PptxGenJS();
pres.layout = "LAYOUT_WIDE";
pres.author = "Yinying Kuang";
pres.title = "IC-HGAT-MAPPO: Cooperative Multi-UAV Search under Intermittent Communication";
pres.subject = "CAC 2026";

function sh(opts) {
  return Object.assign({ type: "outer", color: "1F2A44", blur: 4, offset: 1, opacity: 0.12 }, opts);
}

const AR = {
  "hust.png": 688 / 475,
  "caa.png": 342 / 210,
  "chal1.png": 590 / 393,
  "chal2.png": 590 / 393,
  "chal3.png": 590 / 393,
  "fig1.png": 877 / 609,
  "fig3.png": 1055 / 575,
  "fig4.png": 2200 / 730,
  "framework.png": 2078 / 709,
  "hgat.png": 1024 / 648,
  "actor_critic.png": 900 / 586,
  "per_maps.png": 1631 / 569,
  "source.png": 1008 / 547,
  "photo_eq.png": 4.05 / 2.15,
  "photo_wf.png": 4.05 / 2.15,
  "photo_mr.png": 4.05 / 2.15,
};

function contain(slide, file, box) {
  const name = path.basename(file);
  const ar = AR[name] || 1.6;
  const boxAr = box.w / box.h;
  let w, h, x, y;
  if (ar > boxAr) {
    w = box.w;
    h = box.w / ar;
    x = box.x;
    y = box.y + (box.h - h) / 2;
  } else {
    h = box.h;
    w = box.h * ar;
    x = box.x + (box.w - w) / 2;
    y = box.y;
  }
  slide.addImage({ path: file, x, y, w, h });
}

function eqBox(slide, token, x, y, w, h, fs) {
  slide.addText(token, {
    x, y, w, h,
    fontFace: "Cambria Math", fontSize: fs || 18, color: DARK,
    align: "center", valign: "middle", margin: 0,
  });
}

function accent(slide, x, y, h) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w: 0.07, h, fill: { color: BLUE },
  });
}

function header(slide, section, subtitle) {
  contain(slide, path.join(A, "hust.png"), { x: 12.38, y: 0.06, w: 0.82, h: 0.54 });
  slide.addText(section, {
    x: 0.38, y: 0.08, w: 11.7, h: 0.48,
    fontFace: "Calibri", fontSize: 26, bold: true, color: BLUE, margin: 0, valign: "middle",
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0.62, w: 13.333, h: 0.52, fill: { color: BLUE },
  });
  slide.addText(subtitle, {
    x: 0.38, y: 0.64, w: 12.5, h: 0.48,
    fontFace: "Calibri", fontSize: 20, bold: true, color: WHITE, margin: 0, valign: "middle",
  });
}

function footer(slide, n) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 7.42, w: 13.333, h: 0.08, fill: { color: BLUE },
  });
  slide.addText("CAC 2026  ·  IC-HGAT-MAPPO  ·  Dec-POMDP / CTDE / HGAT / MAPPO", {
    x: 0.38, y: 7.10, w: 11.9, h: 0.28,
    fontFace: "Calibri", fontSize: 11, color: MUTED, margin: 0, valign: "middle",
  });
  slide.addText(String(n), {
    x: 12.40, y: 7.08, w: 0.70, h: 0.30,
    fontFace: "Calibri", fontSize: 13, bold: true, color: BLUE, align: "right", margin: 0,
  });
}

function card(slide, x, y, w, h, fill) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, fill: { color: fill }, rectRadius: 0.08,
    shadow: sh(),
  });
}

function notes(slide, text) {
  slide.addNotes(text);
}

const N = [
`【中文】
各位老师、同学，大家好。感谢主持人，也感谢各位在 CAC 2026 抽出时间听我们汇报。今天的工作面向间歇通信约束下的多无人机协同搜索。请先把问题记清楚：我们不把“全局地图可共享、链路始终连通”当作默认前提，而是把它建模成一个通信受限的 Dec-POMDP——每架无人机只能使用机载观测，以及当前物理上仍可达的邻居消息。算法名称 IC-HGAT-MAPPO 会在问题讲清楚之后再给出。接下来请各位先看目录，了解整场报告的三条主线。
【English】
Good morning, and thank you for joining this CAC 2026 session. This talk studies cooperative multi-UAV search under intermittent communication. Please keep the problem first: we do not assume a shared global map or always-on links. The setting is a communication-constrained Dec-POMDP, so each UAV may use only onboard observations and currently reachable peer messages. The name IC-HGAT-MAPPO comes after the problem is clear. Next, a short contents slide will mark the three parts of the talk.`,

`【中文】
报告严格对应论文结构，分成三部分，这里十几秒带过即可。第一部分是背景与问题：从应用场景走到间歇通信下的三个挑战，再映射到三个设计问题。第二部分是方法：问题形式化、整体框架、P/E/R 记忆与历史感知选源，以及异构图注意力加 MAPPO。第三部分是实验与结论：主结果、消融和敏感性，最后收束贡献与展望。下面请先进入第一部分，用三个典型任务说明：为什么多无人机搜索在通信不可靠时会变成一个决策问题。
【English】
The talk has three parts, matching the paper. Part I is background and the problem: applications, three communication-induced challenges, then three design questions. Part II is the method: formulation, the IC-HGAT-MAPPO pipeline, P/E/R memory with history-aware source selection, and heterogeneous graph attention with MAPPO. Part III reports experiments and conclusions. We now enter Part I, starting from three application scenes that make unreliable communication a decision problem, not only a networking problem.`,


`【中文】
多无人机可以并行覆盖震后搜救、林火监测和海上救援。这类任务的共同需求是：在动态障碍和运动目标下尽快提高发现率，同时把格子级信念推到更决断的状态。很多 MARL 搜索方法默认通信充分、信息接近全局，策略几乎可以当中心化控制器来训。但在干扰、遮挡或距离超限时，链路会间歇中断，可用邻居集合 𝒩_i(t) 随时间跳变。于是问题不再只是“怎么搜”，而是“信息不完整时如何协同搜”。请看下方三栏：左侧是常规设定，中间是更现实的设定，右侧三条研究需求——维持有用信念、利用可达同伴信息、建模异构关系——正好对应后文三个模块。下一页我们把这个现实设定拆成三个可以单独回答的挑战。
【English】
Multi-UAV teams can search in earthquake response, wildfire monitoring, and maritime rescue. The shared need is to raise discovery quickly while driving cell-wise beliefs toward more decisive states, under moving targets and dynamic obstacles. Many MARL search methods assume stable communication and nearly global information, so the policy almost trains as a centralized controller. Under interference, occlusion, or range limits, links drop and the reachable set 𝒩_i(t) jumps over time. The question is therefore not only how to search, but how to coordinate under incomplete information. The three columns map conventional, practical, and research needs onto the three modules that follow. Next we split that practical setting into three challenges we can answer one by one.`,

`【中文】
带着刚才的应用，间歇通信会落到三个核心挑战。第一，没有共享全局地图，观测是局部的，可用邻居随 A_{i,j,t} 变化，决策必须去中心化。第二，同伴、任务或前沿、障碍物角色不同：同伴提供协作信息，任务节点编码搜索效用，障碍编码碰撞风险，不能只用同构 UAV 图。第三，通信恢复后拿到的信念可能过时，可达不等于可信，必须回答“该信哪一个源”。请强调一句：我们优化的是通信约束下的决策，而不是去改 MAC 或路由协议。下一页把三个挑战一一映射到机制，完成从问题到方法的转折。
【English】
From those applications, intermittent communication yields three challenges. First, there is no shared global map; observations are local and the neighbor set changes with A_{i,j,t}, so decisions are decentralized. Second, peers, tasks or frontiers, and obstacles play different roles: cooperation, search utility, and collision risk. A homogeneous UAV graph is not enough. Third, beliefs recovered after an outage may be stale, so reachability is not usefulness: we must choose which source to trust. Our focus is decision-making under communication constraints, not protocol optimization. The next slide maps each challenge onto a mechanism and turns the problem into a method.`,

`【中文】
这一页是转折页，请放慢一点。挑战一用不完整信念来表述，对应 revisit-aware 的 P/E/R 记忆：P 是目标存在概率，E 是不确定性，R 是回访陈旧度。挑战二是过时邻居信息，对应通信历史感知的源选择，每个格子只选一个源，避免相关 log-odds 相加。挑战三是异构实体，对应异构图注意力。学习框架是共享 Actor、训练期集中 Critic、MAPPO 加 CTDE；底部安全投影对所有对比方法相同，不是本文贡献。三个问题汇到 IC-HGAT-MAPPO 之后，下一页进入第二部分，先把搜索场景和评价指标写清楚。
【English】
This is the turning slide; please slow down. Incomplete beliefs are handled by revisit-aware P/E/R memory: existence P, uncertainty E, and revisit staleness R. Stale peer information is handled by communication-history-aware source selection, one source per cell, without adding correlated log-odds. Heterogeneous entities are handled by heterogeneous graph attention. Training uses a shared actor, a centralized critic, and MAPPO with CTDE. The safety projection is shared by all methods and is not a contribution. After these three questions collapse into IC-HGAT-MAPPO, Part II begins with the search scene and the evaluation metrics.`,


`【中文】
问题设定对应论文第二节，请先看左侧场景图：实线是当前连通，虚线是断链，还有局部感知、运动目标和动态障碍。仿真是 6 架无人机、15 个目标、10 个障碍、30×30 网格；动作为八方向加悬停。运动方程是可行域投影后的离散更新，决策间隔 Δt 可在 0.45 到 1.0 秒自适应。通信用指示函数 A_{i,j,t}：距离不超过 Rc，且双方都在工作、都允许发射。策略只能使用瞬时可达集合 𝒩_i(t)。评价指标请连公式一起念：发现率 Dt 是已发现目标占比；信念决断度 Ct 是格子上 |2P−1| 的均值，越接近 1 越决断；T80 是 Dt 首次达到 0.8 的最早时刻。Dt、Ct 越高越好，T80 越低越快。设定讲完后，下一页按数据流把整套算法串起来。
【English】
This is the problem formulation. The figure shows live and broken links, local sensing, moving targets, and dynamic obstacles. The setting has 6 UAVs, 15 targets, 10 obstacles, and a 30 by 30 grid, with eight move directions plus hover. Motion is a discrete update with a feasible-set projection and an adaptive interval Δt in [0.45, 1.0] s. The link indicator A_{i,j,t} requires range within Rc and both UAVs active. The policy may use only the instantaneous set 𝒩_i(t). Discovery Dt is the fraction of found targets; decisiveness Ct is the mean of |2P−1| over cells; T80 is the first time Dt reaches 0.8. Higher Dt and Ct are better; lower T80 is faster. With the setting fixed, the next slide walks the full data flow.`,

`【中文】
请严格按箭头从左到右讲，不要跳模块。执行时：局部观测和链路掩码先更新 P/E/R 记忆，再构造异构图，共享 Actor 输出九个名义动作，最后经安全投影落到可行动作。训练时集中 Critic 看到全局状态，用 MAPPO 的 clip 目标和 GAE 优势写回 Actor；执行阶段去掉 Critic，这就是 CTDE。务必说明：安全投影对所有对比方法相同，性能不能记在 safety layer 上。框架清楚以后，下一页拆开两个信念模块：记什么，以及信谁。
【English】
Please follow the arrows from left to right. At execution, local observation and the link mask update P/E/R memory, then the heterogeneous graph, then the shared actor with nine nominal actions, then safety projection. During training, a centralized critic sees global state and MAPPO writes the update back through a clipped objective and GAE advantages. The critic is removed at execution; that is CTDE. Safety projection is shared by all methods, so it is not the source of the gains. After the pipeline, the next slide opens the two belief modules: what to remember, and whom to trust.`,

`【中文】
左栏回答“记什么”。每架无人机在格子 x 上维护 log-odds L，再映射成存在概率 P=σ(L)；不确定性 E=exp(−kq|L|)，|L| 越大越确定。回访场 R 以老化系数 α=0.035 增长，并以 β=0.08 向邻域扩散；被感知格子重置为 0。前沿分数 q=E+0.15P+0.65R，后面会用来抽任务节点。右栏回答“信谁”。即使 j 和 k 都能通信，也按距离衰减和通信历史 g_{j,t} 加权，βd=0.28；每个格子只选使 w|L| 最大的一个源，避免把相关 log-odds 简单相加。信念怎么进策略，下一页用异构图注意力和 MAPPO 来回答。
【English】
The left column answers what to remember. Each UAV keeps a log-odds field L and maps it to existence P=σ(L). Uncertainty is E=exp(−kq|L|): larger |L| is more certain. Revisit staleness R ages with α=0.035 and diffuses with β=0.08; sensed cells reset to zero. The frontier score q=E+0.15P+0.65R later nominates task nodes. The right column answers whom to trust. Even if both j and k are reachable, weights use distance decay and communication history g_{j,t} with βd=0.28, and one source is chosen per cell by maximizing w|L|. We do not add correlated log-odds. The next slide feeds these beliefs into heterogeneous graph attention and MAPPO.`,

`【中文】
异构图以本机 UAV 为中心，分别连同伴、任务和障碍三类关系。注意力不是普通 GAT：除 QK 项外，还有关系置信 log χ、距离先验 −κr d_ik，以及链路掩码。κr 对同伴取 1.5，对任务和障碍取 0.7；网络是 2 层 4 头，任务槽 6 个、障碍槽 4 个。右侧是共享 Actor 输出 9 个名义动作；单一集中 Critic 只在训练使用。奖励是发现增量加决断度增量，减去安全代价，权重 1.0、0.30、0.10。优化用 MAPPO：重要性比 ρt、clip 系数 ε=0.20、GAE λ=0.95、γ=0.99。完整推导不必展开。方法讲完，下一页进入实验：先看协议和主结果。
【English】
The graph is centered on UAV i, with peer, task, and obstacle relations. Attention is not a plain GAT: besides the QK term, it uses relation confidence log χ, a distance prior −κr d_ik, and a link mask. κr is 1.5 for peers and 0.7 for tasks and obstacles. The network has two layers and four heads, with six task slots and four obstacle slots. The shared actor outputs nine nominal actions; one centralized critic is used only in training. The reward is discovery plus decisiveness minus a safety cost, with weights 1.0, 0.30, and 0.10. MAPPO uses the importance ratio ρt, clip ε=0.20, GAE λ=0.95, and γ=0.99. The full PPO derivation can wait for questions. With the method in place, the next slide turns to experiments: protocol and main results.`,


`【中文】
实验协议公平：同一环境、奖励、动作空间和训练预算；感知半径 2 格、通信半径 10 格，对应 10 米和 50 米；时域 200 步，训练 2000 回合；5 个独立策略，30 个保留场景。左下轨迹图只说明系统长什么样，不是定量证据。主结果请看柱状图：决断度 62.15%，发现率 86.78%。相对启发式提高 17.16 和 9.00 个百分点，95% 置信区间大约 12.96–21.36；相对 QMIX 提高 32.15 和 14.67 个百分点，区间大约 28.30–36.01。T80 成功率为 82.0%，成功条件下 88.63 步。安全干预仅 0.17%，碰撞与启发式相当，不要把安全列说成本文优势。主结果之后，下一页用消融证明三个模块各自有贡献，并用敏感性看扰动是否吃掉优势。
【English】
The protocol is shared: same environment, reward, action space, and budget; sensing radius 2 cells and communication radius 10 cells, that is 10 m and 50 m; horizon 200 and 2,000 training episodes; five independently trained policies and thirty held-out scenarios. The trajectory figure shows what a run looks like; it is not the quantitative evidence. Read the bars: 62.15 percent decisiveness and 86.78 percent discovery. Versus the heuristic that is plus 17.16 and plus 9.00 points, with a 95 percent CI of about 12.96 to 21.36 on decisiveness; versus QMIX, plus 32.15 and plus 14.67, with a CI of about 28.30 to 36.01. T80 succeeds in 82.0 percent of runs at 88.63 steps when successful. Safety intervention is only 0.17 percent, and collision is comparable to the heuristic, so do not rank the method on safety columns. Next, ablations show that each module contributes, and sensitivity checks whether the gain survives perturbations.`,

`【中文】
消融说明三个模块都有用。完整模型 62.15 / 86.78。相对同构图 47.05 / 79.78，异构关系带来约 15.10 个百分点决断度。去掉 revisit 记忆落到 45.82 / 73.11，决断度和发现率分别下降 16.33 和 13.67 个百分点。去掉历史加权落到 53.99 / 85.33，决断度下降 8.16，条件 T80 从 88.63 变到 98.19，但阈值成功率从 82.0% 升到 86.7%——这是需要主动承认的权衡：历史加权更决断、条件上更快，但不加权时更多回合能跨过 80% 发现阈值。下方敏感性不必六张图逐张讲，举目标速度和通信半径或中断概率两例，然后说在测试扰动范围内平均发现率更高。数字和机制都齐了，下一页用三条贡献收束，并给出按论文写的未来工作。
【English】
Ablations support all three modules. The full model is 62.15 / 86.78. Heterogeneous relations add about 15.10 points of decisiveness over the homogeneous graph at 47.05 / 79.78. Removing revisit memory drops to 45.82 / 73.11, that is minus 16.33 and minus 13.67 points. Removing history weighting drops to 53.99 / 85.33: decisiveness falls by 8.16 points and conditional T80 slows from 88.63 to 98.19, while threshold success rises from 82.0 to 86.7 percent. That trade-off should be stated: history weighting is more decisive and conditionally faster, but without it more runs cross the 80 percent discovery threshold. Do not walk through all six sensitivity plots. Mention target speed and communication radius or outage, then conclude that mean discovery stays higher over the tested ranges. With numbers and mechanisms in place, the next slide closes with three contributions and the paper’s future work.`,

`【中文】
总结只留三条，请对应前面三个挑战再念一遍：第一，通信约束下的去中心决策，策略只用局部观测和瞬时可达邻居；第二，信念管理，也就是 P/E/R 记忆加上历史感知选源；第三，异构关系编码，用 HGAT 把同伴、任务和障碍写进 MAPPO。请再报核心数字：62.15% 决断度、86.78% 发现率、82.0% 的 T80 成功率，成功条件下 88.63 步。未来工作按论文来：三维环境、完全去中心安全、以及含真实通信的半实物验证。如果被问到仓库里的围捕代码，只作为口头补充，不要把这篇 search 报告讲成 pursuit。最后一页致谢，欢迎提问。
【English】
Leave three contributions, matching the three challenges: communication-constrained decentralized decisions that use only local observations and reachable peers; belief management with P/E/R and history-aware selection; and heterogeneous relational encoding inside MAPPO. Repeat 62.15 percent, 86.78 percent, 82.0 percent, and 88.63 conditional steps. Future work follows the paper: 3D environments, fully decentralized safety, and hardware-in-the-loop with real communication. If asked about pursuit code in the repo, mention it only as an ongoing extension. Do not turn this search talk into a pursuit talk. The last slide is the thank-you, and then we welcome questions.`,

`【中文】
谢谢各位老师、同学。欢迎就间歇通信下的 Dec-POMDP 建模、log-odds 信念维护、历史感知源选择，以及异构图注意力提问。联系邮箱是 kuangyinying039@163.com。如果时间允许，我可以从公式或消融里任选一处展开。
【English】
Thank you for your attention. I welcome questions on the Dec-POMDP model, log-odds belief maintenance, history-aware source selection, and heterogeneous graph attention under intermittent communication. My email is kuangyinying039@163.com. If time allows, I can expand any equation or ablation in more detail.`,
];


// ---------------------------------------------------------------------------
// 1 Cover
// ---------------------------------------------------------------------------
{
  const s = pres.addSlide();
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 13.333, h: 7.5, fill: { color: WHITE } });
  contain(s, path.join(A, "hust.png"), { x: 0.40, y: 0.22, w: 1.70, h: 1.18 });
  contain(s, path.join(A, "caa.png"), { x: 5.55, y: 0.28, w: 1.95, h: 1.10 });
  s.addText("Chinese Automation Congress  ·  CAC 2026", {
    x: 7.55, y: 0.42, w: 5.3, h: 0.80,
    fontFace: "Calibri", fontSize: 18, color: BLUE, bold: true, margin: 0, valign: "middle",
  });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 1.55, w: 13.333, h: 3.35, fill: { color: BLUE } });
  s.addText("A Heterogeneous Graph-Attention MAPPO Algorithm\nfor Cooperative Multi-UAV Search\nunder Intermittent Communication Constraints", {
    x: 0.45, y: 1.65, w: 12.4, h: 3.15,
    fontFace: "Calibri", fontSize: 30, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0,
  });
  s.addText("Yinying Kuang, Zhaowei Liang, Yongran Zhi, Huijin Fan*, Lei Liu*, Bo Wang*", {
    x: 0.5, y: 5.05, w: 12.3, h: 0.50,
    fontFace: "Calibri", fontSize: 18, color: DARK, align: "center", valign: "middle", margin: 0,
  });
  s.addText("Huazhong University of Science and Technology\nWuhan Second Ship Design and Research Institute", {
    x: 0.5, y: 5.55, w: 12.3, h: 0.78,
    fontFace: "Calibri", fontSize: 17, color: MUTED, align: "center", valign: "middle", margin: 0,
  });
  s.addText("CAC 2026  ·  Beijing  ·  October 2026", {
    x: 0.5, y: 6.32, w: 12.3, h: 0.36,
    fontFace: "Calibri", fontSize: 17, color: BLUE, align: "center", bold: true, valign: "middle", margin: 0,
  });
  const chips = ["Dec-POMDP", "Intermittent links", "P / E / R memory", "History-aware source", "HGAT + MAPPO / CTDE"];
  chips.forEach((c, i) => {
    const x = 0.38 + i * 2.58;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y: 6.78, w: 2.48, h: 0.48, fill: { color: BLUE_LT }, rectRadius: 0.08,
    });
    s.addText(c, {
      x, y: 6.78, w: 2.48, h: 0.48,
      fontFace: "Calibri", fontSize: 11, bold: true, color: BLUE_DK, align: "center", valign: "middle", margin: 0,
    });
  });
  notes(s, N[0]);
}

// ---------------------------------------------------------------------------
// 2 CONTENTS
// ---------------------------------------------------------------------------
{
  const s = pres.addSlide();
  contain(s, path.join(A, "hust.png"), { x: 12.38, y: 0.10, w: 0.82, h: 0.54 });
  s.addText("CONTENTS", {
    x: 0.38, y: 0.12, w: 11.7, h: 0.50,
    fontFace: "Calibri", fontSize: 28, bold: true, color: BLUE, margin: 0, valign: "middle",
  });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0.72, w: 13.333, h: 0.14, fill: { color: BLUE } });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.28, y: 1.15, w: 12.75, h: 0.52, fill: { color: BLUE_LT }, rectRadius: 0.08,
  });
  s.addText("A 13-page academic talk  ·  problem first, then IC-HGAT-MAPPO, then evidence", {
    x: 0.42, y: 1.15, w: 12.45, h: 0.52,
    fontFace: "Calibri", fontSize: 16, italic: true, color: BLUE_DK, align: "center", valign: "middle", margin: 0,
  });
  const items = [
    ["01", "Background and Problem Formulation", "Applications  →  three challenges  →  Q1–Q3 mapping\nDec-POMDP under intermittent links  Aij,t"],
    ["02", "Proposed IC-HGAT-MAPPO", "Motion / link / metrics   ·   P/E/R + source selection\nHGAT attention   ·   MAPPO clip / GAE / CTDE"],
    ["03", "Experimental Results and Conclusions", "Main results vs Heuristic / QMIX   ·   ablation\nSensitivity   ·   contributions and future work"],
  ];
  items.forEach((it, i) => {
    const y = 1.85 + i * 1.68;
    card(s, 0.35, y, 12.62, 1.52, i === 1 ? "FFF3E8" : WHITE);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 0.55, y: y + 0.28, w: 1.35, h: 0.95, fill: { color: BLUE }, rectRadius: 0.10,
    });
    s.addText(it[0], {
      x: 0.55, y: y + 0.28, w: 1.35, h: 0.95,
      fontFace: "Calibri", fontSize: 28, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0,
    });
    s.addText(it[1], {
      x: 2.10, y: y + 0.12, w: 10.55, h: 0.48,
      fontFace: "Calibri", fontSize: 22, bold: true, color: DARK, valign: "middle", margin: 0,
    });
    s.addText(it[2], {
      x: 2.10, y: y + 0.62, w: 10.55, h: 0.72,
      fontFace: "Calibri", fontSize: 16, color: MUTED, valign: "middle", margin: 0,
    });
  });
  footer(s, 2);
  notes(s, N[1]);
}

// ---------------------------------------------------------------------------
// 3 Background applications
// ---------------------------------------------------------------------------
{
  const s = pres.addSlide();
  header(s, "1. Background", "1.1 Cooperative Multi-UAV Search");
  const photos = [
    [path.join(A, "photo_eq.png"), "Earthquake Response"],
    [path.join(A, "photo_wf.png"), "Wildfire Monitoring"],
    [path.join(A, "photo_mr.png"), "Maritime Rescue"],
  ];
  photos.forEach((ph, i) => {
    const x = 0.40 + i * 4.30;
    contain(s, ph[0], { x, y: 1.22, w: 4.05, h: 1.85 });
    s.addText(ph[1], {
      x, y: 3.08, w: 4.05, h: 0.32,
      fontFace: "Calibri", fontSize: 15, bold: true, color: DARK, align: "center", valign: "middle", margin: 0,
    });
  });
  card(s, 0.40, 3.42, 12.55, 1.12, WHITE);
  s.addText("Dec-POMDP  (partial observation, joint reward, no assumed global map)", {
    x: 0.55, y: 3.46, w: 12.25, h: 0.32,
    fontFace: "Calibri", fontSize: 14, bold: true, color: BLUE, margin: 0, valign: "middle",
  });
  eqBox(s, "EQDEC", 0.45, 3.76, 12.40, 0.68, 16);
  s.addShape(pres.shapes.RECTANGLE, { x: 0.40, y: 4.62, w: 12.55, h: 0.38, fill: { color: GRAY } });
  s.addText("Global / reliable information    →     Local observation + intermittent communication    ·    More realistic, more challenging", {
    x: 0.50, y: 4.62, w: 12.35, h: 0.38,
    fontFace: "Calibri", fontSize: 13, color: DARK, align: "center", valign: "middle", margin: 0,
  });

  const cols = [
    ["Conventional Setting", "Stable communication\nSynchronized / near-global information\nHomogeneous UAV–UAV graph", BLUE_LT, BLUE],
    ["Practical Setting", "Intermittent links  A(i,j,t)  and  N_i(t)\nLocal, incomplete, possibly stale beliefs\nUAV / target / obstacle interactions", "FFF3E8", ORANGE],
    ["Research Needs", "1. Maintain useful local beliefs  P, E, R\n2. Select reachable peer evidence\n3. Encode heterogeneous relations", "FDECEA", RED],
  ];
  cols.forEach((c, i) => {
    const x = 0.40 + i * 4.30;
    card(s, x, 5.08, 4.05, 1.88, c[2]);
    accent(s, x, 5.08, 1.88);
    s.addText(c[0], {
      x: x + 0.18, y: 5.14, w: 3.72, h: 0.34,
      fontFace: "Calibri", fontSize: 15, bold: true, color: c[3], margin: 0, valign: "middle",
    });
    s.addText(c[1], {
      x: x + 0.18, y: 5.48, w: 3.72, h: 1.38,
      fontFace: "Calibri", fontSize: 13, color: DARK, margin: 0, valign: "middle",
    });
  });
  footer(s, 3);
  notes(s, N[2]);
}

// ---------------------------------------------------------------------------
// 4 Three challenges
// ---------------------------------------------------------------------------
{
  const s = pres.addSlide();
  header(s, "1. Background", "1.2 Key Challenges under Intermittent Communication");
  const titles = [
    ["1  Incomplete Information Exchange", "chal1.png"],
    ["2  Heterogeneous Interactions", "chal2.png"],
    ["3  Information Staleness", "chal3.png"],
  ];
  const bodies = [
    "No shared global map; local o_i,t only\nReachable set changes with A(i,j,t)\nDecentralized policy on local memory",
    "Peer: cooperation / message fusion\nTask / frontier: search utility q(x)\nObstacle: collision risk in feasible set",
    "Recovered beliefs may be stale\nReachable ≠ useful  (history gj,t)\nSelect one source per cell",
  ];
  titles.forEach((t, i) => {
    const x = 0.35 + i * 4.32;
    card(s, x, 1.22, 4.15, 4.18, WHITE);
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.22, w: 4.15, h: 0.44, fill: { color: i === 2 ? RED : BLUE } });
    s.addText(t[0], {
      x: x + 0.10, y: 1.24, w: 3.95, h: 0.40,
      fontFace: "Calibri", fontSize: 13, bold: true, color: WHITE, margin: 0, valign: "middle",
    });
    contain(s, path.join(A, t[1]), { x: x + 0.12, y: 1.74, w: 3.90, h: 1.85 });
    s.addText(bodies[i], {
      x: x + 0.16, y: 3.62, w: 3.82, h: 1.62,
      fontFace: "Calibri", fontSize: 13, color: DARK, margin: 0, valign: "middle",
    });
  });
  card(s, 0.35, 5.48, 12.62, 1.48, BLUE_LT);
  s.addText("Instantaneous neighbor set used by the policy (not a delayed topology)", {
    x: 0.50, y: 5.54, w: 12.32, h: 0.28,
    fontFace: "Calibri", fontSize: 13, bold: true, color: BLUE, margin: 0, valign: "middle",
  });
  eqBox(s, "EQNI", 0.40, 5.82, 12.50, 0.58, 16);
  s.addText("Focus: cooperative decision-making under communication constraints, rather than protocol optimization.", {
    x: 0.50, y: 6.42, w: 12.32, h: 0.42,
    fontFace: "Calibri", fontSize: 14, italic: true, color: BLUE_DK, align: "center", valign: "middle", margin: 0,
  });
  footer(s, 4);
  notes(s, N[3]);
}

// ---------------------------------------------------------------------------
// 5 Challenge–solution mapping
// ---------------------------------------------------------------------------
{
  const s = pres.addSlide();
  header(s, "1. Background", "1.3 Proposed Solution");
  const maps = [
    ["Challenge 1", "Incomplete beliefs", "Revisit-aware\nP/E/R Memory", BLUE],
    ["Challenge 2", "Stale peer information", "Communication-History-\nAware Source Selection", ORANGE],
    ["Challenge 3", "Heterogeneous entities", "Heterogeneous\nGraph Attention", GREEN],
  ];
  maps.forEach((m, i) => {
    const x = 0.35 + i * 3.15;
    card(s, x, 1.22, 3.00, 2.55, WHITE);
    accent(s, x, 1.22, 2.55);
    s.addText(m[0], { x, y: 1.28, w: 3.00, h: 0.28, fontFace: "Calibri", fontSize: 12, color: MUTED, align: "center", margin: 0, valign: "middle" });
    s.addText(m[1], { x: x + 0.10, y: 1.54, w: 2.80, h: 0.42, fontFace: "Calibri", fontSize: 14, bold: true, color: DARK, align: "center", margin: 0, valign: "middle" });
    s.addShape(pres.shapes.DOWN_ARROW, { x: x + 1.15, y: 1.98, w: 0.70, h: 0.28, fill: { color: m[3] } });
    s.addText(m[2], { x: x + 0.08, y: 2.28, w: 2.84, h: 1.32, fontFace: "Calibri", fontSize: 14, bold: true, color: m[3], align: "center", margin: 0, valign: "middle" });
  });
  card(s, 9.85, 1.22, 3.12, 2.55, BLUE);
  s.addText("Learning (CTDE)", {
    x: 10.00, y: 1.32, w: 2.85, h: 0.36, fontFace: "Calibri", fontSize: 15, bold: true, color: WHITE, margin: 0, valign: "middle",
  });
  s.addText("Shared actor  pi(a|o)\nCentralized critic V(s)\nMAPPO: clip + GAE\nCommon safety projection", {
    x: 10.00, y: 1.70, w: 2.85, h: 1.90, fontFace: "Calibri", fontSize: 14, color: WHITE, margin: 0, valign: "middle",
  });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.35, y: 3.90, w: 12.62, h: 3.05, fill: { color: BLUE_LT }, rectRadius: 0.08,
  });
  s.addText("From three questions to one algorithm  ·  IC-HGAT-MAPPO", {
    x: 0.55, y: 3.98, w: 12.22, h: 0.36, fontFace: "Calibri", fontSize: 16, bold: true, color: BLUE_DK, margin: 0, valign: "middle",
  });
  const qs = [
    ["Q1", "How to maintain useful beliefs without global map sharing?", "Bayesian log-odds L  →  P, E, R"],
    ["Q2", "How to select useful information after communication recovery?", "One source j⋆ per cell via w |L|"],
    ["Q3", "How to model UAV, task, and obstacle interactions jointly?", "Relation-specific attention  alpha_r,i,k"],
  ];
  qs.forEach((q, i) => {
    const y = 4.40 + i * 0.78;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 0.52, y, w: 0.72, h: 0.64, fill: { color: RED }, rectRadius: 0.06,
    });
    s.addText(q[0], {
      x: 0.52, y, w: 0.72, h: 0.64,
      fontFace: "Calibri", fontSize: 16, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0,
    });
    s.addText(q[1], {
      x: 1.38, y, w: 7.85, h: 0.36,
      fontFace: "Calibri", fontSize: 14, bold: true, color: DARK, margin: 0, valign: "middle",
    });
    s.addText(q[2], {
      x: 1.38, y: y + 0.32, w: 7.85, h: 0.30,
      fontFace: "Calibri", fontSize: 13, italic: true, color: MUTED, margin: 0, valign: "middle",
    });
  });
  s.addShape(pres.shapes.RIGHT_ARROW, { x: 9.40, y: 5.35, w: 0.50, h: 0.30, fill: { color: BLUE } });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 10.00, y: 4.70, w: 2.75, h: 1.90, fill: { color: BLUE }, rectRadius: 0.08,
  });
  s.addText("IC-HGAT-\nMAPPO", {
    x: 10.05, y: 4.88, w: 2.65, h: 1.00,
    fontFace: "Calibri", fontSize: 20, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0,
  });
  s.addText("exec: local only\ntrain: global critic", {
    x: 10.05, y: 5.90, w: 2.65, h: 0.55,
    fontFace: "Calibri", fontSize: 12, color: "D6E2F5", align: "center", valign: "middle", margin: 0,
  });
  footer(s, 5);
  notes(s, N[4]);
}

// ---------------------------------------------------------------------------
// 6 Problem formulation
// ---------------------------------------------------------------------------
{
  const s = pres.addSlide();
  header(s, "2. Proposed IC-HGAT-MAPPO", "2.1 Problem Formulation");
  contain(s, path.join(A, "fig1.png"), { x: 0.28, y: 1.20, w: 6.40, h: 3.05 });
  s.addText("Fig. 1  Local sensing, intermittent links, moving targets, dynamic obstacles", {
    x: 0.28, y: 4.26, w: 6.40, h: 0.28, fontFace: "Calibri", fontSize: 11, italic: true, color: MUTED, margin: 0, valign: "middle",
  });
  card(s, 0.28, 4.56, 6.40, 2.40, GRAY);
  s.addText("Scenario  ·  sensing / comm. radii", { x: 0.42, y: 4.60, w: 6.10, h: 0.28, fontFace: "Calibri", fontSize: 14, bold: true, color: BLUE, margin: 0, valign: "middle" });
  s.addText("6 UAVs · 15 targets · 10 obstacles · 30×30 grid\nAction: 8 directions + hover     Horizon T=200\nRs = 2 cells (10 m)    Rc = 10 cells (50 m)\ndt in [0.45, 1.0] s    ·    policy uses only N_i(t)", {
    x: 0.42, y: 4.90, w: 6.10, h: 1.92, fontFace: "Calibri", fontSize: 14, color: DARK, margin: 0, valign: "middle",
  });

  card(s, 6.85, 1.18, 6.15, 1.42, WHITE);
  s.addText("Motion model  (feasible-set projection)", { x: 6.98, y: 1.20, w: 5.90, h: 0.26, fontFace: "Calibri", fontSize: 13, bold: true, color: BLUE, margin: 0, valign: "middle" });
  eqBox(s, "EQMOTION", 6.90, 1.44, 6.02, 0.70, 16);
  s.addText("Discrete kinematics with adaptive decision interval dt", {
    x: 6.98, y: 2.16, w: 5.90, h: 0.36, fontFace: "Calibri", fontSize: 12, italic: true, color: MUTED, margin: 0, valign: "middle",
  });

  card(s, 6.85, 2.68, 6.15, 1.58, WHITE);
  s.addText("Communication constraint", { x: 6.98, y: 2.72, w: 5.90, h: 0.24, fontFace: "Calibri", fontSize: 13, bold: true, color: BLUE, margin: 0, valign: "middle" });
  eqBox(s, "EQLINK", 6.88, 2.94, 6.08, 0.72, 14);
  s.addText("Only instantaneous reachable links enter the policy", {
    x: 6.98, y: 3.68, w: 5.90, h: 0.46, fontFace: "Calibri", fontSize: 13, bold: true, color: RED, margin: 0, valign: "middle",
  });

  card(s, 6.85, 4.34, 6.15, 2.62, BLUE_LT);
  s.addText("Evaluation  ·  higher Dt, Ct better; lower T80 faster", { x: 6.98, y: 4.38, w: 5.90, h: 0.26, fontFace: "Calibri", fontSize: 13, bold: true, color: BLUE, margin: 0, valign: "middle" });
  eqBox(s, "EQDT", 6.88, 4.64, 6.08, 0.58, 15);
  eqBox(s, "EQCT", 6.88, 5.20, 6.08, 0.70, 14);
  eqBox(s, "EQT80", 6.88, 5.88, 6.08, 0.58, 15);
  s.addText("Ct: cell-wise decisiveness of occupancy beliefs", {
    x: 6.98, y: 6.46, w: 5.90, h: 0.38, fontFace: "Calibri", fontSize: 12, italic: true, color: MUTED, margin: 0, valign: "middle",
  });
  footer(s, 6);
  notes(s, N[5]);
}

// ---------------------------------------------------------------------------
// 7 Framework
// ---------------------------------------------------------------------------
{
  const s = pres.addSlide();
  header(s, "2. Proposed IC-HGAT-MAPPO", "2.2 IC-HGAT-MAPPO Framework");
  contain(s, path.join(A, "framework.png"), { x: 0.22, y: 1.18, w: 12.90, h: 3.05 });
  const feats = [
    ["1  Observe", "oi,t + link mask A·,t"],
    ["2  Remember", "update P / E / R fields"],
    ["3  Select", "one source j⋆ per cell"],
    ["4  Attend", "HGAT attention  →  embedding"],
    ["5  Act", "shared actor  →  safety proj."],
    ["6  Train", "central critic, MAPPO clip"],
  ];
  feats.forEach((f, i) => {
    const x = 0.28 + (i % 6) * 2.16;
    card(s, x, 4.30, 2.08, 1.22, i === 5 ? "FFF3E8" : BLUE_LT);
    s.addText(f[0], { x: x + 0.08, y: 4.36, w: 1.92, h: 0.36, fontFace: "Calibri", fontSize: 13, bold: true, color: BLUE_DK, margin: 0, valign: "middle" });
    s.addText(f[1], { x: x + 0.08, y: 4.72, w: 1.92, h: 0.68, fontFace: "Calibri", fontSize: 12, color: DARK, margin: 0, valign: "middle" });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.28, y: 5.62, w: 12.78, h: 1.32, fill: { color: GRAY }, rectRadius: 0.08,
  });
  s.addText("CTDE  ·  execution uses local + reachable information only; the critic sees global s only in training.", {
    x: 0.42, y: 5.68, w: 12.50, h: 0.32, fontFace: "Calibri", fontSize: 13, bold: true, color: BLUE, margin: 0, valign: "middle",
  });
  s.addText("Safety projection is shared by Heuristic, QMIX, Local, Homogeneous-Graph, and IC-HGAT-MAPPO — not a contribution and not the source of the gains.\nDecentralized execution  ·  communication-aware masking  ·  centralized training with MAPPO.", {
    x: 0.42, y: 6.02, w: 12.50, h: 0.80, fontFace: "Calibri", fontSize: 13, color: DARK, margin: 0, valign: "middle",
  });
  footer(s, 7);
  notes(s, N[6]);
}

// ---------------------------------------------------------------------------
// 8 P/E/R + source selection
// ---------------------------------------------------------------------------
{
  const s = pres.addSlide();
  header(s, "2. Proposed IC-HGAT-MAPPO", "2.3 Local Belief Maintenance and Information Selection");
  card(s, 0.28, 1.18, 6.45, 5.78, WHITE);
  accent(s, 0.28, 1.18, 5.78);
  s.addText("Revisit-Aware P/E/R Memory", {
    x: 0.46, y: 1.22, w: 6.10, h: 0.34, fontFace: "Calibri", fontSize: 16, bold: true, color: BLUE, margin: 0, valign: "middle",
  });
  contain(s, path.join(A, "per_maps.png"), { x: 0.42, y: 1.56, w: 6.15, h: 1.55 });
  eqBox(s, "EQL", 0.36, 3.12, 6.25, 0.58, 14);
  eqBox(s, "EQP", 0.36, 3.68, 6.25, 0.55, 14);
  eqBox(s, "EQE", 0.36, 4.22, 6.25, 0.52, 14);
  eqBox(s, "EQRFIELD", 0.36, 4.72, 6.25, 0.58, 13);
  eqBox(s, "EQFRONT", 0.36, 5.28, 6.25, 0.52, 14);
  s.addText("Sensed cells: R <- 0.    Unsensed: alpha=0.035 aging, beta=0.08 diffusion.\nFrontier score q nominates task nodes for HGAT.", {
    x: 0.46, y: 5.82, w: 6.10, h: 0.98, fontFace: "Calibri", fontSize: 12, color: MUTED, margin: 0, valign: "middle",
  });

  card(s, 6.90, 1.18, 6.12, 5.78, WHITE);
  accent(s, 6.90, 1.18, 5.78);
  s.addText("Communication-History-Aware Source Selection", {
    x: 7.08, y: 1.22, w: 5.78, h: 0.40, fontFace: "Calibri", fontSize: 14, bold: true, color: ORANGE, margin: 0, valign: "middle",
  });
  contain(s, path.join(A, "source.png"), { x: 7.05, y: 1.64, w: 5.80, h: 1.85 });
  s.addText("βd = 0.28     ·     δ = 0.82 history decay     ·     gj,t = outage age", {
    x: 7.08, y: 3.52, w: 5.78, h: 0.32, fontFace: "Calibri", fontSize: 12, color: MUTED, margin: 0, valign: "middle",
  });
  eqBox(s, "EQW", 6.98, 3.84, 5.95, 0.62, 14);
  eqBox(s, "EQJSTAR", 6.98, 4.46, 5.95, 0.58, 14);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 7.08, y: 5.18, w: 5.76, h: 1.58, fill: { color: "FDECEA" }, rectRadius: 0.08,
  });
  s.addText("One reachable source per cell", {
    x: 7.22, y: 5.26, w: 5.48, h: 0.36, fontFace: "Calibri", fontSize: 15, bold: true, color: RED, margin: 0, valign: "middle",
  });
  s.addText("Do not sum correlated log-odds from j and k.\nWinner takes the cell: fuse L from j⋆ only.\nDistance decay + communication history gj,t.", {
    x: 7.22, y: 5.62, w: 5.48, h: 0.98, fontFace: "Calibri", fontSize: 13, color: DARK, margin: 0, valign: "middle",
  });
  footer(s, 8);
  notes(s, N[7]);
}

// ---------------------------------------------------------------------------
// 9 HGAT + MAPPO
// ---------------------------------------------------------------------------
{
  const s = pres.addSlide();
  header(s, "2. Proposed IC-HGAT-MAPPO", "2.4 Heterogeneous Graph Policy and MAPPO Optimization");
  contain(s, path.join(A, "hgat.png"), { x: 0.18, y: 1.18, w: 6.40, h: 2.35 });
  s.addText("Relation-specific attention  ·  k_peer=1.5, k_task=k_obs=0.7  ·  2 layers, 4 heads  ·  6 task / 4 obstacle slots", {
    x: 0.22, y: 3.52, w: 6.40, h: 0.32, fontFace: "Calibri", fontSize: 11, color: GREEN, bold: true, margin: 0, valign: "middle",
  });
  eqBox(s, "EQATT", 0.18, 3.82, 6.45, 0.70, 14);
  contain(s, path.join(A, "actor_critic.png"), { x: 6.75, y: 1.18, w: 6.25, h: 2.35 });
  eqBox(s, "EQREWARD", 6.80, 3.52, 6.15, 0.52, 15);
  s.addText("(lambda_D, lambda_C, lambda_safe) = (1.0, 0.30, 0.10)", {
    x: 6.85, y: 4.02, w: 6.10, h: 0.26, fontFace: "Calibri", fontSize: 12, color: MUTED, margin: 0, valign: "middle",
  });
  card(s, 0.18, 4.58, 13.00, 2.38, WHITE);
  s.addText("MAPPO  (CTDE)  ·  clip eps=0.20  ·  GAE lambda=0.95  ·  gamma=0.99  ·  shared actor, one critic", {
    x: 0.32, y: 4.62, w: 12.70, h: 0.28, fontFace: "Calibri", fontSize: 13, bold: true, color: BLUE, margin: 0, valign: "middle",
  });
  eqBox(s, "EQRHO", 0.22, 4.90, 6.40, 0.58, 13);
  eqBox(s, "EQDELTA", 6.70, 4.90, 6.35, 0.58, 13);
  eqBox(s, "EQCLIP", 0.22, 5.48, 6.40, 0.72, 12);
  eqBox(s, "EQGAE", 6.70, 5.48, 6.35, 0.72, 13);
  s.addText("Link mask zeros unreachable peer attention.  Critic is removed at execution.", {
    x: 0.32, y: 6.22, w: 12.70, h: 0.62, fontFace: "Calibri", fontSize: 13, italic: true, color: MUTED, margin: 0, valign: "middle",
  });
  footer(s, 9);
  notes(s, N[8]);
}

// ---------------------------------------------------------------------------
// 10 Main results
// ---------------------------------------------------------------------------
{
  const s = pres.addSlide();
  header(s, "3. Experimental Results", "3.1 Experimental Setup and Main Results");
  card(s, 0.28, 1.16, 5.50, 2.28, GRAY);
  s.addText("Experiment protocol  (fair comparison)", { x: 0.40, y: 1.20, w: 5.26, h: 0.28, fontFace: "Calibri", fontSize: 14, bold: true, color: BLUE, margin: 0, valign: "middle" });
  s.addText("6 UAVs · 15 targets · 10 obstacles · 30×30\nHorizon 200  ·  2,000 episodes  ·  5 seeds\n30 held-out scenarios   Rs=2, Rc=10 cells\nSame env, reward, action, budget", {
    x: 0.40, y: 1.50, w: 5.26, h: 1.28, fontFace: "Calibri", fontSize: 13, color: DARK, margin: 0,
  });
  s.addText("Baselines: Heuristic, QMIX, Local MAPPO, Homog. Graph MAPPO", {
    x: 0.40, y: 2.80, w: 5.26, h: 0.52, fontFace: "Calibri", fontSize: 12, italic: true, color: MUTED, margin: 0, valign: "middle",
  });
  contain(s, path.join(A, "fig3.png"), { x: 0.28, y: 3.50, w: 5.50, h: 2.55 });
  s.addText("Trajectories are qualitative; bars below are the evidence.", {
    x: 0.28, y: 6.08, w: 5.50, h: 0.28, fontFace: "Calibri", fontSize: 11, italic: true, color: MUTED, margin: 0, valign: "middle",
  });
  card(s, 0.28, 6.38, 5.50, 0.58, BLUE_LT);
  s.addText("95% CI vs Heuristic Ct: 12.96–21.36 pp   ·   vs QMIX: 28.30–36.01 pp", {
    x: 0.36, y: 6.38, w: 5.34, h: 0.58, fontFace: "Calibri", fontSize: 11, color: BLUE_DK, margin: 0, valign: "middle",
  });

  const labels = ["Heuristic", "QMIX", "Local", "H-Graph", "IC-HGAT"];
  s.addChart(pres.charts.BAR, [{
    name: "Decisiveness",
    labels,
    values: [44.99, 30.00, 41.91, 47.05, 62.15],
  }], {
    x: 5.95, y: 1.15, w: 3.60, h: 2.55,
    barDir: "col",
    showValue: true,
    dataLabelPosition: "inEnd",
    dataLabelColor: DARK,
    dataLabelFontSize: 9,
    showLegend: false,
    showTitle: true,
    title: "Belief Decisiveness (%)",
    titleColor: DARK,
    titleFontSize: 11,
    titleFontFace: "Calibri",
    chartColors: ["5B8CC9", "5B8CC9", "5B8CC9", "5B8CC9", ORANGE],
    valAxisMaxValue: 80,
    catAxisLabelColor: DARK,
    valAxisLabelColor: MUTED,
    catAxisLabelFontSize: 8,
    valGridLine: { color: "D0D7E2", size: 0.5 },
    catGridLine: { style: "none" },
  });
  s.addChart(pres.charts.BAR, [{
    name: "Discovery",
    labels,
    values: [77.78, 72.11, 76.22, 79.78, 86.78],
  }], {
    x: 9.55, y: 1.15, w: 3.50, h: 2.55,
    barDir: "col",
    showValue: true,
    dataLabelPosition: "inEnd",
    dataLabelColor: DARK,
    dataLabelFontSize: 9,
    showLegend: false,
    showTitle: true,
    title: "Target Discovery (%)",
    titleColor: DARK,
    titleFontSize: 11,
    titleFontFace: "Calibri",
    chartColors: ["5B8CC9", "5B8CC9", "5B8CC9", "5B8CC9", ORANGE],
    valAxisMaxValue: 100,
    catAxisLabelColor: DARK,
    valAxisLabelColor: MUTED,
    catAxisLabelFontSize: 8,
    valGridLine: { color: "D0D7E2", size: 0.5 },
    catGridLine: { style: "none" },
  });

  card(s, 5.95, 3.80, 3.50, 1.55, "FDECEA");
  s.addText("vs. Heuristic", { x: 6.10, y: 3.88, w: 3.20, h: 0.28, fontFace: "Calibri", fontSize: 12, color: RED, bold: true, margin: 0 });
  s.addText("+17.16 pp  decisiveness\n+9.00 pp  discovery", {
    x: 6.10, y: 4.20, w: 3.20, h: 0.95, fontFace: "Calibri", fontSize: 16, bold: true, color: DARK, margin: 0, valign: "middle",
  });
  card(s, 9.55, 3.80, 3.50, 1.55, "FFF3E8");
  s.addText("vs. QMIX", { x: 9.70, y: 3.88, w: 3.20, h: 0.28, fontFace: "Calibri", fontSize: 12, color: ORANGE, bold: true, margin: 0 });
  s.addText("+32.15 pp  decisiveness\n+14.67 pp  discovery", {
    x: 9.70, y: 4.20, w: 3.20, h: 0.95, fontFace: "Calibri", fontSize: 16, bold: true, color: DARK, margin: 0, valign: "middle",
  });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 5.95, y: 5.50, w: 7.10, h: 1.28, fill: { color: BLUE }, rectRadius: 0.08,
  });
  s.addText("T80   ·   82.0% success     ·     88.63 steps conditional on success", {
    x: 6.15, y: 5.58, w: 6.75, h: 0.55, fontFace: "Calibri", fontSize: 16, bold: true, color: WHITE, margin: 0, valign: "middle",
  });
  s.addText("Safety intervention 0.17%; collision rate comparable to the heuristic (not ranked).", {
    x: 6.15, y: 6.16, w: 6.75, h: 0.44, fontFace: "Calibri", fontSize: 13, color: "D6E2F5", margin: 0, valign: "middle",
  });
  footer(s, 10);
  notes(s, N[9]);
}

// ---------------------------------------------------------------------------
// 11 Ablation + sensitivity
// ---------------------------------------------------------------------------
{
  const s = pres.addSlide();
  header(s, "3. Experimental Results", "3.2 Ablation and Sensitivity Analysis");
  const alabels = ["Full", "w/o Revisit", "w/o History", "Homog. Graph"];
  s.addChart(pres.charts.BAR, [
    { name: "Decisiveness", labels: alabels, values: [62.15, 45.82, 53.99, 47.05] },
    { name: "Discovery", labels: alabels, values: [86.78, 73.11, 85.33, 79.78] },
  ], {
    x: 0.20, y: 1.12, w: 6.40, h: 2.55,
    barDir: "col",
    barGrouping: "clustered",
    showValue: true,
    dataLabelPosition: "inEnd",
    dataLabelFontSize: 8,
    dataLabelColor: DARK,
    showLegend: true,
    legendPos: "r",
    showTitle: true,
    title: "Ablation  (Decisiveness / Discovery, %)",
    titleColor: DARK,
    titleFontSize: 11,
    titleFontFace: "Calibri",
    chartColors: [BLUE, ORANGE],
    valAxisMaxValue: 100,
    catAxisLabelColor: DARK,
    valAxisLabelColor: MUTED,
    catAxisLabelFontSize: 9,
    valGridLine: { color: "D0D7E2", size: 0.5 },
    catGridLine: { style: "none" },
  });

  const cards = [
    ["Heterogeneous relations", "+15.10 pp decisiveness\nvs. Homogeneous Graph MAPPO", BLUE],
    ["Revisit memory", "−16.33 pp decisiveness\n−13.67 pp discovery when removed", ORANGE],
    ["History weighting", "−8.16 pp decisiveness when removed\nT80 88.63→98.19; success 82.0% vs 86.7%", RED],
  ];
  cards.forEach((c, i) => {
    const y = 1.12 + i * 0.85;
    card(s, 6.75, y, 6.25, 0.78, WHITE);
    accent(s, 6.75, y, 0.78);
    s.addText(c[0], { x: 6.95, y: y + 0.04, w: 5.90, h: 0.26, fontFace: "Calibri", fontSize: 13, bold: true, color: c[2], margin: 0 });
    s.addText(c[1], { x: 6.95, y: y + 0.30, w: 5.90, h: 0.42, fontFace: "Calibri", fontSize: 12, color: DARK, margin: 0 });
  });

  contain(s, path.join(A, "fig4.png"), { x: 0.25, y: 3.72, w: 12.80, h: 2.22 });
  s.addText("Do not walk all six panels.  Cite target speed and Rc / outage; mean discovery stays higher in the tested ranges.", {
    x: 0.30, y: 5.96, w: 12.7, h: 0.28, fontFace: "Calibri", fontSize: 12, italic: true, color: BLUE_DK, margin: 0, valign: "middle",
  });
  card(s, 0.28, 6.28, 12.78, 0.68, BLUE_LT);
  s.addText("Trade-off to state: w/o history  T80 success 86.7% vs Full 82.0%, but conditional T80 98.19 vs 88.63 and Ct 53.99 vs 62.15.", {
    x: 0.42, y: 6.28, w: 12.50, h: 0.68, fontFace: "Calibri", fontSize: 13, color: DARK, margin: 0, valign: "middle",
  });
  footer(s, 11);
  notes(s, N[10]);
}

// ---------------------------------------------------------------------------
// 12 Summary
// ---------------------------------------------------------------------------
{
  const s = pres.addSlide();
  header(s, "4. Summary and Future Plan", "Contributions, results, and next steps");
  const cons = [
    ["1. Communication-Constrained Decision Making", "Decentralized policies use only local observations and reachable peer information."],
    ["2. Belief Management", "Revisit-aware P/E/R memory + communication-history-aware source selection."],
    ["3. Heterogeneous Relational Encoding", "HGAT explicitly models peer, task and obstacle relations within MAPPO."],
  ];
  cons.forEach((c, i) => {
    const y = 1.18 + i * 0.92;
    card(s, 0.35, y, 12.62, 0.84, i === 1 ? "FFF3E8" : BLUE_LT);
    accent(s, 0.35, y, 0.84);
    s.addText(c[0], { x: 0.55, y: y + 0.04, w: 12.22, h: 0.32, fontFace: "Calibri", fontSize: 16, bold: true, color: BLUE_DK, margin: 0, valign: "middle" });
    s.addText(c[1], { x: 0.55, y: y + 0.36, w: 12.22, h: 0.40, fontFace: "Calibri", fontSize: 14, color: DARK, margin: 0, valign: "middle" });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.35, y: 3.98, w: 12.62, h: 0.72, fill: { color: BLUE }, rectRadius: 0.08,
  });
  s.addText("62.15%  Ct     ·     86.78%  Dt     ·     82.0%  T80 success     ·     88.63 steps | success", {
    x: 0.50, y: 3.98, w: 12.32, h: 0.72, fontFace: "Calibri", fontSize: 18, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0,
  });
  card(s, 0.35, 4.80, 12.62, 0.62, WHITE);
  s.addText("Key identities:  r_t = lambda_D dD_t + lambda_C dC_t - lambda_safe c_t     ·     attention = softmax(QK/sqrt(d) + log chi - k_r d_ik)", {
    x: 0.50, y: 4.80, w: 12.32, h: 0.62, fontFace: "Calibri", fontSize: 14, color: DARK, align: "center", valign: "middle", margin: 0,
  });
  s.addText("Future Plan", { x: 0.40, y: 5.50, w: 12.5, h: 0.28, fontFace: "Calibri", fontSize: 15, bold: true, color: BLUE, margin: 0 });
  const fut = [
    ["3D environments", "Extend planar search to 3-D UAV kinematics and occupancy"],
    ["Fully decentralized safety", "Remove the simulator-level centralized safety supervisor"],
    ["Realistic validation", "HIL + real radios; keep search (not pursuit) as the core story"],
  ];
  fut.forEach((f, i) => {
    const x = 0.35 + i * 4.32;
    card(s, x, 5.95, 4.15, 0.95, GRAY);
    s.addText(f[0], { x: x + 0.12, y: 6.02, w: 3.90, h: 0.32, fontFace: "Calibri", fontSize: 15, bold: true, color: BLUE, margin: 0, valign: "middle" });
    s.addText(f[1], { x: x + 0.12, y: 6.34, w: 3.90, h: 0.48, fontFace: "Calibri", fontSize: 13, color: DARK, margin: 0, valign: "middle" });
  });
  footer(s, 12);
  notes(s, N[11]);
}

// ---------------------------------------------------------------------------
// 13 Thank you
// ---------------------------------------------------------------------------
{
  const s = pres.addSlide();
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 13.333, h: 7.5, fill: { color: WHITE } });
  contain(s, path.join(A, "hust.png"), { x: 0.40, y: 0.22, w: 1.55, h: 1.08 });
  contain(s, path.join(A, "caa.png"), { x: 11.05, y: 0.26, w: 1.85, h: 1.05 });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 1.50, w: 13.333, h: 3.15, fill: { color: BLUE } });
  s.addText("Thank you for your attention!", {
    x: 0.50, y: 2.05, w: 12.3, h: 1.20,
    fontFace: "Calibri", fontSize: 40, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0,
  });
  s.addText("A Heterogeneous Graph-Attention MAPPO Algorithm for\nCooperative Multi-UAV Search under Intermittent Communication Constraints", {
    x: 0.70, y: 3.28, w: 11.9, h: 1.10,
    fontFace: "Calibri", fontSize: 18, color: "D6E2F5", align: "center", valign: "middle", margin: 0,
  });
  s.addText("Yinying Kuang\nHuazhong University of Science and Technology\nkuangyinying039@163.com", {
    x: 0.50, y: 4.82, w: 12.3, h: 1.15,
    fontFace: "Calibri", fontSize: 18, color: DARK, align: "center", valign: "middle", margin: 0,
  });
  const qchips = ["Dec-POMDP / N_i(t)", "Log-odds P, E, R", "Source j*", "HGAT attention", "MAPPO clip + GAE"];
  qchips.forEach((c, i) => {
    const x = 0.38 + i * 2.58;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y: 6.08, w: 2.48, h: 0.42, fill: { color: BLUE_LT }, rectRadius: 0.08,
    });
    s.addText(c, {
      x, y: 6.08, w: 2.48, h: 0.42,
      fontFace: "Calibri", fontSize: 12, bold: true, color: BLUE_DK, align: "center", valign: "middle", margin: 0,
    });
  });
  s.addText("CAC 2026  ·  October 2026  ·  Welcome questions on beliefs, source selection, and HGAT", {
    x: 0.50, y: 6.60, w: 12.3, h: 0.40,
    fontFace: "Calibri", fontSize: 15, color: BLUE, align: "center", bold: true, valign: "middle", margin: 0,
  });
  notes(s, N[12]);
}

const out = "/workspace/presentations/IC-HGAT-MAPPO_CAC2026.pptx";
pres.writeFile({ fileName: out }).then(() => {
  console.log("wrote", out);
});
