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

function eqBox(slide, token, x, y, w, h) {
  slide.addText(token, {
    x, y, w, h,
    fontFace: "Cambria Math", fontSize: 20, color: DARK,
    align: "center", valign: "middle", margin: 0,
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
  slide.addText(String(n), {
    x: 12.55, y: 7.12, w: 0.55, h: 0.28,
    fontFace: "Calibri", fontSize: 12, color: BLUE, align: "right", margin: 0,
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
各位老师、同学，大家好。今天汇报的工作是间歇通信约束下的多无人机协同搜索。请先记住问题本身：不是默认全局信息可共享，而是每架无人机只能使用机载观测，以及当前物理上可达的邻居信息。算法名称 IC-HGAT-MAPPO 后面再给出。
【English】
Good morning. This talk studies cooperative multi-UAV search under intermittent communication. Please keep the problem first: we do not assume a shared global map. Each UAV may use only onboard observations and currently reachable peer messages. The name IC-HGAT-MAPPO comes after the problem is clear.`,

`【中文】
报告分三部分，大约十几秒带过即可。第一部分讲背景与问题；第二部分讲提出的 IC-HGAT-MAPPO；第三部分给出实验与结论。
【English】
The talk has three parts. I will introduce the problem and motivation, then the proposed IC-HGAT-MAPPO framework, and finally the experimental results. This slide takes about ten to fifteen seconds.`,

`【中文】
多无人机可以并行覆盖震后搜救、林火监测和海上救援等场景。很多 MARL 方法默认通信充分、信息接近全局。但在干扰环境下链路会间歇中断，无人机实际上只能依赖局部观测和当前仍连通的邻居。因此问题不只是“怎么搜索”，而是“信息不完整时如何协同搜索”。左侧是常规设定，中间是更现实的设定，右侧三条研究需求对应后文三个模块。
【English】
Multi-UAV teams can search in earthquake response, wildfire monitoring, and maritime rescue. Many MARL methods assume stable communication and nearly global information. Under interference, links drop, so each UAV has only local observations and currently reachable neighbors. The question is not only how to search, but how to coordinate under incomplete information. The three research needs on the right map to the three designs that follow.`,

`【中文】
间歇通信带来三个核心挑战。第一，没有共享全局地图，可用邻居集合随时间变化，决策必须去中心化。第二，无人机、目标或前沿、障碍物角色不同，不能只用同构 UAV 图。第三，通信恢复后拿到的信念可能过时，不能默认“能连上就可信”。请注意：我们优化的是通信约束下的决策，而不是通信协议本身。
【English】
Three challenges follow. First, there is no shared global map, and the reachable set changes over time, so decisions are decentralized. Second, UAVs, targets or frontiers, and obstacles play different roles, so a homogeneous UAV graph is not enough. Third, beliefs recovered after an outage may be stale, so reachability is not the same as usefulness. Our focus is decision-making under communication constraints, not protocol optimization.`,

`【中文】
这一页完成转折：三个挑战分别由三个机制回答。不完整信念用 revisit-aware P/E/R 记忆；过时邻居信息用通信历史感知的源选择；异构实体用异构图注意力。学习框架是共享 Actor、训练期集中 Critic、MAPPO/CTDE；底部的安全投影是所有方法共用的，不是本文贡献。三个问题汇到 IC-HGAT-MAPPO。
【English】
This slide is the turning point. Incomplete beliefs are handled by revisit-aware P/E/R memory; stale peer information by communication-history-aware source selection; heterogeneous entities by heterogeneous graph attention. Training uses a shared actor, a centralized critic, and MAPPO with CTDE. The safety projection is shared by all methods and is not a contribution. These three questions lead to IC-HGAT-MAPPO.`,

`【中文】
问题设定对应论文第二节。左侧是搜索场景：局部感知、实线连通、虚线断链、运动目标和动态障碍。仿真是 6 架无人机、15 个目标、10 个障碍、30×30 区域，动作为八方向加悬停。运动是带可行域投影的离散更新。最关键的一句：策略只能使用瞬时可达链路。评价指标三个：发现率 Dt、信念决断度 Ct、到达 80% 发现的时间 T80。Dt、Ct 越高越好，T80 越低越快。
【English】
This is the problem formulation. The figure shows local sensing, live and broken links, moving targets, and dynamic obstacles. The setting has 6 UAVs, 15 targets, 10 obstacles, and a 30 by 30 map, with eight move directions plus hover. Motion is discrete with a feasible-set projection. The key sentence is that only instantaneous reachable links are available. We report discovery Dt, belief decisiveness Ct, and T80. Higher Dt and Ct are better; lower T80 is faster.`,

`【中文】
请按箭头从左到右讲，不要跳模块。执行时：局部观测与链路掩码进入 P/E/R 记忆，再进入异构图，共享 Actor 给出名义动作，再经安全投影执行。训练时集中 Critic 看到全局状态，MAPPO 更新参数后写回 Actor。执行阶段去掉 Critic。务必说明：安全投影对所有对比方法相同，性能不能记在 safety layer 上。
【English】
Please follow the arrows from left to right. At execution, local observation and the link mask update P/E/R memory, then the heterogeneous graph, then the shared actor, then safety projection. During training, a centralized critic sees global state and MAPPO writes the update back to the actor. The critic is removed at execution. Safety projection is shared by all methods, so it is not the source of the gains.`,

`【中文】
左栏回答“记什么”。每架无人机维护目标存在信念 P、不确定性 E 和回访陈旧度 R。被感知格子重置为 0，未感知格子随时间老化并向邻域扩散。右栏回答“信谁”。即使 j 和 k 都能通信，也按距离和通信历史加权，每个格子只选一个源，避免把相关 log-odds 简单相加。
【English】
The left column answers what to remember: target-existence P, uncertainty E, and revisit staleness R. Sensed cells reset to zero; unsensed cells age and diffuse. The right column answers whom to trust. Even if both j and k are reachable, weights use distance and communication history, and one source is chosen per cell. We do not add correlated log-odds.`,

`【中文】
异构图以本机 UAV 为中心，分别连同伴、任务和障碍，再做关系专用注意力与语义融合，得到嵌入。注意力由 QK、关系置信、距离先验和链路掩码共同决定，这是相对普通 GAT 的差别。右侧是共享 Actor 输出 9 个名义动作；单一集中 Critic 只在训练使用。奖励是发现增量加决断度增量，减去安全代价，权重 1.0、0.30、0.10。完整 PPO 公式不必展开。
【English】
The graph is centered on UAV i, with peer, task, and obstacle relations, then relation-specific attention and semantic fusion. Attention uses the QK term, relation confidence, a distance prior, and the link mask. That is the difference from a plain GAT. The shared actor outputs nine nominal actions; one centralized critic is used only in training. The reward is discovery plus decisiveness minus a safety cost, with weights 1.0, 0.30, and 0.10. The full PPO derivation can wait for questions.`,

`【中文】
实验协议公平：同一环境、奖励、动作空间和训练预算；5 个独立策略，30 个保留场景。左下轨迹图只说明系统长什么样，不作为定量证据。主结果请看柱状图而不是表格：决断度 62.15%，发现率 86.78%。相对启发式分别提高 17.16 和 9.00 个百分点，相对 QMIX 提高 32.15 和 14.67 个百分点。T80 成功率为 82.0%，成功条件下 88.63 步。安全干预仅 0.17%，碰撞与启发式相当，不要把安全列说成本文优势。
【English】
The protocol is shared: same environment, reward, action space, and budget; five independently trained policies and thirty held-out scenarios. The trajectory figure shows what a run looks like; it is not the quantitative evidence. Please read the bars, not a table: 62.15 percent decisiveness and 86.78 percent discovery. Versus the heuristic that is plus 17.16 and plus 9.00 points; versus QMIX, plus 32.15 and plus 14.67. T80 succeeds in 82.0 percent of runs at 88.63 steps when successful. Safety intervention is only 0.17 percent, and collision is comparable to the heuristic, so do not rank the method on safety columns.`,

`【中文】
消融说明三个模块都有用。相对同构图，异构关系带来约 15.10 个百分点决断度。去掉 revisit 记忆，决断度和发现率分别下降 16.33 和 13.67 个百分点。去掉历史加权，决断度下降 8.16，条件 T80 从 88.63 变到 98.19，但阈值成功率从 82.0% 升到 86.7%，这是需要主动承认的权衡。下方敏感性不必六张图逐张讲，举目标速度和通信半径或中断概率两例，然后说在测试扰动范围内平均发现率更高。
【English】
Ablations support all three modules. Heterogeneous relations add about 15.10 points of decisiveness over a homogeneous graph. Removing revisit memory drops decisiveness and discovery by 16.33 and 13.67 points. Removing history weighting drops decisiveness by 8.16 points and slows conditional T80 from 88.63 to 98.19, while threshold success rises from 82.0 to 86.7 percent; that trade-off should be stated. Do not walk through all six sensitivity plots. Mention target speed and communication radius or outage, then conclude that mean discovery stays higher over the tested ranges.`,

`【中文】
总结只留三条：通信约束下的去中心决策、P/E/R 与历史感知选源的信念管理、异构关系编码。请再念一遍核心数字：62.15%、86.78%、82.0%，成功条件下 88.63 步。未来工作按论文来：三维环境、完全去中心安全、以及含真实通信的半实物验证。如果被问到仓库里的围捕代码，只作为口头补充，不要把这篇 search 报告讲成 pursuit。
【English】
Leave three contributions: communication-constrained decentralized decisions; belief management with P/E/R and history-aware selection; and heterogeneous relational encoding. Repeat 62.15 percent, 86.78 percent, 82.0 percent, and 88.63 conditional steps. Future work follows the paper: 3D environments, fully decentralized safety, and hardware-in-the-loop with real communication. If asked about pursuit code in the repo, mention it only as an ongoing extension. Do not turn this search talk into a pursuit talk.`,

`【中文】
谢谢各位。欢迎就间歇通信下的信念维护、源选择和异构图提问。联系邮箱是 kuangyinying039@163.com。
【English】
Thank you for your attention. I welcome questions on belief maintenance, source selection, and heterogeneous graphs under intermittent communication. My email is kuangyinying039@163.com.`,
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
    x: 0.5, y: 6.40, w: 12.3, h: 0.42,
    fontFace: "Calibri", fontSize: 18, color: BLUE, align: "center", bold: true, valign: "middle", margin: 0,
  });
  notes(s, N[0]);
}

// ---------------------------------------------------------------------------
// 2 CONTENTS
// ---------------------------------------------------------------------------
{
  const s = pres.addSlide();
  contain(s, path.join(A, "hust.png"), { x: 12.38, y: 0.10, w: 0.82, h: 0.54 });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0.72, w: 13.333, h: 0.14, fill: { color: BLUE } });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: -0.2, y: 2.45, w: 4.85, h: 2.05, fill: { color: BLUE }, rectRadius: 0.95,
  });
  s.addText("CONTENTS", {
    x: 0.20, y: 2.85, w: 4.15, h: 1.25,
    fontFace: "Calibri", fontSize: 36, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0,
  });
  s.addShape(pres.shapes.RECTANGLE, { x: 5.05, y: 1.70, w: 0.07, h: 4.2, fill: { color: BLUE } });
  const items = [
    ["01", "Background and Problem Formulation"],
    ["02", "Proposed IC-HGAT-MAPPO"],
    ["03", "Experimental Results and Conclusions"],
  ];
  items.forEach((it, i) => {
    const y = 2.00 + i * 1.15;
    s.addText(it[0], { x: 5.45, y, w: 0.90, h: 0.62, fontFace: "Calibri", fontSize: 30, bold: true, color: BLUE, margin: 0, valign: "middle" });
    s.addText(it[1], { x: 6.40, y, w: 6.4, h: 0.78, fontFace: "Calibri", fontSize: 22, color: DARK, valign: "middle", margin: 0 });
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
    contain(s, ph[0], { x, y: 1.28, w: 4.05, h: 2.15 });
    s.addText(ph[1], {
      x, y: 3.46, w: 4.05, h: 0.38,
      fontFace: "Calibri", fontSize: 16, bold: true, color: DARK, align: "center", valign: "middle", margin: 0,
    });
  });
  // transition bar
  s.addShape(pres.shapes.RECTANGLE, { x: 0.40, y: 3.80, w: 12.55, h: 0.48, fill: { color: GRAY } });
  s.addText("Global / Reliable Information", {
    x: 0.50, y: 3.88, w: 5.3, h: 0.48,
    fontFace: "Calibri", fontSize: 15, color: MUTED, align: "right", valign: "middle", margin: 0,
  });
  s.addText("  →   Local Observation + Intermittent Communication", {
    x: 5.85, y: 3.88, w: 7.0, h: 0.48,
    fontFace: "Calibri", fontSize: 15, color: RED, bold: true, valign: "middle", margin: 0,
  });
  s.addText("More realistic but more challenging", {
    x: 0.40, y: 4.36, w: 12.55, h: 0.32,
    fontFace: "Calibri", fontSize: 14, italic: true, color: MUTED, align: "center", margin: 0,
  });

  const cols = [
    ["Conventional Setting", "Stable communication\nSynchronized / global information\nHomogeneous interaction modeling", BLUE_LT, BLUE],
    ["Practical Setting", "Intermittent links\nLocal and incomplete beliefs\nUAV / target / obstacle interactions", "FFF3E8", ORANGE],
    ["Research Needs", "1. Maintain useful local beliefs\n2. Exploit reachable peer information\n3. Model heterogeneous relations", "FDECEA", RED],
  ];
  cols.forEach((c, i) => {
    const x = 0.40 + i * 4.30;
    card(s, x, 4.62, 4.05, 2.20, c[2]);
    s.addText(c[0], {
      x: x + 0.15, y: 4.72, w: 3.75, h: 0.42,
      fontFace: "Calibri", fontSize: 17, bold: true, color: c[3], margin: 0, valign: "middle",
    });
    s.addText(c[1], {
      x: x + 0.15, y: 5.16, w: 3.75, h: 1.50,
      fontFace: "Calibri", fontSize: 15, color: DARK, margin: 0, valign: "middle",
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
    "No shared global map\nOnly reachable peer messages\nDecentralized decisions",
    "Peer cooperation\nSearch utility\nCollision risk",
    "Recovered information may be outdated\nNot all reachable beliefs are equally useful\n\nWhich source should be trusted?",
  ];
  titles.forEach((t, i) => {
    const x = 0.35 + i * 4.32;
    card(s, x, 1.25, 4.15, 4.55, WHITE);
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.25, w: 4.15, h: 0.46, fill: { color: i === 2 ? RED : BLUE } });
    s.addText(t[0], {
      x: x + 0.10, y: 1.28, w: 3.95, h: 0.40,
      fontFace: "Calibri", fontSize: 14, bold: true, color: WHITE, margin: 0, valign: "middle",
    });
    contain(s, path.join(A, t[1]), { x: x + 0.12, y: 1.78, w: 3.90, h: 2.20 });
    s.addText(bodies[i], {
      x: x + 0.18, y: 4.02, w: 3.80, h: 1.60,
      fontFace: "Calibri", fontSize: 15, color: DARK, margin: 0, valign: "middle",
    });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.35, y: 5.95, w: 12.62, h: 0.95, fill: { color: BLUE_LT }, rectRadius: 0.08,
  });
  s.addText("Our focus: cooperative decision-making under communication constraints,\nrather than communication-protocol optimization.", {
    x: 0.55, y: 6.05, w: 12.25, h: 0.75,
    fontFace: "Calibri", fontSize: 18, bold: true, color: BLUE_DK, align: "center", valign: "middle", margin: 0,
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
    card(s, x, 1.28, 3.00, 2.85, WHITE);
    s.addText(m[0], { x, y: 1.38, w: 3.00, h: 0.36, fontFace: "Calibri", fontSize: 14, color: MUTED, align: "center", margin: 0, valign: "middle" });
    s.addText(m[1], { x: x + 0.08, y: 1.74, w: 2.84, h: 0.58, fontFace: "Calibri", fontSize: 16, bold: true, color: DARK, align: "center", margin: 0, valign: "middle" });
    s.addShape(pres.shapes.DOWN_ARROW, { x: x + 1.15, y: 2.28, w: 0.70, h: 0.38, fill: { color: m[3] } });
    s.addText("Solution", { x, y: 2.68, w: 3.00, h: 0.28, fontFace: "Calibri", fontSize: 13, color: MUTED, align: "center", margin: 0 });
    s.addText(m[2], { x: x + 0.08, y: 2.95, w: 2.84, h: 0.95, fontFace: "Calibri", fontSize: 16, bold: true, color: m[3], align: "center", margin: 0, valign: "middle" });
  });
  card(s, 9.85, 1.28, 3.12, 2.85, BLUE);
  s.addText("Learning Framework", {
    x: 10.00, y: 1.42, w: 2.85, h: 0.48, fontFace: "Calibri", fontSize: 16, bold: true, color: WHITE, margin: 0, valign: "middle",
  });
  s.addText("Shared Actor\nCentralized Critic\nMAPPO / CTDE\nCommon Safety Projection", {
    x: 10.00, y: 1.95, w: 2.85, h: 1.95, fontFace: "Calibri", fontSize: 16, color: WHITE, margin: 0, valign: "middle",
  });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.35, y: 4.30, w: 12.62, h: 2.55, fill: { color: BLUE_LT }, rectRadius: 0.08,
  });
  s.addText("Our problem: Cooperative Multi-UAV Search under Intermittent Communication Constraints", {
    x: 0.55, y: 4.42, w: 12.22, h: 0.44, fontFace: "Calibri", fontSize: 17, bold: true, color: BLUE_DK, margin: 0, valign: "middle",
  });
  const qs = [
    "Q1. How to maintain useful beliefs without global map sharing?",
    "Q2. How to select useful information after communication recovery?",
    "Q3. How to model UAV, task, and obstacle interactions jointly?",
  ];
  qs.forEach((q, i) => {
    s.addText(q, {
      x: 0.60, y: 4.88 + i * 0.38, w: 8.6, h: 0.36,
      fontFace: "Calibri", fontSize: 16, color: RED, bold: true, margin: 0, valign: "middle",
    });
  });
  s.addShape(pres.shapes.RIGHT_ARROW, { x: 9.35, y: 5.35, w: 0.55, h: 0.32, fill: { color: BLUE } });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 10.00, y: 5.05, w: 2.75, h: 1.15, fill: { color: BLUE }, rectRadius: 0.08,
  });
  s.addText("IC-HGAT-\nMAPPO", {
    x: 10.05, y: 5.15, w: 2.65, h: 0.95,
    fontFace: "Calibri", fontSize: 18, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0,
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
  contain(s, path.join(A, "fig1.png"), { x: 0.30, y: 1.28, w: 6.55, h: 3.55 });
  s.addText("Fig. 1  Local sensing, intermittent links, moving targets, dynamic obstacles", {
    x: 0.30, y: 4.88, w: 6.55, h: 0.32, fontFace: "Calibri", fontSize: 12, italic: true, color: MUTED, margin: 0, valign: "middle",
  });
  card(s, 0.30, 5.28, 6.55, 1.55, GRAY);
  s.addText("Scenario", { x: 0.45, y: 5.34, w: 6.25, h: 0.32, fontFace: "Calibri", fontSize: 16, bold: true, color: BLUE, margin: 0, valign: "middle" });
  s.addText("6 UAVs · 15 targets · 10 obstacles · 30×30 area\nLocal sensing + intermittent links\nAction: 8 directions + hover", {
    x: 0.45, y: 5.66, w: 6.25, h: 1.05, fontFace: "Calibri", fontSize: 15, color: DARK, margin: 0, valign: "middle",
  });

  card(s, 7.05, 1.22, 5.95, 1.55, WHITE);
  s.addText("Motion model", { x: 7.20, y: 1.28, w: 5.65, h: 0.32, fontFace: "Calibri", fontSize: 16, bold: true, color: BLUE, margin: 0, valign: "middle" });
  eqBox(s, "EQMOTION", 7.12, 1.58, 5.78, 0.70);
  s.addText("Discrete motion with adaptive decision interval", {
    x: 7.20, y: 2.28, w: 5.65, h: 0.36, fontFace: "Calibri", fontSize: 14, italic: true, color: MUTED, margin: 0, valign: "middle",
  });

  card(s, 7.05, 2.90, 5.95, 1.85, WHITE);
  s.addText("Communication constraint", { x: 7.20, y: 2.96, w: 5.65, h: 0.32, fontFace: "Calibri", fontSize: 16, bold: true, color: BLUE, margin: 0, valign: "middle" });
  eqBox(s, "EQLINK", 7.08, 3.26, 5.85, 0.72);
  s.addText("Only instantaneous reachable links are available", {
    x: 7.20, y: 4.00, w: 5.65, h: 0.58, fontFace: "Calibri", fontSize: 16, bold: true, color: RED, margin: 0, valign: "middle",
  });

  card(s, 7.05, 4.90, 5.95, 1.92, BLUE_LT);
  s.addText("Evaluation objectives", { x: 7.20, y: 4.98, w: 5.65, h: 0.32, fontFace: "Calibri", fontSize: 16, bold: true, color: BLUE, margin: 0, valign: "middle" });
  s.addText("Target discovery Dt     Belief decisiveness Ct     Speed T80", {
    x: 7.20, y: 5.32, w: 5.65, h: 0.42, fontFace: "Calibri", fontSize: 15, color: DARK, margin: 0, valign: "middle",
  });
  s.addText("Higher Dt, Ct  →  better     ·     Lower T80  →  faster", {
    x: 7.20, y: 5.80, w: 5.65, h: 0.70, fontFace: "Calibri", fontSize: 16, bold: true, color: BLUE_DK, margin: 0, valign: "middle",
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
  contain(s, path.join(A, "framework.png"), { x: 0.25, y: 1.26, w: 12.85, h: 3.70 });
  const feats = [
    ["Decentralized", "local + reachable information only"],
    ["Communication-aware", "instantaneous link masking"],
    ["CTDE", "global state only used during training"],
  ];
  feats.forEach((f, i) => {
    const x = 0.35 + i * 4.32;
    card(s, x, 5.12, 4.15, 1.05, i === 1 ? "FFF3E8" : BLUE_LT);
    s.addText(f[0], { x: x + 0.15, y: 5.18, w: 3.85, h: 0.36, fontFace: "Calibri", fontSize: 16, bold: true, color: BLUE_DK, margin: 0, valign: "middle" });
    s.addText(f[1], { x: x + 0.15, y: 5.52, w: 3.85, h: 0.52, fontFace: "Calibri", fontSize: 14, color: DARK, margin: 0, valign: "middle" });
  });
  s.addText("Safety projection is shared by all methods and is not a contribution.", {
    x: 0.35, y: 6.28, w: 12.6, h: 0.36, fontFace: "Calibri", fontSize: 14, italic: true, color: MUTED, margin: 0, valign: "middle",
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
  card(s, 0.28, 1.20, 6.45, 5.55, WHITE);
  s.addText("Revisit-Aware P/E/R Memory", {
    x: 0.42, y: 1.28, w: 6.15, h: 0.42, fontFace: "Calibri", fontSize: 18, bold: true, color: BLUE, margin: 0, valign: "middle",
  });
  contain(s, path.join(A, "per_maps.png"), { x: 0.40, y: 1.72, w: 6.20, h: 2.05 });
  eqBox(s, "EQP", 0.40, 3.80, 6.15, 0.72);
  s.addText("R  ←  aging  +  spatial diffusion", {
    x: 0.50, y: 4.54, w: 6.05, h: 0.40, fontFace: "Calibri", fontSize: 17, color: DARK, margin: 0, valign: "middle",
  });
  s.addText("sensed cells  →  reset to 0\nunsensed cells  →  gradually become stale", {
    x: 0.50, y: 4.98, w: 6.05, h: 1.45, fontFace: "Calibri", fontSize: 16, color: MUTED, margin: 0, valign: "middle",
  });

  card(s, 6.90, 1.20, 6.12, 5.55, WHITE);
  s.addText("Communication-History-Aware Source Selection", {
    x: 7.05, y: 1.28, w: 5.82, h: 0.48, fontFace: "Calibri", fontSize: 16, bold: true, color: ORANGE, margin: 0, valign: "middle",
  });
  contain(s, path.join(A, "source.png"), { x: 7.05, y: 1.78, w: 5.80, h: 2.20 });
  eqBox(s, "EQW", 7.00, 4.02, 5.85, 0.58);
  eqBox(s, "EQJSTAR", 7.00, 4.58, 5.85, 0.55);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 7.10, y: 5.30, w: 5.70, h: 1.20, fill: { color: "FDECEA" }, rectRadius: 0.08,
  });
  s.addText("One reachable source is selected per cell", {
    x: 7.25, y: 5.38, w: 5.40, h: 0.46, fontFace: "Calibri", fontSize: 16, bold: true, color: RED, margin: 0, valign: "middle",
  });
  s.addText("Avoid correlated evidence accumulation", {
    x: 7.25, y: 5.84, w: 5.40, h: 0.50, fontFace: "Calibri", fontSize: 15, color: DARK, margin: 0, valign: "middle",
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
  contain(s, path.join(A, "hgat.png"), { x: 0.20, y: 1.26, w: 6.55, h: 3.35 });
  eqBox(s, "EQATT", 0.20, 4.62, 6.55, 0.78);
  s.addText("Relation confidence     ·     Distance prior     ·     Link mask", {
    x: 0.28, y: 5.42, w: 6.45, h: 0.38, fontFace: "Calibri", fontSize: 14, color: BLUE, bold: true, margin: 0, valign: "middle",
  });
  contain(s, path.join(A, "actor_critic.png"), { x: 6.85, y: 1.26, w: 6.15, h: 3.35 });
  eqBox(s, "EQREWARD", 6.90, 4.62, 6.10, 0.78);
  s.addText("(λD, λC, λsafe) = (1.0, 0.30, 0.10)     ·     MAPPO: PPO clipping + GAE", {
    x: 6.95, y: 5.42, w: 6.05, h: 0.42, fontFace: "Calibri", fontSize: 13, color: DARK, margin: 0, valign: "middle",
  });
  s.addText("Heterogeneous Graph Attention", {
    x: 0.28, y: 5.88, w: 6.4, h: 0.42, fontFace: "Calibri", fontSize: 18, bold: true, color: GREEN, margin: 0, valign: "middle",
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
  card(s, 0.28, 1.18, 5.55, 2.15, GRAY);
  s.addText("Experiment setup", { x: 0.42, y: 1.24, w: 5.25, h: 0.34, fontFace: "Calibri", fontSize: 16, bold: true, color: BLUE, margin: 0, valign: "middle" });
  s.addText("6 UAVs  ·  15 targets  ·  10 obstacles  ·  30×30 grid\nHorizon = 200     ·     2,000 training episodes\n5 independently trained policies\n30 held-out scenarios", {
    x: 0.42, y: 1.58, w: 5.25, h: 1.20, fontFace: "Calibri", fontSize: 15, color: DARK, margin: 0,
  });
  s.addText("Common environment, reward, action space and training budget", {
    x: 0.42, y: 2.82, w: 5.25, h: 0.40, fontFace: "Calibri", fontSize: 13, italic: true, color: MUTED, margin: 0, valign: "middle",
  });
  contain(s, path.join(A, "fig3.png"), { x: 0.28, y: 3.40, w: 5.55, h: 2.90 });
  s.addText("UAV trajectories  ·  discovered / undiscovered targets  ·  dynamic obstacles", {
    x: 0.28, y: 6.32, w: 5.55, h: 0.32, fontFace: "Calibri", fontSize: 12, italic: true, color: MUTED, margin: 0, valign: "middle",
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
    x: 0.20, y: 1.12, w: 6.55, h: 2.85,
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
    const y = 1.12 + i * 0.95;
    card(s, 6.90, y, 6.10, 0.88, WHITE);
    s.addText(c[0], { x: 7.08, y: y + 0.06, w: 5.75, h: 0.28, fontFace: "Calibri", fontSize: 13, bold: true, color: c[2], margin: 0 });
    s.addText(c[1], { x: 7.08, y: y + 0.34, w: 5.75, h: 0.48, fontFace: "Calibri", fontSize: 12, color: DARK, margin: 0 });
  });

  contain(s, path.join(A, "fig4.png"), { x: 0.25, y: 4.15, w: 12.80, h: 2.48 });
  s.addText("IC-HGAT-MAPPO maintains higher mean discovery across the tested perturbation ranges.", {
    x: 0.30, y: 6.66, w: 12.7, h: 0.32, fontFace: "Calibri", fontSize: 14, italic: true, color: BLUE_DK, margin: 0, valign: "middle",
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
    const y = 1.22 + i * 1.05;
    card(s, 0.35, y, 12.62, 0.95, i === 1 ? "FFF3E8" : BLUE_LT);
    s.addText(c[0], { x: 0.55, y: y + 0.08, w: 12.22, h: 0.36, fontFace: "Calibri", fontSize: 17, bold: true, color: BLUE_DK, margin: 0, valign: "middle" });
    s.addText(c[1], { x: 0.55, y: y + 0.44, w: 12.22, h: 0.42, fontFace: "Calibri", fontSize: 16, color: DARK, margin: 0, valign: "middle" });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.35, y: 4.42, w: 12.62, h: 1.05, fill: { color: BLUE }, rectRadius: 0.08,
  });
  s.addText("62.15% belief decisiveness   |   86.78% discovery   |   82.0% T80 success", {
    x: 0.50, y: 4.48, w: 12.32, h: 0.50, fontFace: "Calibri", fontSize: 20, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0,
  });
  s.addText("88.63 steps conditional T80", {
    x: 0.50, y: 4.95, w: 12.32, h: 0.40, fontFace: "Calibri", fontSize: 16, color: "D6E2F5", align: "center", valign: "middle", margin: 0,
  });
  s.addText("Future Plan", { x: 0.40, y: 5.58, w: 12.5, h: 0.32, fontFace: "Calibri", fontSize: 16, bold: true, color: BLUE, margin: 0 });
  const fut = [
    ["3D environments", "Extend planar search to three-dimensional UAV scenarios"],
    ["Fully decentralized safety", "Remove the simulator-level centralized safety supervisor"],
    ["Realistic validation", "Hardware-in-the-loop and real communication systems"],
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
    x: 0.50, y: 5.00, w: 12.3, h: 1.35,
    fontFace: "Calibri", fontSize: 18, color: DARK, align: "center", valign: "middle", margin: 0,
  });
  s.addText("CAC 2026  ·  October 2026", {
    x: 0.50, y: 6.42, w: 12.3, h: 0.40,
    fontFace: "Calibri", fontSize: 16, color: BLUE, align: "center", bold: true, valign: "middle", margin: 0,
  });
  notes(s, N[12]);
}

const out = "/workspace/presentations/IC-HGAT-MAPPO_CAC2026.pptx";
pres.writeFile({ fileName: out }).then(() => {
  console.log("wrote", out);
});
