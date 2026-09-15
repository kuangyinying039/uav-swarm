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

function header(slide, section, subtitle) {
  slide.addImage({ path: path.join(A, "hust.png"), x: 12.42, y: 0.08, w: 0.72, h: 0.50 });
  slide.addText(section, {
    x: 0.38, y: 0.10, w: 11.8, h: 0.42,
    fontFace: "Calibri", fontSize: 22, bold: true, color: BLUE, margin: 0,
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0.62, w: 13.333, h: 0.46, fill: { color: BLUE },
  });
  slide.addText(subtitle, {
    x: 0.38, y: 0.64, w: 12.5, h: 0.40,
    fontFace: "Calibri", fontSize: 16, bold: true, color: WHITE, margin: 0,
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

// ---------------------------------------------------------------------------
// 1 Cover
// ---------------------------------------------------------------------------
{
  const s = pres.addSlide();
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 13.333, h: 7.5, fill: { color: WHITE } });
  s.addImage({ path: path.join(A, "hust.png"), x: 0.45, y: 0.28, w: 1.55, h: 1.07 });
  s.addImage({ path: path.join(A, "caa.png"), x: 5.75, y: 0.30, w: 1.85, h: 1.02 });
  s.addText("Chinese Automation Congress  ·  CAC 2026", {
    x: 7.70, y: 0.48, w: 5.1, h: 0.70,
    fontFace: "Calibri", fontSize: 14, color: BLUE, bold: true, margin: 0, valign: "middle",
  });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 1.55, w: 13.333, h: 3.35, fill: { color: BLUE } });
  s.addText("A Heterogeneous Graph-Attention MAPPO Algorithm\nfor Cooperative Multi-UAV Search\nunder Intermittent Communication Constraints", {
    x: 0.55, y: 1.75, w: 12.2, h: 2.95,
    fontFace: "Calibri", fontSize: 26, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0,
  });
  s.addText("Yinying Kuang, Zhaowei Liang, Yongran Zhi, Huijin Fan*, Lei Liu*, Bo Wang*", {
    x: 0.5, y: 5.10, w: 12.3, h: 0.42,
    fontFace: "Calibri", fontSize: 16, color: DARK, align: "center", margin: 0,
  });
  s.addText("Huazhong University of Science and Technology\nWuhan Second Ship Design and Research Institute", {
    x: 0.5, y: 5.55, w: 12.3, h: 0.70,
    fontFace: "Calibri", fontSize: 15, color: MUTED, align: "center", margin: 0,
  });
  s.addText("CAC 2026  ·  Beijing  ·  October 2026", {
    x: 0.5, y: 6.40, w: 12.3, h: 0.36,
    fontFace: "Calibri", fontSize: 15, color: BLUE, align: "center", bold: true, margin: 0,
  });
  notes(s, "Cover. State the problem first: cooperative multi-UAV search under intermittent communication. Do not lead with the acronym IC-HGAT-MAPPO.");
}

// ---------------------------------------------------------------------------
// 2 CONTENTS
// ---------------------------------------------------------------------------
{
  const s = pres.addSlide();
  s.addImage({ path: path.join(A, "hust.png"), x: 12.42, y: 0.12, w: 0.72, h: 0.50 });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0.72, w: 13.333, h: 0.14, fill: { color: BLUE } });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: -0.2, y: 2.55, w: 4.7, h: 1.85, fill: { color: BLUE }, rectRadius: 0.9,
  });
  s.addText("CONTENTS", {
    x: 0.25, y: 3.00, w: 3.9, h: 0.95,
    fontFace: "Calibri", fontSize: 32, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0,
  });
  s.addShape(pres.shapes.RECTANGLE, { x: 5.05, y: 1.70, w: 0.07, h: 4.2, fill: { color: BLUE } });
  const items = [
    ["01", "Background and Problem Formulation"],
    ["02", "Proposed IC-HGAT-MAPPO"],
    ["03", "Experimental Results and Conclusions"],
  ];
  items.forEach((it, i) => {
    const y = 2.00 + i * 1.15;
    s.addText(it[0], { x: 5.45, y, w: 0.85, h: 0.55, fontFace: "Calibri", fontSize: 26, bold: true, color: BLUE, margin: 0 });
    s.addText(it[1], { x: 6.35, y, w: 6.4, h: 0.70, fontFace: "Calibri", fontSize: 20, color: DARK, valign: "middle", margin: 0 });
  });
  footer(s, 2);
  notes(s, "I will introduce our work in three parts: the problem and motivation, the proposed IC-HGAT-MAPPO framework, and the experimental results. 10–15 seconds.");
}

// ---------------------------------------------------------------------------
// 3 Background applications
// ---------------------------------------------------------------------------
{
  const s = pres.addSlide();
  header(s, "1. Background", "1.1 Cooperative Multi-UAV Search");
  const photos = [
    [path.join(P, "earthquake.jpg"), "Earthquake Response"],
    [path.join(P, "wildfire_fallback.jpg"), "Wildfire Monitoring"],
    [path.join(P, "maritime.jpg"), "Maritime Rescue"],
  ];
  photos.forEach((ph, i) => {
    const x = 0.40 + i * 4.30;
    s.addImage({ path: ph[0], x, y: 1.22, w: 4.05, h: 2.15 });
    s.addText(ph[1], {
      x, y: 3.40, w: 4.05, h: 0.32,
      fontFace: "Calibri", fontSize: 14, bold: true, color: DARK, align: "center", margin: 0,
    });
  });
  // transition bar
  s.addShape(pres.shapes.RECTANGLE, { x: 0.40, y: 3.80, w: 12.55, h: 0.48, fill: { color: GRAY } });
  s.addText("Global / Reliable Information", {
    x: 0.50, y: 3.82, w: 5.3, h: 0.44,
    fontFace: "Calibri", fontSize: 13, color: MUTED, align: "right", valign: "middle", margin: 0,
  });
  s.addText("  →   Local Observation + Intermittent Communication", {
    x: 5.85, y: 3.82, w: 7.0, h: 0.44,
    fontFace: "Calibri", fontSize: 13, color: RED, bold: true, valign: "middle", margin: 0,
  });
  s.addText("More realistic but more challenging", {
    x: 0.40, y: 4.28, w: 12.55, h: 0.28,
    fontFace: "Calibri", fontSize: 12, italic: true, color: MUTED, align: "center", margin: 0,
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
      x: x + 0.15, y: 4.72, w: 3.75, h: 0.38,
      fontFace: "Calibri", fontSize: 15, bold: true, color: c[3], margin: 0,
    });
    s.addText(c[1], {
      x: x + 0.15, y: 5.12, w: 3.75, h: 1.55,
      fontFace: "Calibri", fontSize: 13, color: DARK, margin: 0, valign: "top",
    });
  });
  footer(s, 3);
  notes(s, "Multi-UAV search is useful in disaster settings, but many MARL methods assume sufficient information exchange. Under interference, communication is intermittent, so UAVs can use only onboard observations and currently reachable neighbors. The problem is cooperative search under incomplete information.");
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
      fontFace: "Calibri", fontSize: 12, bold: true, color: WHITE, margin: 0, valign: "middle",
    });
    s.addImage({ path: path.join(A, t[1]), x: x + 0.15, y: 1.80, w: 3.85, h: 2.05 });
    s.addText(bodies[i], {
      x: x + 0.18, y: 3.90, w: 3.80, h: 1.75,
      fontFace: "Calibri", fontSize: 13, color: DARK, margin: 0,
    });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.35, y: 5.95, w: 12.62, h: 0.95, fill: { color: BLUE_LT }, rectRadius: 0.08,
  });
  s.addText("Our focus: cooperative decision-making under communication constraints,\nrather than communication-protocol optimization.", {
    x: 0.55, y: 6.05, w: 12.25, h: 0.75,
    fontFace: "Calibri", fontSize: 16, bold: true, color: BLUE_DK, align: "center", valign: "middle", margin: 0,
  });
  footer(s, 4);
  notes(s, "Three challenges from the paper: incomplete exchange, heterogeneous UAV/task/obstacle roles, and stale information after reconnection. We do not optimize the communication protocol; we constrain the policy to physically available information.");
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
    s.addText(m[0], { x, y: 1.38, w: 3.00, h: 0.32, fontFace: "Calibri", fontSize: 12, color: MUTED, align: "center", margin: 0 });
    s.addText(m[1], { x: x + 0.08, y: 1.70, w: 2.84, h: 0.55, fontFace: "Calibri", fontSize: 13, bold: true, color: DARK, align: "center", margin: 0 });
    s.addShape(pres.shapes.DOWN_ARROW, { x: x + 1.15, y: 2.28, w: 0.70, h: 0.38, fill: { color: m[3] } });
    s.addText("Solution", { x, y: 2.68, w: 3.00, h: 0.26, fontFace: "Calibri", fontSize: 11, color: MUTED, align: "center", margin: 0 });
    s.addText(m[2], { x: x + 0.08, y: 2.95, w: 2.84, h: 0.95, fontFace: "Calibri", fontSize: 14, bold: true, color: m[3], align: "center", margin: 0 });
  });
  card(s, 9.85, 1.28, 3.12, 2.85, BLUE);
  s.addText("Learning Framework", {
    x: 10.00, y: 1.42, w: 2.85, h: 0.40, fontFace: "Calibri", fontSize: 14, bold: true, color: WHITE, margin: 0,
  });
  s.addText("Shared Actor\nCentralized Critic\nMAPPO / CTDE\nCommon Safety Projection", {
    x: 10.00, y: 1.90, w: 2.85, h: 1.95, fontFace: "Calibri", fontSize: 14, color: WHITE, margin: 0,
  });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.35, y: 4.30, w: 12.62, h: 2.55, fill: { color: BLUE_LT }, rectRadius: 0.08,
  });
  s.addText("Our problem: Cooperative Multi-UAV Search under Intermittent Communication Constraints", {
    x: 0.55, y: 4.42, w: 12.22, h: 0.40, fontFace: "Calibri", fontSize: 15, bold: true, color: BLUE_DK, margin: 0,
  });
  const qs = [
    "Q1. How to maintain useful beliefs without global map sharing?",
    "Q2. How to select useful information after communication recovery?",
    "Q3. How to model UAV, task, and obstacle interactions jointly?",
  ];
  qs.forEach((q, i) => {
    s.addText(q, {
      x: 0.60, y: 4.88 + i * 0.38, w: 8.6, h: 0.36,
      fontFace: "Calibri", fontSize: 14, color: RED, bold: true, margin: 0,
    });
  });
  s.addShape(pres.shapes.RIGHT_ARROW, { x: 9.35, y: 5.35, w: 0.55, h: 0.32, fill: { color: BLUE } });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 10.00, y: 5.05, w: 2.75, h: 1.15, fill: { color: BLUE }, rectRadius: 0.08,
  });
  s.addText("IC-HGAT-\nMAPPO", {
    x: 10.05, y: 5.15, w: 2.65, h: 0.95,
    fontFace: "Calibri", fontSize: 16, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0,
  });
  footer(s, 5);
  notes(s, "This slide maps the three challenges to P/E/R memory, history-aware source selection, and heterogeneous graph attention, trained with MAPPO/CTDE. Safety projection is shared and is not a contribution.");
}

// ---------------------------------------------------------------------------
// 6 Problem formulation
// ---------------------------------------------------------------------------
{
  const s = pres.addSlide();
  header(s, "2. Proposed IC-HGAT-MAPPO", "2.1 Problem Formulation");
  s.addImage({ path: path.join(A, "fig1.png"), x: 0.30, y: 1.22, w: 6.55, h: 3.70 });
  s.addText("Fig. 1  Local sensing, intermittent links, moving targets, dynamic obstacles", {
    x: 0.30, y: 4.95, w: 6.55, h: 0.28, fontFace: "Calibri", fontSize: 10, italic: true, color: MUTED, margin: 0,
  });
  card(s, 0.30, 5.28, 6.55, 1.55, GRAY);
  s.addText("Scenario", { x: 0.45, y: 5.34, w: 6.25, h: 0.28, fontFace: "Calibri", fontSize: 13, bold: true, color: BLUE, margin: 0 });
  s.addText("6 UAVs · 15 targets · 10 obstacles · 30×30 area\nLocal sensing + intermittent links\nAction: 8 directions + hover", {
    x: 0.45, y: 5.64, w: 6.25, h: 1.05, fontFace: "Calibri", fontSize: 13, color: DARK, margin: 0,
  });

  card(s, 7.05, 1.22, 5.95, 1.55, WHITE);
  s.addText("Motion model", { x: 7.20, y: 1.28, w: 5.65, h: 0.28, fontFace: "Calibri", fontSize: 13, bold: true, color: BLUE, margin: 0 });
  s.addImage({ path: path.join(A, "eq_motion.png"), x: 7.15, y: 1.55, w: 5.70, h: 0.72 });
  s.addText("Discrete motion with adaptive decision interval", {
    x: 7.20, y: 2.28, w: 5.65, h: 0.32, fontFace: "Calibri", fontSize: 12, italic: true, color: MUTED, margin: 0,
  });

  card(s, 7.05, 2.90, 5.95, 1.85, WHITE);
  s.addText("Communication constraint", { x: 7.20, y: 2.96, w: 5.65, h: 0.28, fontFace: "Calibri", fontSize: 13, bold: true, color: BLUE, margin: 0 });
  s.addImage({ path: path.join(A, "eq_link.png"), x: 7.10, y: 3.24, w: 5.80, h: 0.70 });
  s.addText("Only instantaneous reachable links are available", {
    x: 7.20, y: 4.00, w: 5.65, h: 0.55, fontFace: "Calibri", fontSize: 14, bold: true, color: RED, margin: 0,
  });

  card(s, 7.05, 4.90, 5.95, 1.92, BLUE_LT);
  s.addText("Evaluation objectives", { x: 7.20, y: 4.98, w: 5.65, h: 0.28, fontFace: "Calibri", fontSize: 13, bold: true, color: BLUE, margin: 0 });
  s.addText("Target discovery Dt     Belief decisiveness Ct     Speed T80", {
    x: 7.20, y: 5.32, w: 5.65, h: 0.42, fontFace: "Calibri", fontSize: 13, color: DARK, margin: 0,
  });
  s.addText("Higher Dt, Ct  →  better     ·     Lower T80  →  faster", {
    x: 7.20, y: 5.95, w: 5.65, h: 0.55, fontFace: "Calibri", fontSize: 14, bold: true, color: BLUE_DK, margin: 0,
  });
  footer(s, 6);
  notes(s, "Problem: 6 UAVs, 15 targets, 10 obstacles on a 30×30 grid. Policies may use only instantaneous reachable links. Metrics: discovery Dt, belief decisiveness Ct, and T80.");
}

// ---------------------------------------------------------------------------
// 7 Framework
// ---------------------------------------------------------------------------
{
  const s = pres.addSlide();
  header(s, "2. Proposed IC-HGAT-MAPPO", "2.2 IC-HGAT-MAPPO Framework");
  s.addImage({ path: path.join(A, "framework.png"), x: 0.25, y: 1.18, w: 12.85, h: 3.85 });
  const feats = [
    ["Decentralized", "local + reachable information only"],
    ["Communication-aware", "instantaneous link masking"],
    ["CTDE", "global state only used during training"],
  ];
  feats.forEach((f, i) => {
    const x = 0.35 + i * 4.32;
    card(s, x, 5.12, 4.15, 1.05, i === 1 ? "FFF3E8" : BLUE_LT);
    s.addText(f[0], { x: x + 0.15, y: 5.18, w: 3.85, h: 0.32, fontFace: "Calibri", fontSize: 14, bold: true, color: BLUE_DK, margin: 0 });
    s.addText(f[1], { x: x + 0.15, y: 5.50, w: 3.85, h: 0.52, fontFace: "Calibri", fontSize: 12, color: DARK, margin: 0 });
  });
  s.addText("Safety projection is shared by all methods and is not a contribution.", {
    x: 0.35, y: 6.28, w: 12.6, h: 0.32, fontFace: "Calibri", fontSize: 12, italic: true, color: MUTED, margin: 0,
  });
  footer(s, 7);
  notes(s, "Walk left to right: local observation → link mask → P/E/R memory → heterogeneous graph → shared actor → safety projection. During training the centralized critic sees global state; it is removed at execution.");
}

// ---------------------------------------------------------------------------
// 8 P/E/R + source selection
// ---------------------------------------------------------------------------
{
  const s = pres.addSlide();
  header(s, "2. Proposed IC-HGAT-MAPPO", "2.3 Local Belief Maintenance and Information Selection");
  card(s, 0.28, 1.20, 6.45, 5.55, WHITE);
  s.addText("Revisit-Aware P/E/R Memory", {
    x: 0.42, y: 1.28, w: 6.15, h: 0.36, fontFace: "Calibri", fontSize: 16, bold: true, color: BLUE, margin: 0,
  });
  s.addImage({ path: path.join(A, "per_maps.png"), x: 0.40, y: 1.68, w: 6.20, h: 2.05 });
  s.addImage({ path: path.join(A, "eq_p.png"), x: 0.40, y: 3.75, w: 6.15, h: 0.72 });
  s.addText("R  ←  aging  +  spatial diffusion", {
    x: 0.50, y: 4.50, w: 6.05, h: 0.35, fontFace: "Calibri", fontSize: 15, color: DARK, margin: 0,
  });
  s.addText("sensed cells  →  reset to 0\nunsensed cells  →  gradually become stale", {
    x: 0.50, y: 4.95, w: 6.05, h: 1.15, fontFace: "Calibri", fontSize: 14, color: MUTED, margin: 0,
  });

  card(s, 6.90, 1.20, 6.12, 5.55, WHITE);
  s.addText("Communication-History-Aware Source Selection", {
    x: 7.05, y: 1.28, w: 5.82, h: 0.42, fontFace: "Calibri", fontSize: 14, bold: true, color: ORANGE, margin: 0,
  });
  s.addImage({ path: path.join(A, "source.png"), x: 7.05, y: 1.72, w: 5.80, h: 2.35 });
  s.addImage({ path: path.join(A, "eq_w.png"), x: 7.00, y: 4.05, w: 5.85, h: 0.62 });
  s.addImage({ path: path.join(A, "eq_jstar.png"), x: 7.00, y: 4.62, w: 5.85, h: 0.58 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 7.10, y: 5.30, w: 5.70, h: 1.20, fill: { color: "FDECEA" }, rectRadius: 0.08,
  });
  s.addText("One reachable source is selected per cell", {
    x: 7.25, y: 5.38, w: 5.40, h: 0.42, fontFace: "Calibri", fontSize: 14, bold: true, color: RED, margin: 0,
  });
  s.addText("Avoid correlated evidence accumulation", {
    x: 7.25, y: 5.82, w: 5.40, h: 0.48, fontFace: "Calibri", fontSize: 13, color: DARK, margin: 0,
  });
  footer(s, 8);
  notes(s, "P/E/R remembers what to believe. History-aware weights decide whom to trust. One source per cell, not a sum of correlated log-odds.");
}

// ---------------------------------------------------------------------------
// 9 HGAT + MAPPO
// ---------------------------------------------------------------------------
{
  const s = pres.addSlide();
  header(s, "2. Proposed IC-HGAT-MAPPO", "2.4 Heterogeneous Graph Policy and MAPPO Optimization");
  s.addImage({ path: path.join(A, "hgat.png"), x: 0.20, y: 1.18, w: 6.55, h: 3.55 });
  s.addImage({ path: path.join(A, "eq_att.png"), x: 0.20, y: 4.70, w: 6.55, h: 0.72 });
  s.addText("Relation confidence     ·     Distance prior     ·     Link mask", {
    x: 0.28, y: 5.42, w: 6.45, h: 0.32, fontFace: "Calibri", fontSize: 12, color: BLUE, bold: true, margin: 0,
  });
  s.addImage({ path: path.join(A, "actor_critic.png"), x: 6.85, y: 1.18, w: 6.15, h: 3.55 });
  s.addImage({ path: path.join(A, "eq_r.png"), x: 6.90, y: 4.70, w: 6.10, h: 0.62 });
  s.addText("(λD, λC, λsafe) = (1.0, 0.30, 0.10)     ·     MAPPO: PPO clipping + GAE", {
    x: 6.95, y: 5.38, w: 6.05, h: 0.40, fontFace: "Calibri", fontSize: 12, color: DARK, margin: 0,
  });
  s.addText("Heterogeneous Graph Attention", {
    x: 0.28, y: 5.85, w: 6.4, h: 0.40, fontFace: "Calibri", fontSize: 16, bold: true, color: GREEN, margin: 0,
  });
  footer(s, 9);
  notes(s, "HGAT uses relation-specific attention with confidence, distance, and the instantaneous link mask. Shared actor, one centralized critic. Reward is discovery + decisiveness minus a small safety cost. Full PPO derivation is omitted.");
}

// ---------------------------------------------------------------------------
// 10 Main results
// ---------------------------------------------------------------------------
{
  const s = pres.addSlide();
  header(s, "3. Experimental Results", "3.1 Experimental Setup and Main Results");
  card(s, 0.28, 1.18, 5.55, 2.15, GRAY);
  s.addText("Experiment setup", { x: 0.42, y: 1.24, w: 5.25, h: 0.30, fontFace: "Calibri", fontSize: 14, bold: true, color: BLUE, margin: 0 });
  s.addText("6 UAVs  ·  15 targets  ·  10 obstacles  ·  30×30 grid\nHorizon = 200     ·     2,000 training episodes\n5 independently trained policies\n30 held-out scenarios", {
    x: 0.42, y: 1.56, w: 5.25, h: 1.20, fontFace: "Calibri", fontSize: 13, color: DARK, margin: 0,
  });
  s.addText("Common environment, reward, action space and training budget", {
    x: 0.42, y: 2.82, w: 5.25, h: 0.38, fontFace: "Calibri", fontSize: 12, italic: true, color: MUTED, margin: 0,
  });
  s.addImage({ path: path.join(A, "fig3.png"), x: 0.28, y: 3.45, w: 5.55, h: 2.85 });
  s.addText("UAV trajectories  ·  discovered / undiscovered targets  ·  dynamic obstacles", {
    x: 0.28, y: 6.32, w: 5.55, h: 0.28, fontFace: "Calibri", fontSize: 10, italic: true, color: MUTED, margin: 0,
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
    x: 6.10, y: 4.20, w: 3.20, h: 0.95, fontFace: "Calibri", fontSize: 14, bold: true, color: DARK, margin: 0,
  });
  card(s, 9.55, 3.80, 3.50, 1.55, "FFF3E8");
  s.addText("vs. QMIX", { x: 9.70, y: 3.88, w: 3.20, h: 0.28, fontFace: "Calibri", fontSize: 12, color: ORANGE, bold: true, margin: 0 });
  s.addText("+32.15 pp  decisiveness\n+14.67 pp  discovery", {
    x: 9.70, y: 4.20, w: 3.20, h: 0.95, fontFace: "Calibri", fontSize: 14, bold: true, color: DARK, margin: 0,
  });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 5.95, y: 5.50, w: 7.10, h: 1.28, fill: { color: BLUE }, rectRadius: 0.08,
  });
  s.addText("T80   ·   82.0% success     ·     88.63 steps conditional on success", {
    x: 6.15, y: 5.62, w: 6.75, h: 0.55, fontFace: "Calibri", fontSize: 15, bold: true, color: WHITE, margin: 0,
  });
  s.addText("Safety intervention 0.17%; collision rate comparable to the heuristic (not ranked).", {
    x: 6.15, y: 6.20, w: 6.75, h: 0.40, fontFace: "Calibri", fontSize: 12, color: "D6E2F5", margin: 0,
  });
  footer(s, 10);
  notes(s, "Main comparison from Table I, shown as charts. IC-HGAT-MAPPO leads on decisiveness and discovery. T80 succeeds in 82% of runs at 88.63 steps. Do not sell safety columns as a method advantage.");
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

  s.addImage({ path: path.join(A, "fig4.png"), x: 0.25, y: 4.15, w: 12.80, h: 2.55 });
  s.addText("IC-HGAT-MAPPO maintains higher mean discovery across the tested perturbation ranges.", {
    x: 0.30, y: 6.68, w: 12.7, h: 0.28, fontFace: "Calibri", fontSize: 12, italic: true, color: BLUE_DK, margin: 0,
  });
  footer(s, 11);
  notes(s, "Ablations: heterogeneous relations and revisit memory give the largest, most consistent gains. History weighting improves belief quality and conditional T80, with a success-rate trade-off. Sensitivity: only mention target speed and communication radius / outage, then conclude robustness.");
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
    s.addText(c[0], { x: 0.55, y: y + 0.08, w: 12.22, h: 0.32, fontFace: "Calibri", fontSize: 15, bold: true, color: BLUE_DK, margin: 0 });
    s.addText(c[1], { x: 0.55, y: y + 0.42, w: 12.22, h: 0.42, fontFace: "Calibri", fontSize: 14, color: DARK, margin: 0 });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.35, y: 4.42, w: 12.62, h: 1.05, fill: { color: BLUE }, rectRadius: 0.08,
  });
  s.addText("62.15% belief decisiveness   |   86.78% discovery   |   82.0% T80 success", {
    x: 0.50, y: 4.50, w: 12.32, h: 0.45, fontFace: "Calibri", fontSize: 18, bold: true, color: WHITE, align: "center", margin: 0,
  });
  s.addText("88.63 steps conditional T80", {
    x: 0.50, y: 4.95, w: 12.32, h: 0.38, fontFace: "Calibri", fontSize: 14, color: "D6E2F5", align: "center", margin: 0,
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
    s.addText(f[0], { x: x + 0.12, y: 6.02, w: 3.90, h: 0.30, fontFace: "Calibri", fontSize: 13, bold: true, color: BLUE, margin: 0 });
    s.addText(f[1], { x: x + 0.12, y: 6.34, w: 3.90, h: 0.48, fontFace: "Calibri", fontSize: 12, color: DARK, margin: 0 });
  });
  footer(s, 12);
  notes(s, "Three contributions only. Repeat the headline numbers. Future work matches the paper: 3D, decentralized safety, HIL. Optionally mention 3D pursuit as an oral aside, not a new section.");
}

// ---------------------------------------------------------------------------
// 13 Thank you
// ---------------------------------------------------------------------------
{
  const s = pres.addSlide();
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 13.333, h: 7.5, fill: { color: WHITE } });
  s.addImage({ path: path.join(A, "hust.png"), x: 0.45, y: 0.28, w: 1.45, h: 1.00 });
  s.addImage({ path: path.join(A, "caa.png"), x: 11.15, y: 0.28, w: 1.70, h: 0.95 });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 1.50, w: 13.333, h: 3.15, fill: { color: BLUE } });
  s.addText("Thank you for your attention!", {
    x: 0.50, y: 2.15, w: 12.3, h: 1.10,
    fontFace: "Calibri", fontSize: 36, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0,
  });
  s.addText("A Heterogeneous Graph-Attention MAPPO Algorithm for\nCooperative Multi-UAV Search under Intermittent Communication Constraints", {
    x: 0.70, y: 3.30, w: 11.9, h: 1.00,
    fontFace: "Calibri", fontSize: 16, color: "D6E2F5", align: "center", margin: 0,
  });
  s.addText("Yinying Kuang\nHuazhong University of Science and Technology\nkuangyinying039@163.com", {
    x: 0.50, y: 5.05, w: 12.3, h: 1.25,
    fontFace: "Calibri", fontSize: 16, color: DARK, align: "center", margin: 0,
  });
  s.addText("CAC 2026  ·  October 2026", {
    x: 0.50, y: 6.45, w: 12.3, h: 0.35,
    fontFace: "Calibri", fontSize: 14, color: BLUE, align: "center", bold: true, margin: 0,
  });
  notes(s, "Thank the audience. Leave the email visible. Do not add a GitHub QR unless the chair confirms code can be public.");
}

const out = "/workspace/presentations/IC-HGAT-MAPPO_CAC2026.pptx";
pres.writeFile({ fileName: out }).then(() => {
  console.log("wrote", out);
});
