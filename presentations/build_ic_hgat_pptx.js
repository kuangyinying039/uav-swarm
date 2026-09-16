#!/usr/bin/env node
/**
 * 13-page academic talk deck: IC-HGAT-MAPPO
 * Visual language follows the uploaded formation-control template
 * (HUST logo, CAA/CAC branding, blue title blocks, 13.333 x 7.5 in).
 */
const PptxGenJS = require("pptxgenjs");
const path = require("path");

const A = "/tmp/ppt-work/assets";
const EQ = path.join(__dirname, "eq");
const FONT = "Times New Roman";

const BLUE = "2F5597";
const BLUE_DK = "1E3F73";
const BLUE_LT = "EAF0F8";
const ORANGE = "ED7D31";
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
  "framework.png": 2101 / 693,
  "hgat.png": 1024 / 648,
  "actor_critic.png": 900 / 586,
  "per_maps.png": 1631 / 569,
  "source.png": 1008 / 547,
  "aging_flow.png": 1140 / 201,
  "photo_eq.png": 4.05 / 2.15,
  "photo_wf.png": 4.05 / 2.15,
  "photo_mr.png": 4.05 / 2.15,
};
Object.assign(AR, require(path.join(EQ, "ar.json")));

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

function cover(slide, file, box) {
  slide.addImage({
    path: file,
    x: box.x,
    y: box.y,
    w: box.w,
    h: box.h,
    sizing: { type: "cover", w: box.w, h: box.h },
  });
}

function eqImg(slide, name, box) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: box.x, y: box.y, w: box.w, h: box.h,
    fill: { color: "F4EFE4" }, rectRadius: 0.05,
  });
  contain(slide, path.join(EQ, name + ".png"), {
    x: box.x + 0.08,
    y: box.y + 0.04,
    w: box.w - 0.16,
    h: box.h - 0.08,
  });
}

function accent(slide, x, y, h) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w: 0.07, h, fill: { color: BLUE },
  });
}

function header(slide, section, subtitle) {
  contain(slide, path.join(A, "hust.png"), { x: 12.48, y: 0.06, w: 0.70, h: 0.40 });
  slide.addText(section, {
    x: 0.38, y: 0.05, w: 11.9, h: 0.34,
    fontFace: FONT, fontSize: 16, bold: true, color: BLUE, margin: 0, valign: "middle",
  });
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0.42, w: 13.333, h: 0.42, fill: { color: BLUE },
  });
  slide.addText(subtitle, {
    x: 0.38, y: 0.43, w: 12.5, h: 0.40,
    fontFace: FONT, fontSize: 18, bold: true, color: WHITE, margin: 0, valign: "middle",
  });
}

function footer(slide, n) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 7.42, w: 13.333, h: 0.08, fill: { color: BLUE },
  });
  slide.addText("CAC 2026  ·  IC-HGAT-MAPPO", {
    x: 0.38, y: 7.10, w: 11.9, h: 0.28,
    fontFace: FONT, fontSize: 12, color: MUTED, margin: 0, valign: "middle",
  });
  slide.addText(String(n), {
    x: 12.40, y: 7.08, w: 0.70, h: 0.30,
    fontFace: FONT, fontSize: 13, bold: true, color: BLUE, align: "right", margin: 0,
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

const N = require("./speaker_notes");



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
    fontFace: "Times New Roman", fontSize: 18, color: BLUE, bold: true, margin: 0, valign: "middle",
  });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 1.55, w: 13.333, h: 3.35, fill: { color: BLUE } });
  s.addText("A Heterogeneous Graph-Attention MAPPO Algorithm\nfor Cooperative Multi-UAV Search\nunder Intermittent Communication Constraints", {
    x: 0.45, y: 1.65, w: 12.4, h: 3.15,
    fontFace: "Times New Roman", fontSize: 30, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0,
  });
  s.addText([
    { text: "Yinying Kuang", options: { bold: true, fontSize: 22, color: BLUE } },
    { text: ", Zhaowei Liang, Yongran Zhi, Huijin Fan*, Lei Liu*, Bo Wang*", options: { bold: false, fontSize: 18, color: DARK } },
  ], {
    x: 0.5, y: 5.05, w: 12.3, h: 0.50,
    fontFace: FONT, align: "center", valign: "middle", margin: 0,
  });
  s.addText("Huazhong University of Science and Technology\nWuhan Second Ship Design and Research Institute", {
    x: 0.5, y: 5.55, w: 12.3, h: 0.78,
    fontFace: "Times New Roman", fontSize: 17, color: MUTED, align: "center", valign: "middle", margin: 0,
  });
  s.addText("CAC 2026  ·  Beijing  ·  October 2026", {
    x: 0.5, y: 6.32, w: 12.3, h: 0.36,
    fontFace: "Times New Roman", fontSize: 17, color: BLUE, align: "center", bold: true, valign: "middle", margin: 0,
  });
  const chips = ["Dec-POMDP", "Intermittent links", "P / E / R memory", "History-aware source", "HGAT + MAPPO / CTDE"];
  chips.forEach((c, i) => {
    const x = 0.38 + i * 2.58;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y: 6.78, w: 2.48, h: 0.48, fill: { color: BLUE_LT }, rectRadius: 0.08,
    });
    s.addText(c, {
      x, y: 6.78, w: 2.48, h: 0.48,
      fontFace: "Times New Roman", fontSize: 11, bold: true, color: BLUE_DK, align: "center", valign: "middle", margin: 0,
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
    fontFace: "Times New Roman", fontSize: 28, bold: true, color: BLUE, margin: 0, valign: "middle",
  });
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0.72, w: 13.333, h: 0.14, fill: { color: BLUE } });
  const items = [
    ["01", "Background and Problem Formulation", "Applications and challenges of intermittent communication"],
    ["02", "Proposed IC-HGAT-MAPPO", "Local beliefs, source selection, and heterogeneous graph policy"],
    ["03", "Experimental Results and Conclusions", "Comparison, ablation, sensitivity, and main findings"],
  ];
  items.forEach((it, i) => {
    const y = 1.70 + i * 1.72;
    card(s, 0.35, y, 12.62, 1.50, i === 1 ? "FFF3E8" : WHITE);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 0.55, y: y + 0.28, w: 1.35, h: 0.95, fill: { color: BLUE }, rectRadius: 0.10,
    });
    s.addText(it[0], {
      x: 0.55, y: y + 0.28, w: 1.35, h: 0.95,
      fontFace: FONT, fontSize: 28, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0,
    });
    s.addText(it[1], {
      x: 2.10, y: y + 0.22, w: 10.55, h: 0.52,
      fontFace: FONT, fontSize: 22, bold: true, color: DARK, valign: "middle", margin: 0,
    });
    s.addText(it[2], {
      x: 2.10, y: y + 0.78, w: 10.55, h: 0.48,
      fontFace: FONT, fontSize: 16, color: MUTED, valign: "middle", margin: 0,
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
    cover(s, ph[0], { x, y: 0.98, w: 4.05, h: 2.55 });
    s.addText(ph[1], {
      x, y: 3.56, w: 4.05, h: 0.36,
      fontFace: FONT, fontSize: 16, bold: true, color: DARK, align: "center", valign: "middle", margin: 0,
    });
  });
  s.addShape(pres.shapes.DOWN_ARROW, { x: 6.35, y: 3.95, w: 0.62, h: 0.32, fill: { color: BLUE } });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 2.55, y: 4.32, w: 8.22, h: 0.48, fill: { color: BLUE }, rectRadius: 0.08,
  });
  s.addText("Multi-UAV cooperative search", {
    x: 2.55, y: 4.32, w: 8.22, h: 0.48,
    fontFace: FONT, fontSize: 18, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0,
  });
  s.addShape(pres.shapes.DOWN_ARROW, { x: 6.35, y: 4.86, w: 0.62, h: 0.28, fill: { color: ORANGE } });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 1.55, y: 5.18, w: 10.22, h: 0.48, fill: { color: "FFF3E8" }, rectRadius: 0.08,
  });
  s.addText("Local observation  +  intermittent communication", {
    x: 1.55, y: 5.18, w: 10.22, h: 0.48,
    fontFace: FONT, fontSize: 17, bold: true, color: ORANGE, align: "center", valign: "middle", margin: 0,
  });
  const tags = ["Local belief", "Stale peer information", "Heterogeneous interaction"];
  tags.forEach((t, i) => {
    const x = 0.40 + i * 4.30;
    card(s, x, 5.82, 4.05, 1.10, i === 1 ? "FFF3E8" : BLUE_LT);
    s.addText(t, {
      x, y: 5.82, w: 4.05, h: 1.10,
      fontFace: FONT, fontSize: 16, bold: true, color: BLUE_DK, align: "center", valign: "middle", margin: 0,
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
    ["Challenge 1", "Incomplete local beliefs", "chal1.png", BLUE],
    ["Challenge 2", "Stale peer information", "chal3.png", ORANGE],
    ["Challenge 3", "Heterogeneous interactions", "chal2.png", BLUE_DK],
  ];
  titles.forEach((t, i) => {
    const x = 0.35 + i * 4.32;
    card(s, x, 0.98, 4.15, 4.85, WHITE);
    s.addShape(pres.shapes.RECTANGLE, { x, y: 0.98, w: 4.15, h: 0.70, fill: { color: t[3] } });
    s.addText(t[0], {
      x: x + 0.12, y: 1.00, w: 3.90, h: 0.28,
      fontFace: FONT, fontSize: 13, color: "D6E2F5", margin: 0, valign: "middle",
    });
    s.addText(t[1], {
      x: x + 0.12, y: 1.26, w: 3.90, h: 0.38,
      fontFace: FONT, fontSize: 16, bold: true, color: WHITE, margin: 0, valign: "middle",
    });
    contain(s, path.join(A, t[2]), { x: x + 0.18, y: 1.85, w: 3.80, h: 2.55 });
    s.addText(i === 0 ? "No shared global map" : i === 1 ? "Reachable  ≠  useful" : "Peer / task / obstacle", {
      x: x + 0.18, y: 4.50, w: 3.80, h: 1.10,
      fontFace: FONT, fontSize: 16, color: DARK, align: "center", valign: "middle", margin: 0,
    });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.35, y: 6.00, w: 12.62, h: 0.92, fill: { color: BLUE_LT }, rectRadius: 0.08,
  });
  s.addText("Focus: decision-making with intermittent information, not communication-protocol optimization.", {
    x: 0.50, y: 6.00, w: 12.32, h: 0.92,
    fontFace: FONT, fontSize: 16, bold: true, color: BLUE_DK, align: "center", valign: "middle", margin: 0,
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
    ["Challenge 1", "Incomplete local beliefs", "Revisit-aware\nP/E/R memory", BLUE],
    ["Challenge 2", "Stale peer information", "History-aware\nsource selection", ORANGE],
    ["Challenge 3", "Heterogeneous interactions", "Heterogeneous\ngraph attention", BLUE_DK],
  ];
  maps.forEach((m, i) => {
    const x = 0.40 + i * 4.30;
    card(s, x, 1.00, 4.05, 2.85, WHITE);
    s.addText(m[0], { x, y: 1.08, w: 4.05, h: 0.32, fontFace: FONT, fontSize: 13, color: MUTED, align: "center", margin: 0, valign: "middle" });
    s.addText(m[1], { x: x + 0.12, y: 1.40, w: 3.80, h: 0.55, fontFace: FONT, fontSize: 16, bold: true, color: DARK, align: "center", margin: 0, valign: "middle" });
    s.addShape(pres.shapes.DOWN_ARROW, { x: x + 1.67, y: 2.00, w: 0.70, h: 0.32, fill: { color: m[3] } });
    s.addText(m[2], { x: x + 0.12, y: 2.38, w: 3.80, h: 1.25, fontFace: FONT, fontSize: 18, bold: true, color: m[3], align: "center", margin: 0, valign: "middle" });
  });
  s.addShape(pres.shapes.DOWN_ARROW, { x: 6.32, y: 3.95, w: 0.70, h: 0.36, fill: { color: BLUE } });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 3.55, y: 4.40, w: 6.22, h: 1.15, fill: { color: BLUE }, rectRadius: 0.08,
  });
  s.addText("IC-HGAT-MAPPO", {
    x: 3.55, y: 4.46, w: 6.22, h: 0.58,
    fontFace: FONT, fontSize: 24, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0,
  });
  s.addText("Decentralized execution: local observation + reachable peers", {
    x: 3.55, y: 5.00, w: 6.22, h: 0.45,
    fontFace: FONT, fontSize: 13, color: "D6E2F5", align: "center", valign: "middle", margin: 0,
  });
  card(s, 0.40, 5.78, 12.55, 1.12, "FFF3E8");
  s.addText("Training only: centralized critic + MAPPO   ·   Safety projection is shared by all methods", {
    x: 0.55, y: 5.78, w: 12.25, h: 1.12,
    fontFace: FONT, fontSize: 16, color: ORANGE, align: "center", valign: "middle", margin: 0,
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
  contain(s, path.join(A, "fig1.png"), { x: 0.28, y: 0.98, w: 6.45, h: 3.55 });
  s.addText("Fig. 1  Local sensing, intermittent links, moving targets, dynamic obstacles", {
    x: 0.28, y: 4.54, w: 6.45, h: 0.26, fontFace: FONT, fontSize: 11, italic: true, color: MUTED, margin: 0, valign: "middle",
  });
  card(s, 0.28, 4.84, 6.45, 2.08, GRAY);
  s.addText("Scenario", { x: 0.42, y: 4.90, w: 6.15, h: 0.28, fontFace: FONT, fontSize: 15, bold: true, color: BLUE, margin: 0, valign: "middle" });
  s.addText("6 UAVs · 15 moving targets · 10 dynamic obstacles\n30×30 grid · 8 directions + hover · T = 200\nRs = 2 cells (10 m) · Rc = 10 cells (50 m)\nAdaptive dt in [0.45, 1.0] s · local + reachable peers only", {
    x: 0.42, y: 5.20, w: 6.15, h: 1.58, fontFace: FONT, fontSize: 14, color: DARK, margin: 0, valign: "middle",
  });

  card(s, 6.90, 0.96, 6.10, 1.38, WHITE);
  s.addText("Motion", { x: 7.04, y: 0.98, w: 5.82, h: 0.24, fontFace: FONT, fontSize: 13, bold: true, color: BLUE, margin: 0 });
  eqImg(s, "motion", { x: 7.00, y: 1.22, w: 5.90, h: 0.62 });
  s.addText("Feasible-set projection with adaptive interval", {
    x: 7.04, y: 1.86, w: 5.82, h: 0.38, fontFace: FONT, fontSize: 12, italic: true, color: MUTED, margin: 0, valign: "middle",
  });

  card(s, 6.90, 2.42, 6.10, 1.38, WHITE);
  s.addText("Communication", { x: 7.04, y: 2.44, w: 5.82, h: 0.24, fontFace: FONT, fontSize: 13, bold: true, color: BLUE, margin: 0 });
  eqImg(s, "link", { x: 7.00, y: 2.68, w: 5.90, h: 0.62 });
  s.addText("Only instantaneous reachable links enter the policy", {
    x: 7.04, y: 3.32, w: 5.82, h: 0.38, fontFace: FONT, fontSize: 12, italic: true, color: MUTED, margin: 0, valign: "middle",
  });

  card(s, 6.90, 3.88, 6.10, 3.04, BLUE_LT);
  s.addText("Evaluation", { x: 7.04, y: 3.92, w: 5.82, h: 0.24, fontFace: FONT, fontSize: 14, bold: true, color: BLUE, margin: 0 });
  eqImg(s, "dt", { x: 7.00, y: 4.16, w: 5.90, h: 0.40 });
  eqImg(s, "ct", { x: 7.00, y: 4.58, w: 5.90, h: 0.72 });
  eqImg(s, "plim", { x: 7.00, y: 5.32, w: 5.90, h: 0.36 });
  s.addText("Class-balanced Brier score (BBS) ↓: ground-truth belief error", {
    x: 7.04, y: 5.70, w: 5.82, h: 0.36, fontFace: FONT, fontSize: 13, color: DARK, margin: 0, valign: "middle",
  });
  eqImg(s, "t80", { x: 7.00, y: 6.08, w: 5.90, h: 0.40 });
  s.addText("T80: steps / physical time to 80% discovery", {
    x: 7.04, y: 6.50, w: 5.82, h: 0.32, fontFace: FONT, fontSize: 12, italic: true, color: MUTED, margin: 0, valign: "middle",
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
  contain(s, path.join(A, "framework.png"), { x: 0.22, y: 0.96, w: 12.90, h: 3.55 });
  const feats = [
    ["1  Observe", "local o_i,t + link mask"],
    ["2  Remember", "update P / E / R"],
    ["3  Select", "source selection"],
    ["4  Encode", "HGAT embedding"],
    ["5  Act", "actor + safety"],
    ["6  Train", "central critic"],
  ];
  feats.forEach((f, i) => {
    const x = 0.28 + i * 2.16;
    card(s, x, 4.62, 2.08, 1.05, i === 2 || i === 5 ? "FFF3E8" : BLUE_LT);
    s.addText(f[0], { x: x + 0.08, y: 4.66, w: 1.92, h: 0.34, fontFace: FONT, fontSize: 13, bold: true, color: BLUE_DK, margin: 0, valign: "middle" });
    s.addText(f[1], { x: x + 0.08, y: 5.00, w: 1.92, h: 0.55, fontFace: FONT, fontSize: 12, color: DARK, margin: 0, valign: "middle" });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.28, y: 5.82, w: 12.78, h: 1.10, fill: { color: GRAY }, rectRadius: 0.08,
  });
  s.addText("Source selection sits between P/E/R memory and the heterogeneous graph.\nSafety projection is shared by all methods and is not a contribution.", {
    x: 0.42, y: 5.82, w: 12.50, h: 1.10, fontFace: FONT, fontSize: 15, color: DARK, margin: 0, valign: "middle",
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
  card(s, 0.28, 0.98, 6.45, 5.92, WHITE);
  accent(s, 0.28, 0.98, 5.92);
  s.addText("Revisit-aware P/E/R memory", {
    x: 0.46, y: 1.04, w: 6.10, h: 0.38, fontFace: FONT, fontSize: 17, bold: true, color: BLUE, margin: 0, valign: "middle",
  });
  contain(s, path.join(A, "per_maps.png"), { x: 0.42, y: 1.46, w: 6.15, h: 2.35 });
  s.addText("P  existence    ·    E  uncertainty    ·    R  revisit staleness", {
    x: 0.46, y: 3.84, w: 6.10, h: 0.32, fontFace: FONT, fontSize: 13, color: MUTED, margin: 0, valign: "middle",
  });
  contain(s, path.join(A, "aging_flow.png"), { x: 0.42, y: 4.22, w: 6.15, h: 1.18 });
  s.addText("Log-odds update only on sensed cells.  R ages, then diffuses, then resets when sensed.", {
    x: 0.46, y: 5.48, w: 6.10, h: 1.20, fontFace: FONT, fontSize: 14, color: DARK, margin: 0, valign: "middle",
  });

  card(s, 6.90, 0.98, 6.12, 5.92, WHITE);
  accent(s, 6.90, 0.98, 5.92);
  s.addText("History-aware source selection", {
    x: 7.08, y: 1.04, w: 5.78, h: 0.38, fontFace: FONT, fontSize: 16, bold: true, color: ORANGE, margin: 0, valign: "middle",
  });
  contain(s, path.join(A, "source.png"), { x: 7.05, y: 1.48, w: 5.80, h: 2.45 });
  s.addText("g(j, t): node-level belief-age summary", {
    x: 7.08, y: 4.00, w: 5.78, h: 0.36, fontFace: FONT, fontSize: 15, bold: true, color: BLUE_DK, margin: 0, valign: "middle",
  });
  s.addText("Freshness of UAV j's belief, not pair-wise outage duration.", {
    x: 7.08, y: 4.36, w: 5.78, h: 0.40, fontFace: FONT, fontSize: 13, color: MUTED, margin: 0, valign: "middle",
  });
  eqImg(s, "jstar", { x: 7.05, y: 4.82, w: 5.82, h: 0.70 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 7.08, y: 5.62, w: 5.76, h: 1.08, fill: { color: BLUE_LT }, rectRadius: 0.08,
  });
  s.addText("One reachable source per cell, instead of adding correlated log-odds.", {
    x: 7.22, y: 5.62, w: 5.48, h: 1.08, fontFace: FONT, fontSize: 14, color: DARK, margin: 0, valign: "middle",
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
  contain(s, path.join(A, "hgat.png"), { x: 0.22, y: 0.96, w: 6.45, h: 3.35 });
  contain(s, path.join(A, "actor_critic.png"), { x: 6.78, y: 0.96, w: 6.22, h: 3.35 });
  s.addText("Peer / task / obstacle relations     ·     2 layers, 4 heads     ·     link mask in attention", {
    x: 0.28, y: 4.34, w: 12.75, h: 0.28, fontFace: FONT, fontSize: 13, color: BLUE, margin: 0, valign: "middle",
  });
  eqImg(s, "att", { x: 0.28, y: 4.66, w: 12.75, h: 0.62 });
  eqImg(s, "reward", { x: 0.28, y: 5.34, w: 6.28, h: 0.58 });
  eqImg(s, "clip", { x: 6.68, y: 5.34, w: 6.35, h: 0.58 });
  s.addText("Reward weights (1.0, 0.30, 0.10)     ·     MAPPO clip 0.20     ·     critic used only in training", {
    x: 0.28, y: 6.00, w: 12.75, h: 0.92, fontFace: FONT, fontSize: 14, color: MUTED, margin: 0, valign: "middle",
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
  card(s, 0.28, 0.96, 5.50, 1.72, GRAY);
  s.addText("Experiment protocol  (fair comparison)", {
    x: 0.40, y: 1.00, w: 5.26, h: 0.28, fontFace: FONT, fontSize: 14, bold: true, color: BLUE, margin: 0, valign: "middle",
  });
  s.addText("6 UAVs · 15 targets · 10 obstacles · 30×30\nHorizon 200  ·  2,000 episodes  ·  5 seeds\n30 held-out scenarios   Rs=2, Rc=10 cells\nSame env, reward, action, budget", {
    x: 0.40, y: 1.28, w: 5.26, h: 1.32, fontFace: FONT, fontSize: 13, color: DARK, margin: 0, valign: "middle",
  });
  contain(s, path.join(A, "fig3.png"), { x: 0.28, y: 2.74, w: 5.50, h: 3.02 });
  s.addText("Representative search trajectory", {
    x: 0.28, y: 5.80, w: 5.50, h: 0.28, fontFace: FONT, fontSize: 13, italic: true, color: MUTED, margin: 0, valign: "middle",
  });
  card(s, 0.28, 6.14, 5.50, 0.78, BLUE_LT);
  s.addText("Baselines: Heuristic, QMIX, Local MAPPO, Homog. Graph MAPPO", {
    x: 0.40, y: 6.14, w: 5.26, h: 0.78, fontFace: FONT, fontSize: 13, color: BLUE_DK, margin: 0, valign: "middle",
  });

  const labels = ["Heuristic", "QMIX", "Local", "H-Graph", "IC-HGAT"];
  s.addChart(pres.charts.BAR, [{
    name: "Decisiveness",
    labels,
    values: [44.99, 30.00, 41.91, 47.05, 62.15],
  }], {
    x: 5.92, y: 0.94, w: 3.62, h: 2.42,
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
    titleFontFace: FONT,
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
    x: 9.55, y: 0.94, w: 3.50, h: 2.42,
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
    titleFontFace: FONT,
    chartColors: ["5B8CC9", "5B8CC9", "5B8CC9", "5B8CC9", ORANGE],
    valAxisMaxValue: 100,
    catAxisLabelColor: DARK,
    valAxisLabelColor: MUTED,
    catAxisLabelFontSize: 8,
    valGridLine: { color: "D0D7E2", size: 0.5 },
    catGridLine: { style: "none" },
  });

  card(s, 5.92, 3.42, 3.52, 1.18, BLUE_LT);
  s.addText("vs. Heuristic", { x: 6.06, y: 3.48, w: 3.24, h: 0.26, fontFace: FONT, fontSize: 12, color: BLUE, bold: true, margin: 0 });
  s.addText("+17.16 pp  decisiveness\n+9.00 pp  discovery", {
    x: 6.06, y: 3.74, w: 3.24, h: 0.76, fontFace: FONT, fontSize: 15, bold: true, color: DARK, margin: 0, valign: "middle",
  });
  card(s, 9.53, 3.42, 3.52, 1.18, "FFF3E8");
  s.addText("vs. QMIX", { x: 9.67, y: 3.48, w: 3.24, h: 0.26, fontFace: FONT, fontSize: 12, color: ORANGE, bold: true, margin: 0 });
  s.addText("+32.15 pp  decisiveness\n+14.67 pp  discovery", {
    x: 9.67, y: 3.74, w: 3.24, h: 0.76, fontFace: FONT, fontSize: 15, bold: true, color: DARK, margin: 0, valign: "middle",
  });

  const kpis = [
    ["62.15%", "Belief decisiveness", BLUE],
    ["86.78%", "Target discovery", BLUE],
    ["0.112 ± 0.023", "Terminal BBS ↓", ORANGE],
    ["82.0% / 63.80 s", "T80 success / physical time", BLUE_DK],
  ];
  kpis.forEach((k, i) => {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const x = 5.92 + col * 3.61;
    const y = 4.70 + row * 1.08;
    card(s, x, y, 3.52, 1.00, WHITE);
    s.addText(k[0], {
      x: x + 0.10, y: y + 0.06, w: 3.32, h: 0.46, fontFace: FONT, fontSize: 20, bold: true, color: k[2], margin: 0, valign: "middle",
    });
    s.addText(k[1], {
      x: x + 0.10, y: y + 0.52, w: 3.32, h: 0.40, fontFace: FONT, fontSize: 13, color: MUTED, margin: 0, valign: "middle",
    });
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
    x: 0.18, y: 0.90, w: 6.45, h: 2.12,
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
    titleFontFace: FONT,
    chartColors: [BLUE, ORANGE],
    valAxisMaxValue: 100,
    catAxisLabelColor: DARK,
    valAxisLabelColor: MUTED,
    catAxisLabelFontSize: 9,
    valGridLine: { color: "D0D7E2", size: 0.5 },
    catGridLine: { style: "none" },
  });

  const acards = [
    ["Heterogeneous relations", "+15.10 pp decisiveness vs. Homogeneous Graph MAPPO", BLUE],
    ["Revisit memory", "−16.33 pp decisiveness, −13.67 pp discovery when removed", ORANGE],
    ["History weighting", "−8.16 pp decisiveness; T80 88.63→98.19 steps (quality / efficiency)", BLUE_DK],
  ];
  acards.forEach((c, i) => {
    const y = 0.92 + i * 0.70;
    card(s, 6.72, y, 6.28, 0.64, WHITE);
    accent(s, 6.72, y, 0.64);
    s.addText(c[0], { x: 6.92, y: y + 0.02, w: 5.92, h: 0.24, fontFace: FONT, fontSize: 13, bold: true, color: c[2], margin: 0, valign: "middle" });
    s.addText(c[1], { x: 6.92, y: y + 0.28, w: 5.92, h: 0.30, fontFace: FONT, fontSize: 12, color: DARK, margin: 0, valign: "middle" });
  });

  contain(s, path.join(A, "fig4.png"), { x: 0.22, y: 3.08, w: 12.90, h: 3.12 });
  card(s, 0.28, 6.28, 12.78, 0.68, BLUE_LT);
  s.addText("Sensitivity: IC-HGAT-MAPPO retains higher mean discovery across the tested perturbation ranges.", {
    x: 0.42, y: 6.28, w: 12.50, h: 0.68, fontFace: FONT, fontSize: 15, bold: true, color: BLUE_DK, margin: 0, valign: "middle",
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
    ["1. Communication-constrained decision making", "Execution uses only local observations and currently reachable peer information."],
    ["2. Belief management under incomplete and stale information", "Revisit-aware P/E/R memory and communication-history-aware source selection."],
    ["3. Heterogeneous relational encoding", "HGAT models peer, task, and obstacle relations within a MAPPO CTDE framework."],
  ];
  cons.forEach((c, i) => {
    const y = 0.96 + i * 0.82;
    card(s, 0.35, y, 12.62, 0.74, i === 1 ? "FFF3E8" : BLUE_LT);
    accent(s, 0.35, y, 0.74);
    s.addText(c[0], { x: 0.55, y: y + 0.04, w: 12.22, h: 0.30, fontFace: FONT, fontSize: 16, bold: true, color: BLUE_DK, margin: 0, valign: "middle" });
    s.addText(c[1], { x: 0.55, y: y + 0.34, w: 12.22, h: 0.34, fontFace: FONT, fontSize: 14, color: DARK, margin: 0, valign: "middle" });
  });

  const kpis = [
    ["62.15%", "Belief decisiveness"],
    ["86.78%", "Target discovery"],
            ["0.112 ± 0.023", "Terminal BBS ↓"],
    ["82.0% / 63.80 s", "T80 success / time"],
  ];
  kpis.forEach((k, i) => {
    const x = 0.35 + i * 3.16;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y: 3.52, w: 3.05, h: 1.18, fill: { color: BLUE }, rectRadius: 0.08,
    });
    s.addText(k[0], {
      x, y: 3.58, w: 3.05, h: 0.58, fontFace: FONT, fontSize: 20, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0,
    });
    s.addText(k[1], {
      x, y: 4.14, w: 3.05, h: 0.46, fontFace: FONT, fontSize: 13, color: "D6E2F5", align: "center", valign: "middle", margin: 0,
    });
  });

  s.addText("Future work", {
    x: 0.40, y: 4.82, w: 12.5, h: 0.32, fontFace: FONT, fontSize: 16, bold: true, color: BLUE, margin: 0, valign: "middle",
  });
  const fut = [
    ["Motion-aware belief", "Prediction and calibration for moving targets"],
    ["3-D search environments", "Extend planar search to 3-D kinematics and occupancy"],
    ["HIL / real-radio validation", "Asynchronous communication with realistic radio links"],
  ];
  fut.forEach((f, i) => {
    const x = 0.35 + i * 4.32;
    card(s, x, 5.22, 4.15, 1.66, GRAY);
    s.addText(f[0], { x: x + 0.16, y: 5.36, w: 3.82, h: 0.52, fontFace: FONT, fontSize: 16, bold: true, color: BLUE, margin: 0, valign: "middle" });
    s.addText(f[1], { x: x + 0.16, y: 5.92, w: 3.82, h: 0.78, fontFace: FONT, fontSize: 14, color: DARK, margin: 0, valign: "middle" });
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
  s.addText("Thank you for your attention.", {
    x: 0.50, y: 2.05, w: 12.3, h: 1.20,
    fontFace: FONT, fontSize: 40, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0,
  });
  s.addText("A Heterogeneous Graph-Attention MAPPO Algorithm for\nCooperative Multi-UAV Search under Intermittent Communication Constraints", {
    x: 0.70, y: 3.28, w: 11.9, h: 1.10,
    fontFace: FONT, fontSize: 18, color: "D6E2F5", align: "center", valign: "middle", margin: 0,
  });
  s.addText("Yinying Kuang\nHuazhong University of Science and Technology\nkuangyinying039@163.com", {
    x: 0.50, y: 4.82, w: 12.3, h: 1.15,
    fontFace: FONT, fontSize: 18, color: DARK, align: "center", valign: "middle", margin: 0,
  });
  const qchips = ["Dec-POMDP", "P / E / R memory", "Source selection", "HGAT + MAPPO", "CTDE"];
  qchips.forEach((c, i) => {
    const x = 0.38 + i * 2.58;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y: 6.12, w: 2.48, h: 0.44, fill: { color: BLUE_LT }, rectRadius: 0.08,
    });
    s.addText(c, {
      x, y: 6.12, w: 2.48, h: 0.44,
      fontFace: FONT, fontSize: 13, bold: true, color: BLUE_DK, align: "center", valign: "middle", margin: 0,
    });
  });
  s.addText("CAC 2026  ·  Beijing  ·  October 2026", {
    x: 0.50, y: 6.68, w: 12.3, h: 0.36,
    fontFace: FONT, fontSize: 16, color: BLUE, align: "center", bold: true, valign: "middle", margin: 0,
  });
  notes(s, N[12]);
}

const out = "/workspace/presentations/IC-HGAT-MAPPO_CAC2026.pptx";
pres.writeFile({ fileName: out }).then(() => {
  console.log("wrote", out);
});
