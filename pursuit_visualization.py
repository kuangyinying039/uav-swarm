"""Offline interactive visualization for the 3-D quadrotor pursuit task."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from marl_trainers import capture_env_frame


def write_pursuit_animation_html(path: Path, trace: dict, title: str = "Pursuit–Evasion Simulation") -> None:
    """Write a self-contained HTML animation requiring no external libraries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(trace, separators=(",", ":"))
    html = _HTML_TEMPLATE.replace("__TITLE__", title).replace("__TRACE__", payload)
    path.write_text(html, encoding="utf-8")


def simulate_rule_episode(
    output: Path,
    *,
    seed: int = 23,
    steps: int = 180,
    evader_policy: str = "repulsive",
) -> dict:
    from quadrotor_pursuit_env import QuadrotorPursuitConfig, QuadrotorPursuitEnv

    env = QuadrotorPursuitEnv(
        QuadrotorPursuitConfig(seed=seed, search_steps=steps, evader_policy=evader_policy)
    )
    frames = []
    for _ in range(steps):
        actions = heuristic_continuous_actions(env)
        result = env.step_joint(actions)
        frames.append(capture_env_frame(env, result, actions.tolist()))
        if result["terminated"] or result["truncated"]:
            break
    trace = {
        "episode": 1,
        "mode": "3d",
        "frames": frames,
    }
    write_pursuit_animation_html(output, trace, "3-D Quadrotor Pursuit–Evasion Simulation")
    return trace


def heuristic_pursuit_actions(env) -> np.ndarray:
    """Information-causal demonstration policy, not a training baseline."""
    actions = np.full(env.cfg.n_uavs, 8, dtype=int)
    horizontal = np.asarray(
        [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]],
        dtype=float,
    )
    for i in range(env.cfg.n_uavs):
        if env.disabled_uavs[i]:
            continue
        if env.target_belief_timestamp[i, 0] >= 0:
            goal = env.target_belief_mean[i, 0]
            # Offset the three agents around the belief to demonstrate enclosure.
            angle = 2.0 * np.pi * i / max(env.cfg.n_uavs, 1)
            goal = goal + 0.65 * env.cfg.capture_radius * np.array([np.cos(angle), np.sin(angle)])
        else:
            phase = (env.t // 8 + i * 3) % 8
            goal = env.positions[i].astype(float) + horizontal[phase] * 3.0
        delta = goal - env.positions[i].astype(float)
        if np.linalg.norm(delta) > 0.25:
            score = horizontal @ (delta / max(np.linalg.norm(delta), 1e-8))
            order = np.argsort(-score)
            mask = env.action_mask(i)
            for candidate in order:
                if mask[int(candidate)]:
                    actions[i] = int(candidate)
                    break
        if hasattr(env, "altitudes") and hasattr(env, "target_altitudes"):
            dz = float(env.target_altitudes[0] - env.altitudes[i])
            mask = env.action_mask(i)
            if dz > 0.6 and mask[9]:
                actions[i] = 9
            elif dz < -0.6 and mask[10]:
                actions[i] = 10
    return actions


def heuristic_continuous_actions(env) -> np.ndarray:
    dim = env.continuous_action_dim()
    actions = np.zeros((env.cfg.n_uavs, dim), dtype=float)
    for i in range(env.cfg.n_uavs):
        if env.target_belief_timestamp[i, 0] >= 0:
            goal = env.target_belief_mean[i, 0].copy()
            angle = 2.0 * np.pi * i / max(env.cfg.n_uavs, 1)
            goal += 0.75 * env.cfg.capture_radius * np.array([np.cos(angle), np.sin(angle)])
        else:
            angle = 2.0 * np.pi * (i / max(env.cfg.n_uavs, 1) + 0.08 * np.sin(env.t / 12))
            goal = np.array([env.cfg.grid_size / 2, env.cfg.grid_size / 2]) + 0.35 * env.cfg.grid_size * np.array([np.cos(angle), np.sin(angle)])
        delta = goal - env.positions[i].astype(float)
        desired_yaw = np.arctan2(delta[1], delta[0])
        yaw = float(getattr(env, "yaw_angles", env.headings * np.pi / 4.0)[i])
        yaw_error = (desired_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
        if dim == 2:
            desired_speed = min(1.0, np.linalg.norm(delta) / 3.0)
            current = env.forward_speeds[i] / max(env.cfg.max_forward_speed, 1e-6)
            actions[i] = [np.clip(2.0 * (desired_speed - current), -1, 1), np.clip(yaw_error, -1, 1)]
        else:
            direction = delta / max(np.linalg.norm(delta), 1e-8)
            actions[i, :2] = direction
            dz = float(env.target_altitudes[0] - env.altitudes[i])
            actions[i, 2] = np.clip(dz / 2.0, -1.0, 1.0)
            actions[i, 3] = np.clip(yaw_error, -1.0, 1.0)
    return actions


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
body{margin:0;background:#f4f5f7;color:#1b2430;font:15px "Times New Roman",serif}
.wrap{display:grid;grid-template-columns:minmax(620px,1fr) 310px;gap:14px;padding:14px;height:calc(100vh - 28px)}
.panel{background:white;border:1px solid #c8ced8;border-radius:10px;box-shadow:0 3px 12px #0001}
canvas{width:100%;height:100%;display:block;border-radius:10px}.side{padding:16px;overflow:auto}
h1{font-size:22px;margin:0 0 12px;color:#0d2c78}.controls{display:grid;grid-template-columns:1fr 1fr;gap:8px}
button,select,input{font:14px Arial;padding:7px}.wide{grid-column:1/-1}table{width:100%;border-collapse:collapse;margin-top:12px}
td{padding:4px;border-bottom:1px solid #e8ebef}.legend{margin-top:14px;line-height:1.75}
.sw{display:inline-block;width:20px;height:3px;vertical-align:middle;margin-right:7px}
</style></head><body><div class="wrap">
<div class="panel"><canvas id="scene"></canvas></div>
<div class="panel side"><h1>__TITLE__</h1>
<div class="controls"><button id="play">Pause</button><button id="restart">Restart</button>
<label class="wide">Frame <input id="slider" type="range" min="0" value="0" style="width:75%"></label>
<label>Speed <select id="speed"><option value="180">0.5×</option><option value="90" selected>1×</option><option value="45">2×</option></select></label>
<label>View <select id="view"><option value="top">Top</option><option value="iso">3-D/isometric</option></select></label></div>
<table id="stats"></table><div class="legend">
<div><span class="sw" style="background:#2455bd"></span>UAV / trajectory</div>
<div><span class="sw" style="background:#d9272e"></span>evader / trajectory</div>
<div><span class="sw" style="background:#25a55f"></span>active communication</div>
<div><span class="sw" style="background:#6f42c1"></span>target belief and uncertainty</div>
<div><span class="sw" style="background:#4aa3df"></span>sensor field of view (not measured points)</div>
<div id="captureLegend">Gray blocks: buildings; dotted circle: capture radius.</div></div></div></div>
<script>
const trace=__TRACE__, frames=trace.frames||[], cvs=document.getElementById('scene'),ctx=cvs.getContext('2d');
const slider=document.getElementById('slider'),play=document.getElementById('play'),view=document.getElementById('view');
if(trace.mode==='3d')view.value='iso';
slider.max=Math.max(0,frames.length-1);let k=0,running=true,last=0;
const colors=['#1649b7','#ec8b19','#169c62','#8a3fc5','#008b9a','#ba3d8c'];
function resize(){const r=cvs.getBoundingClientRect(),d=devicePixelRatio||1;cvs.width=r.width*d;cvs.height=r.height*d;ctx.setTransform(d,0,0,d,0,0);draw()}
function project(p,z,G,W,H,mode){if(mode==='iso'){const s=Math.min(W/(G*1.55),H/(G*1.12));return [W*.5+(p[0]-p[1])*s*.72,H*.83-(p[0]+p[1])*s*.32-(z||0)*s*.72,s]}const s=Math.min((W-60)/G,(H-60)/G);return [30+p[0]*s,H-30-p[1]*s,s]}
function line(a,b,color,w=1,dash=[]){ctx.save();ctx.strokeStyle=color;ctx.lineWidth=w;ctx.setLineDash(dash);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();ctx.restore()}
function draw(){if(!frames.length)return;const f=frames[k],W=cvs.clientWidth,H=cvs.clientHeight,G=f.grid_size||25,mode=view.value;
ctx.clearRect(0,0,W,H);ctx.fillStyle='#fbfcfe';ctx.fillRect(0,0,W,H);
// floor grid
for(let i=0;i<=G;i+=2){line(project([i,0],0,G,W,H,mode),project([i,G],0,G,W,H,mode),'#e8ebf0');line(project([0,i],0,G,W,H,mode),project([G,i],0,G,W,H,mode),'#e8ebf0')}
// buildings/prisms
(f.buildings||[]).forEach((r,j)=>{const h=(f.building_heights||[])[j]||0,pts=[[r[0],r[1]],[r[2],r[1]],[r[2],r[3]],[r[0],r[3]]];
 const base=pts.map(p=>project(p,0,G,W,H,mode)),top=pts.map(p=>project(p,h,G,W,H,mode));ctx.fillStyle='#747b86aa';ctx.beginPath();base.forEach((p,n)=>n?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath();ctx.fill();
 if(mode==='iso'&&h){ctx.fillStyle='#535b68bb';ctx.beginPath();top.forEach((p,n)=>n?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath();ctx.fill();for(let n=0;n<4;n++)line(base[n],top[n],'#414854',1.2)}
});
// trails
for(let a=0;a<(f.positions||[]).length;a++){ctx.beginPath();for(let q=0;q<=k;q++){const fr=frames[q],p=project(fr.positions[a],(fr.altitudes||[])[a]||0,G,W,H,mode);q?ctx.lineTo(...p):ctx.moveTo(...p)}ctx.strokeStyle=colors[a%colors.length]+'aa';ctx.lineWidth=2;ctx.stroke()}
ctx.beginPath();for(let q=0;q<=k;q++){const fr=frames[q],p=project(fr.targets[0],(fr.target_altitudes||[])[0]||0,G,W,H,mode);q?ctx.lineTo(...p):ctx.moveTo(...p)}ctx.strokeStyle='#d9272ecc';ctx.lineWidth=2.5;ctx.stroke();
// communication
const adj=f.comm_adjacency||[];for(let i=0;i<adj.length;i++)for(let j=i+1;j<adj.length;j++)if(adj[i][j]>0){line(project(f.positions[i],(f.altitudes||[])[i]||0,G,W,H,mode),project(f.positions[j],(f.altitudes||[])[j]||0,G,W,H,mode),'#25a55f99',2,[6,4])}
// line-of-sight status to the evader
const vis=f.direct_visibility||[];(f.positions||[]).forEach((p,i)=>{line(project(p,(f.altitudes||[])[i]||0,G,W,H,mode),project(f.targets[0],(f.target_altitudes||[])[0]||0,G,W,H,mode),(vis[i]&&vis[i][0])?'#25a55faa':'#d9272e55',1.3,(vis[i]&&vis[i][0])?[]:[3,5])});
// Lidar coverage boundaries in the actual mounted sensor frame, not point clouds.
(f.lidar_poses||[]).forEach(pose=>{for(const elevation of f.lidar_vertical_fov_deg){ctx.beginPath();
for(let j=0;j<=48;j++){const az=(-.5+j/48)*f.lidar_horizontal_fov_deg*Math.PI/180,el=elevation*Math.PI/180;
const local=[f.lidar_range*Math.cos(el)*Math.cos(az),f.lidar_range*Math.cos(el)*Math.sin(az),f.lidar_range*Math.sin(el)];
const p=pose.origin.map((v,d)=>v+pose.rotation[d].reduce((s,a,k)=>s+a*local[k],0)),q=project(p,p[2],G,W,H,mode);
j?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1])}ctx.strokeStyle='#4aa3df66';ctx.lineWidth=1;ctx.stroke()}});
// beliefs
const beliefs=f.target_belief_mean||[],cov=f.target_belief_covariance||[];beliefs.forEach((b,i)=>{if(!b.length)return;const p=project(b[0],b[0][2]||0,G,W,H,mode),s=p[2],P=(cov[i]&&cov[i][0])||[],vx=(P[0]&&P[0][0])||.1,vy=(P[1]&&P[1][1])||.1,cx=Math.sqrt(Math.max(vx,.02))*s*2,cy=Math.sqrt(Math.max(vy,.02))*s*2;ctx.save();ctx.strokeStyle='#6f42c1aa';ctx.setLineDash([4,3]);ctx.beginPath();ctx.ellipse(p[0],p[1],cx,cy,0,0,Math.PI*2);ctx.stroke();ctx.restore()});
const hasDistanceCapture=f.capture_radius!==undefined;
document.getElementById('captureLegend').textContent='Gray blocks: buildings; dotted rings: capture sphere (3-D distance rule).';
const tp=project(f.targets[0],(f.target_altitudes||[])[0]||0,G,W,H,mode);{const c=[...f.targets[0].slice(0,2),(f.target_altitudes||[])[0]||0],r=f.capture_radius??trace.capture_radius??1;ctx.save();ctx.strokeStyle='#d9272e88';ctx.setLineDash([3,3]);for(const axes of [[0,1],[0,2],[1,2]]){ctx.beginPath();for(let j=0;j<=64;j++){const a=j*Math.PI/32,p=c.slice();p[axes[0]]+=r*Math.cos(a);p[axes[1]]+=r*Math.sin(a);const q=project(p,p[2],G,W,H,mode);j?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1])}ctx.stroke()}ctx.restore()}ctx.fillStyle='#d9272e';ctx.beginPath();ctx.arc(tp[0],tp[1],7,0,Math.PI*2);ctx.fill();if((f.target_altitudes||[]).length){ctx.fillStyle='#8d1117';ctx.fillText('E z='+(f.target_altitudes[0]||0).toFixed(1),tp[0]+9,tp[1]+14)}
// UAVs
(f.positions||[]).forEach((p,i)=>{const q=project(p,(f.altitudes||[])[i]||0,G,W,H,mode);ctx.fillStyle=colors[i%colors.length];ctx.beginPath();ctx.moveTo(q[0]+9,q[1]);ctx.lineTo(q[0]-7,q[1]-6);ctx.lineTo(q[0]-7,q[1]+6);ctx.closePath();ctx.fill();ctx.fillStyle='#111';const z=(f.altitudes||[]).length?' z='+(f.altitudes[i]||0).toFixed(1):'';ctx.fillText('U'+(i+1)+z,q[0]+9,q[1]-7)});
const meanZ=(f.altitudes||[]).length?(f.altitudes.reduce((a,b)=>a+b,0)/f.altitudes.length).toFixed(2):'fixed';
document.getElementById('stats').innerHTML=[['Mode',trace.mode==='3d'?'3-D continuous':'2-D fixed altitude'],['Control',f.execution_mode||'velocity_yaw_rate'],['Sensor',f.sensor_mode||'lidar'],['Step',f.step],['Step reward',(f.reward||0).toFixed(3)],['Mean UAV altitude',meanZ],['Evader altitude',(f.target_altitudes||[]).length?(f.target_altitudes[0]||0).toFixed(2):'fixed'],['Observed',String((f.direct_visibility||[]).some(r=>r[0]))],...(f.sensor_mode==='lidar'?[['Lidar vertical FOV',f.lidar_vertical_fov_deg.join(' to ')+' deg'],['Detection fraction',((f.lidar_detection_ratio||0)*100).toFixed(1)+'%']]:[]),...(hasDistanceCapture?[['Capture gap',Number(f.minimum_capture_gap).toFixed(3)],['Controller feasible',((f.controller_feasible_rate??0)*100).toFixed(1)+'%'],['Safety interventions',f.continuous_safety_interventions||0]]:[['Capture hold',f.capture_hold_count||0],['Close UAVs',f.capture_close_uavs||0],['Angular span',(f.capture_angular_span_deg||0).toFixed(1)+'°']]),['Captured',f.capture_success?'YES':'no'],['View',mode]].map(r=>`<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join('');slider.value=k}
function tick(t){if(running&&t-last>+document.getElementById('speed').value){k=(k+1)%frames.length;last=t;draw()}requestAnimationFrame(tick)}
play.onclick=()=>{running=!running;play.textContent=running?'Pause':'Play'};document.getElementById('restart').onclick=()=>{k=0;draw()};slider.oninput=()=>{k=+slider.value;draw()};view.onchange=draw;
addEventListener('resize',resize);resize();requestAnimationFrame(tick);
</script></body></html>"""
