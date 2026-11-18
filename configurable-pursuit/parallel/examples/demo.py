"""公开基线展示入口：单局真实Isaac物理、可选窗口/仿真相机MP4，无模型加载。"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import uuid
import torch
from parallel.env.config_loader import load_settings,load_visualization,configure_isaac_app
from parallel.strategies.config import load_config
from parallel.examples.evader_baselines import action
from parallel.examples.demo_render import DemoCamera,VideoRecorder, create_demo_app
from parallel.env.dependencies import select_source, project_output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--environment-config',type=Path)
    p.add_argument('--pursuer-config',type=Path)
    p.add_argument('--visualization-config',type=Path)
    p.add_argument('--omnidrones-source',type=Path)
    p.add_argument('--evader',choices=['hold','goal','goal_avoid'],default='goal_avoid')
    p.add_argument('--follow-evader',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--zoom',type=float,help='Image magnification, e.g. 2 for double size')
    p.add_argument('--view',choices=['top','oblique'],help='Override configured camera view')
    p.add_argument('--seed',type=int,default=41)
    p.add_argument('--episodes',type=int,default=1)
    p.add_argument('--max-steps',type=int)
    p.add_argument('--headless',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--record',action=argparse.BooleanOptionalAction,default=None)
    args=p.parse_args()
    if args.episodes<1 or (args.max_steps is not None and args.max_steps<1):p.error('episodes/max-steps must be positive')
    settings=load_settings(args.environment_config);visuals=load_visualization(args.visualization_config)
    pursuer=load_config(args.pursuer_config)[0]
    if args.view is not None:visuals['camera']['view']=args.view
    if args.follow_evader is not None:visuals['camera']['follow_evader']=args.follow_evader
    if args.zoom is not None:visuals['camera']['zoom']=args.zoom
    visuals=load_visualization(visuals)
    if args.headless is not None:settings['simulation']['headless']=args.headless
    if args.record is not None:visuals['recording']['enabled']=args.record
    # This dedicated demo requests display geometry; per-group visibility still applies.
    visuals['enabled']=True
    record=visuals['recording']
    if record['enabled'] and shutil.which('ffmpeg') is None:p.error('Recording needs ffmpeg with libx264 on PATH')
    try:source=select_source(args.omnidrones_source)
    except ValueError as exc:p.error(str(exc))
    if not torch.cuda.is_available():raise RuntimeError('This demo requires the real Isaac CUDA physics environment')
    probe=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
    if probe.stdout.strip():raise RuntimeError('GPU already has a compute workload; run demo when GPU is free')
    from omegaconf import OmegaConf
    cfg=OmegaConf.load(source/'examples/demo.yaml');configure_isaac_app(cfg,settings['simulation'])
    output=None
    if record['enabled']:
        output=project_output(record['output_dir'])/f'demo-{uuid.uuid4().hex[:12]}'
        output.mkdir(parents=True,exist_ok=False)
    app=None;env=None;camera=None;writer=None;report=[];render_details={}
    try:
        print("[demo] Starting Isaac app",flush=True)
        torch.set_num_threads(4)
        cache=Path(__file__).resolve().parents[2]/'runtime/isaac-kit'
        cache.mkdir(parents=True,exist_ok=True)
        sys.argv.extend(['--portable-root',str(cache)])
        app=create_demo_app(cfg,visuals)
        from omni.isaac.core.utils.extensions import enable_extension
        enable_extension('omni.isaac.sensor')
        from parallel.env.od_env import ParallelEnv
        from parallel.env.wrapper import ScriptedPursuerWrapper
        print("[demo] Creating environment",flush=True)
        base=ParallelEnv(cfg,[args.seed],environment_config=settings,pursuer_config=pursuer,visualization_config=visuals)
        env=ScriptedPursuerWrapper(base)
        print("[demo] Creating camera",flush=True)
        camera=DemoCamera(env,visuals)
        # Scene/render-product creation can restore stage defaults (e.g. DLSS).
        # Apply the requested native-resolution renderer AFTER those are created.
        app.reset_render_settings()
        import omni.replicator.core as rep
        # Render-product initialization may override launch-time AA on frame one.
        # Replicator settings are authored into its render graph, not only startup.
        rep.settings.set_render_rtx_realtime(antialiasing='FXAA')
        import carb
        render_details=dict(anti_aliasing_op=carb.settings.get_settings().get('/rtx/post/aa/op'),
                            renderer=carb.settings.get_settings().get('/rtx/rendermode'))
        dt=settings['simulation']['dt'];limit=args.max_steps or settings['task']['max_episode_steps']
        for episode in range(args.episodes):
            if episode:env.reset(seeds=[args.seed+episode])
            print(f"[demo] Warming camera for episode {episode+1}",flush=True)
            first=camera.warmup()
            print("[demo] Running episode",flush=True)
            if record['enabled']:
                writer=VideoRecorder(output/f'episode-{episode+1:04d}.mp4',fps=record['fps'],resolution=visuals['camera']['resolution'])
                writer.append(first,0.)
            outcome='step_limit';steps=0
            for step in range(limit):
                if not app.is_running():outcome='window_closed';break
                _,_,terminated,truncated,info=env.step(action(env,args.evader))
                frame=camera.frame();steps=step+1
                if writer is not None:writer.append(frame,steps*dt)
                if bool((terminated|truncated)[0]):
                    outcome=info['outcome_names'][int(info['outcome'][0])];break
            frames=writer.frames if writer is not None else 0
            if writer is not None:writer.close();writer=None
            result=dict(episode=episode+1,seed=args.seed+episode,evader=args.evader,
                        outcome=outcome,steps=steps,simulation_seconds=steps*dt,video_frames=frames,
                        pursuer_strategies=[[env.pursuer.strategy_names[int(i)] for i in row] for row in env.pursuer.strategy_ids.cpu().tolist()])
            report.append(result);print(json.dumps(result),flush=True)
            if outcome=='window_closed':break
    except BaseException:
        print("[demo] FAILED before cleanup; original traceback:",file=sys.stderr,flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        try:
            if writer is not None:writer.close(success=False)
        finally:
            if output is not None:
                # Persist the public demo configuration and episode summary.
                (output/'effective_config.json').write_text(json.dumps(dict(environment=settings, pursuer=pursuer, visualization=visuals,
                    strategy_probabilities=dict(zip(env.pursuer.strategy_names,env.pursuer.probabilities)) if env is not None else {}),indent=2)+'\n')
                (output/'summary.json').write_text(json.dumps(dict(public_evader=args.evader,
                    dt=settings['simulation']['dt'],fps=record['fps'],resolution=visuals['camera']['resolution'],render_details=render_details,video_crf=18,
                    output_dir=str(output),episodes=report),indent=2)+'\n')
            if camera is not None:camera.close()
            if env is not None:env.close()
            if app is not None:app.close()
    if output is not None:print(f'Video output: {output.resolve()}',flush=True)


if __name__=='__main__':main()
