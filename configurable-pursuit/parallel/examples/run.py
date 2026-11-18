"""Bounded real-physics acceptance entry point; no long training modes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import torch
from parallel.env.dependencies import resolve_source, select_source


def write(path, data): path.write_text(json.dumps(data, indent=2, default=str)+'\n')


def preflight():
    result = dict(cuda_available=torch.cuda.is_available(), nvidia_devices=[str(p) for p in Path('/dev').glob('nvidia*')],
                  load_average=os.getloadavg(), cpu_count=os.cpu_count())
    probe = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=15)
    result.update(nvidia_smi_exit=probe.returncode, gpu_processes=probe.stdout.strip(), gpu_error=probe.stderr.strip())
    result['ready'] = result['cuda_available'] and probe.returncode == 0 and not result['gpu_processes']
    result['ready'] &= result['load_average'][0] < max(2, result['cpu_count']*.75)
    return result


def fingerprint(output, upstream=None):
    root = Path(__file__).resolve().parents[1]
    snapshot = output/'source'; snapshot.mkdir()
    paths = sorted(p for p in root.rglob('*') if p.suffix in ('.py', '.json')
                   and not any(part in p.relative_to(root).parts for part in ('__pycache__','result','render'))
                   and not p.resolve().is_relative_to(output.resolve()))
    for p in paths:
        target = snapshot/p.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    upstream = resolve_source(upstream)
    paths += [upstream/'omni_drones/robots/drone/multirotor.py', upstream/'omni_drones/robots/assets/usd/hummingbird.usd',
              upstream/'omni_drones/robots/assets/usd/hummingbird.yaml', upstream/'omni_drones/controllers/lee_position_controller.py']
    write(output/'fingerprints.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()})
    write(output/'dependencies.json', dict(torch=torch.__version__, cuda=torch.version.cuda, isaac='4.1.0',
                                         upstream=str(upstream), python=__import__('sys').version))


class Stats:
    def __init__(self):
        self.counts = {}; self.maxima = {}; self.step_seconds = 0.; self.reset_seconds = 0.; self.resets = 0
        self.reset_profile = {}
        self.episode_overlap = None; self.completed_overlap_episodes = 0

    def add(self, info):
        if self.episode_overlap is None:
            self.episode_overlap = torch.zeros_like(info['captured'])
        self.episode_overlap |= info['p_swept_overlap'].any(dim=(-1,-2))
        self.counts['captures_in_overlap_episodes'] = self.counts.get('captures_in_overlap_episodes',0)+int((info['captured'] & self.episode_overlap).sum())
        for key in ('captured', 'success', 'static_collision', 'dynamic_collision', 'boundary_collision', 'timeout',
                    'p_boundary_violation', 'overlap_entered', 'opposite_side_exit', 'escape_timeout',
                    'p_through_crossing', 'constraint_failure', 'physical_instability', 'physical_failure'):
            self.counts[key] = self.counts.get(key, 0)+int(info[key].sum())
        self.counts['captures_during_escape'] = self.counts.get('captures_during_escape', 0)+int((info['capture_mask'] & info['escaping']).sum())
        self.counts['overlap_sample_count'] = self.counts.get('overlap_sample_count', 0)+int(info['p_swept_overlap'].sum())
        for key in ('p_boundary_excess', 'overlap_depth', 'overlap_duration', 'p_swept_depth', 'constraint_residual'):
            self.maxima[key] = max(self.maxima.get(key, 0.), float(info[key].max()) if info[key].numel() else 0.)
        self.counts['altitude_constraint_failure'] = self.counts.get('altitude_constraint_failure',0)+int(info.get('altitude_constraint_failure', torch.zeros(1)).sum())
        s = info['state']
        self.maxima['height_error'] = max(self.maxima.get('height_error', 0.), float((s[..., 2]-info['target_height']).abs().max()))

    def finish_episodes(self, ids):
        if self.episode_overlap is not None:
            self.completed_overlap_episodes += int(self.episode_overlap[ids].sum())
            self.episode_overlap[ids] = False

    def accepted(self):
        return not any(self.counts.get(k, 0) for k in ('p_boundary_violation','opposite_side_exit','p_through_crossing','escape_timeout','constraint_failure','physical_failure','altitude_constraint_failure')) and self.maxima.get('p_swept_depth',0) <= .15

    def result(self):
        return dict(counts=self.counts, maxima=self.maxima,
                    overlap_episodes_observed=self.completed_overlap_episodes+(int(self.episode_overlap.sum()) if self.episode_overlap is not None else 0),
                    counter_units='physical_failure: environment-step entries; physical_instability: aircraft-step entries; other boolean entries per step except overlap_entered/opposite_side_exit are transitions; durations in seconds',
                    ordinary_step_seconds=self.step_seconds,
                    randomized_reset_seconds=self.reset_seconds, randomized_reset_env_count=self.resets,
                    control_acceptance=self.accepted(), reset_profile_seconds=self.reset_profile)


def reset_timed(env, ids, stats):
    torch.cuda.synchronize(); t = time.perf_counter()
    obs = env.reset(ids)
    merged_profile={**env.unwrapped.rules.last_reset_profile, **env.unwrapped.last_reset_profile}
    for key,value in merged_profile.items(): stats.reset_profile[key]=stats.reset_profile.get(key,0.)+value
    stats.finish_episodes(ids)
    torch.cuda.synchronize(); stats.reset_seconds += time.perf_counter()-t; stats.resets += len(ids)
    return obs


def check_loop(env, args, output):
    stats = Stats(); traces = []
    action = torch.zeros(env.n, 2, device=env.device); action[:, 0] = .8
    before = env.state(); bg = env.unwrapped.rules.bg.clone(); steps = env.unwrapped.rules.steps.clone()
    ids = torch.arange(0, env.n, 2, device=env.device); others = torch.arange(1, env.n, 2, device=env.device)
    reset_timed(env, ids, stats)
    after = env.state()
    error = float((after[others]-before[others]).abs().max()) if len(others) else 0.
    assert error < 1e-6 and torch.equal(bg[others], env.unwrapped.rules.bg[others]) and torch.equal(steps[others], env.unwrapped.rules.steps[others])
    assert float((after[ids,:,:3]-env.unwrapped.rules.initial_pos[ids]).abs().max()) < 1e-4
    for step in range(args.steps):
        # Same 32-step reset schedule for all ladder sizes, plus genuine terminals.
        static = env.unwrapped.rules.static[:8].clone(); bg = env.unwrapped.rules.bg[:8].clone(); goal = env.unwrapped.rules.goal[:8].clone()
        torch.cuda.synchronize(); t = time.perf_counter()
        obs, reward, term, trunc, info = env.step(action)
        torch.cuda.synchronize(); stats.step_seconds += time.perf_counter()-t
        assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
        stats.add(info)
        if args.mode == 'check':
            traces.append(dict(state=info['state'][:8].cpu(), static=static.cpu(), bg=bg.cpu(), goal=goal.cpu(),
                               captured=info['captured'][:8].cpu(), success=info['success'][:8].cpu(), physical_failure=info['physical_failure'][:8].cpu(),
                               physical_instability=info['physical_instability'][:8].cpu(),
                               physical_height_low=info['physical_height_low'][:8].cpu(), physical_height_high=info['physical_height_high'][:8].cpu(),
                               physical_excessive_speed=info['physical_excessive_speed'][:8].cpu(),
                               pre_projection_height=info['pre_projection_height'][:8].cpu(), pre_projection_vz=info['pre_projection_vz'][:8].cpu(),
                               altitude_correction=info['altitude_correction'][:8].cpu(), reward=reward[:8].cpu(), reward_terms=info['reward_terms'][:8].cpu(), done=(term|trunc)[:8].cpu()))
        ids = torch.arange(env.n, device=env.device) if args.reset_interval and (step+1)%args.reset_interval == 0 else (term|trunc).nonzero().flatten()
        if len(ids): reset_timed(env, ids, stats)
        if (step+1)%32 == 0: print(json.dumps(dict(step=step+1, **stats.result())), flush=True)
    if traces: torch.save(traces, output/'trajectories.pt')
    result = stats.result()
    result.update(indexed_reset_untouched_error=error, environment_steps_per_second=env.n*args.steps/(stats.step_seconds+stats.reset_seconds),
                  timing='CUDA synchronized, ordinary step + all resets; includes reference RNG and geometry sampling, excludes build')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--omnidrones-source', type=Path)
    p.add_argument('--mode', choices=['preflight','check','bench'], default='check')
    p.add_argument('--num-envs', type=int, default=8)
    p.add_argument('--steps', type=int, default=128)
    p.add_argument('--reset-interval',type=int,default=32)
    p.add_argument('--seed', type=int, default=41)
    p.add_argument('--max-episode-steps', type=int, default=None, help='Explicit override of task.max_episode_steps')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--visualize', action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--sampler-backend', choices=['cpu'], default='cpu')
    p.add_argument('--profile-reset', action='store_true')
    p.add_argument('--flight-mode', choices=['planar','free_3d'], default=None)
    p.add_argument('--pursuer-config', type=Path, help='Project pursuer JSON; defaults to parallel/config/pursuer.json')
    p.add_argument('--environment-config',type=Path,help='Project environment JSON')
    p.add_argument('--visualization-config',type=Path,help='Project visualization JSON')
    p.add_argument('--headless',action=argparse.BooleanOptionalAction,default=None)
    args = p.parse_args()
    from parallel.env.config_loader import load_settings, load_visualization, configure_isaac_app
    environment_config=load_settings(args.environment_config)
    visualization_config=load_visualization(args.visualization_config)
    if visualization_config['recording']['enabled']:
        p.error('Recording is supported by python -m parallel.examples.demo; use that entry point')
    if args.headless is not None: environment_config['simulation']['headless']=args.headless
    if args.flight_mode is not None: environment_config['simulation']['flight_mode']=args.flight_mode
    if args.visualize is not None: visualization_config['enabled']=args.visualize
    environment_config=load_settings(environment_config)
    from parallel.strategies.config import load_config
    pursuer_config = load_config(args.pursuer_config)[0]
    if not 1 <= args.num_envs <= 1024 or not 1 <= args.steps <= (1024 if args.mode=='check' else 256): p.error('N must be 1..1024; steps must be 1..1024 for check or 1..256 otherwise')
    if args.reset_interval < 0 or (args.mode=='bench' and args.reset_interval!=32): p.error('bench uses fixed reset interval32; interval must be nonnegative')
    if args.max_episode_steps is not None and args.max_episode_steps <= 0: p.error('max episode steps must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    source = resolve_source(args.omnidrones_source)
    write(args.output/'args.json',vars(args)); fingerprint(args.output, source)
    app = None
    env = None
    try:
        resource = preflight(); write(args.output/'preflight.json',resource)
        if args.mode == 'preflight':
            write(args.output/'result.json',dict(status='ready' if resource['ready'] else 'blocked', **resource)); return
        if not resource['ready']: raise RuntimeError('GPU preflight blocked: unavailable GPU, existing compute workload or CPU load; see preflight.json')
        # Force requested dependency in this process; do not edit shared activation.
        import sys
        select_source(source)
        from omegaconf import OmegaConf
        from omni_drones import init_simulation_app
        cfg = OmegaConf.load(source/'examples/demo.yaml')
        configure_isaac_app(cfg,environment_config['simulation'])
        OmegaConf.save(cfg,args.output/'sim.yaml')
        torch.set_num_threads(4); torch.manual_seed(args.seed)
        # Account-local portable cache; never write public installation cache.
        private_cache = Path(__file__).resolve().parents[2] / 'runtime/isaac-kit'
        private_cache.mkdir(parents=True, exist_ok=True)
        sys.argv.extend(['--portable-root', str(private_cache)])
        app = init_simulation_app(cfg)
        from parallel.env.od_env import ParallelEnv
        from parallel.env.wrapper import ScriptedPursuerWrapper
        t = time.perf_counter(); env = ParallelEnv(cfg,range(args.seed,args.seed+args.num_envs),args.max_episode_steps,args.visualize,sampler_backend=args.sampler_backend,flight_mode=args.flight_mode,environment_config=environment_config,pursuer_config=pursuer_config,visualization_config=visualization_config)
        env = ScriptedPursuerWrapper(env, pursuer_config=pursuer_config)
        env.set_reset_profiling(args.profile_reset)
        torch.cuda.synchronize(); build_seconds = time.perf_counter()-t
        effective = dict(args=vars(args), effective_geometry=env.unwrapped.rules.cfg,
                         runtime=dict(dynamics=env.unwrapped.rules.settings['simulation']['drone_model']+' rigid-body + rotor dynamics', controller=env.unwrapped.rules.settings['simulation']['controller'],
                                      dt=env.unwrapped.rules.cfg['dt'], target_height=env.unwrapped.rules.settings['simulation']['target_height'], planar_command_limit=env.unwrapped.rules.settings['simulation']['evader_speed_limit'], reset_motion=env.unwrapped.rules.reset_motion, pursuer_strategy='configured_per_episode',
                                      shadow_ttc=False, reference_2d_dynamics_used=False), task_geometry=dict(type='swept circles',evader_radius=env.unwrapped.rules.cfg['robot_radius'],pursuer_radius=env.pursuer.config['spawn']['radius']),
                         success_requires_no_collision=True, p_boundary=dict(axis='local y upper only',limit=env.pursuer.config['spawn']['region_half_extent']),
                         safety={k:getattr(env.safety,k) for k in ('braking_acceleration','lag_seconds','boundary_margin','escape_seconds','escape_speed','hard_static_margin')})
        effective['observation_layout'] = env.observation_layout
        effective['environment'] = env.unwrapped.rules.settings
        effective['visualization'] = env.unwrapped.visualization_config
        effective['reward_term_names'] = env.unwrapped.rules.reward.term_names
        effective['pursuer_policy'] = env.pursuer.config
        effective['pursuer_probabilities'] = dict(zip(env.pursuer.strategy_names,env.pursuer.probabilities))
        write(args.output/'effective_config.json',effective)
        result = check_loop(env,args,args.output)
        result.update(status='passed' if result['control_acceptance'] else 'failed', build_seconds=build_seconds,
                      peak_torch_memory_mb=torch.cuda.max_memory_allocated()/2**20,
                      gpu_memory_sample=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader'],text=True).strip(),
                      gpu_memory_scope='nvidia-smi total GPU at end; torch peak excludes PhysX/USD', random_resets=env.unwrapped.reset_count)
        write(args.output/'result.json',result)
        if result['status'] != 'passed': raise RuntimeError('Engineering acceptance failed; diagnostics retained in result.json')
    except BaseException:
        if not (args.output/'result.json').exists():
            write(args.output/'result.json',dict(status='blocked' if not torch.cuda.is_available() else 'failed',
                                               physical_acceptance=False))
        (args.output/'failure.txt').write_text(traceback.format_exc())
        traceback.print_exc(); __import__('sys').stderr.flush()
        raise
    finally:
        try:
            if env is not None: env.close()
        finally:
            if app is not None: app.close()


if __name__ == '__main__':
    from parallel.tools.runtime_entry import invoke
    invoke(main, 'parallel.examples.run')
