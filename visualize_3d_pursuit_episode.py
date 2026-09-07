"""Render a baseline with the same flight model and renderer as MAPPO."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from quadrotor_pursuit_env import QuadrotorPursuitConfig, QuadrotorPursuitEnv
from pursuit_baselines_3d import BASELINES_3D
from marl_trainers import capture_env_frame
from pursuit_visualization import write_pursuit_animation_html
from train_cooperative_marl import write_episode_scene_3d_svg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--method', choices=sorted(BASELINES_3D), default='mpc')
    parser.add_argument('--env-config', type=Path)
    parser.add_argument('--out-dir', type=Path, default=Path('outputs/visualization_3d'))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('steps must be positive')
    overrides = json.loads(args.env_config.read_text(encoding='utf-8')) if args.env_config else {}
    overrides.update(seed=args.seed, search_steps=args.steps)
    env = QuadrotorPursuitEnv(QuadrotorPursuitConfig(**overrides))
    policy = BASELINES_3D[args.method]()
    policy.reset(env)
    frames = []
    for _ in range(args.steps):
        action = policy.actions(env)
        result = env.step_joint(action)
        frames.append(capture_env_frame(env, result, action))
        if result['terminated'] or result['truncated']:
            break
    trace = {'episode': 1, 'mode': '3d', 'env_config': asdict(env.cfg), 'frames': frames}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f'episode_seed{args.seed}_{args.method}'
    write_pursuit_animation_html(path.with_suffix('.html'), trace, f'Cooperative capture | {args.method}')
    write_episode_scene_3d_svg(path.with_suffix('.svg'), args.method, trace)
    path.with_suffix('.json').write_text(json.dumps(trace, indent=2), encoding='utf-8')
    print(path.with_suffix('.html'))


if __name__ == '__main__':
    main()
