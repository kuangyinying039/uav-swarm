"""Render a baseline or trained pursuit checkpoint in the common 3-D environment."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from artifact_paths import artifact_path, default_output
from quadrotor_pursuit_env import QuadrotorPursuitConfig, QuadrotorPursuitEnv
from pursuit_baselines_3d import BASELINES_3D
from marl_trainers import capture_env_frame
from pursuit_visualization import write_pursuit_animation_html
from train_cooperative_marl import write_episode_scene_3d_svg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--method', choices=['mappo', 'matd3', *sorted(BASELINES_3D)], default='mpc')
    parser.add_argument('--checkpoint', type=artifact_path,
                        help='Required for mappo/matd3; best_capture.pt is recommended')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--env-config', type=Path)
    parser.add_argument('--out-dir', type=artifact_path, default=default_output('visualization_3d'))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('steps must be positive')
    learned = args.method in ('mappo', 'matd3')
    if learned and args.checkpoint is None:
        parser.error('--checkpoint is required for mappo/matd3')
    if not learned and args.checkpoint is not None:
        parser.error('--checkpoint is only valid for mappo/matd3')
    if learned:
        import torch
        from dataclasses import replace
        from train_pursuit_with_demos import load_environment_config, load_evaluation_environment_config

        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        expected = 'mappo' if args.method == 'mappo' else 'hgat_matd3'
        if payload.get('algorithm') != expected:
            parser.error(f"{args.checkpoint} contains {payload.get('algorithm')!r}, expected {expected!r}")
        checkpoint_cfg = load_environment_config(payload['env_config'])
        env_cfg = (load_evaluation_environment_config(args.env_config, checkpoint_cfg)
                   if args.env_config else checkpoint_cfg)
        env_cfg = replace(env_cfg, seed=args.seed, search_steps=args.steps)
        if args.method == 'mappo':
            from marl_trainers import TrainConfig
            from train_pursuit_with_demos import PursuitDemoTrainer, deterministic_action
            train_cfg = TrainConfig(**payload['train_config'])
            train_cfg.device = args.device
            trainer = PursuitDemoTrainer(lambda episode=0: QuadrotorPursuitEnv(env_cfg), train_cfg)
            trainer.load_checkpoint(args.checkpoint)
            action_fn = lambda obs: deterministic_action(trainer, obs)
        else:
            from pursuit.algorithms.matd3 import Matd3Config
            from pursuit.trainers.matd3_trainer import Matd3Trainer
            train_cfg = Matd3Config(**payload['train_config'])
            train_cfg.device = args.device
            trainer = Matd3Trainer(lambda episode=0: QuadrotorPursuitEnv(env_cfg), train_cfg, seed=args.seed)
            trainer.load_checkpoint(args.checkpoint)
            trainer.actor.eval()
            action_fn = trainer.deterministic_action
        env = QuadrotorPursuitEnv(env_cfg)
    else:
        overrides = json.loads(args.env_config.read_text(encoding='utf-8')) if args.env_config else {}
        overrides.update(seed=args.seed, search_steps=args.steps)
        env = QuadrotorPursuitEnv(QuadrotorPursuitConfig(**overrides))
        policy = BASELINES_3D[args.method]()
        policy.reset(env)
        action_fn = lambda obs: policy.actions(env)
    frames = []
    obs = env.observe_search()
    for _ in range(args.steps):
        action = action_fn(obs)
        result = env.step_joint(action)
        frames.append(capture_env_frame(env, result, action))
        obs = result['obs']
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
