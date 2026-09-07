"""Time the current pursuit environment without training."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from quadrotor_pursuit_env import QuadrotorPursuitConfig, QuadrotorPursuitEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--env-config', type=Path)
    parser.add_argument('--out', type=Path, default=Path('outputs/pursuit_speed.json'))
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('steps must be positive')
    overrides = json.loads(args.env_config.read_text(encoding='utf-8')) if args.env_config else {}
    env = QuadrotorPursuitEnv(QuadrotorPursuitConfig(**overrides))
    actions = np.random.default_rng(3).uniform(-.5, .5, (args.steps, env.cfg.n_uavs, 4))
    elapsed, frames, count = 0., [], 0
    for action in actions:
        started = time.perf_counter()
        result = env.step_joint(action)
        elapsed += time.perf_counter() - started
        count += 1
        if args.render:
            from marl_trainers import capture_env_frame
            frames.append(capture_env_frame(env, result, action))
        if result['terminated'] or result['truncated']:
            break
    payload = {'steps': count, 'seconds_per_environment_step': elapsed/count,
               'estimated_300_step_environment_seconds': 300*elapsed/count,
               'note': 'Environment only; excludes policy inference, PPO updates, validation and saving. No capture-rate claim.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    if frames:
        from pursuit_visualization import write_pursuit_animation_html
        write_pursuit_animation_html(args.out.with_suffix('.html'),
                                     {'episode': 0, 'mode': '3d', 'frames': frames},
                                     'MID-360 | velocity + yaw rate | interface check')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
