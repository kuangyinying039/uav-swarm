"""Preserve CLI failure status when Isaac plugin shutdown exits the worker at 0."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import signal
import sys


def invoke(main, module, *, timeout_seconds=None):
    if os.environ.get('PURSUIT_RUNTIME_WORKER') == module:
        main()
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', type=Path)
    args, _ = parser.parse_known_args()
    env = dict(os.environ, PURSUIT_RUNTIME_WORKER=module)
    command=[sys.executable, '-m', module, *sys.argv[1:]]
    if timeout_seconds is None:
        code = subprocess.call(command, env=env)
    else:
        worker=subprocess.Popen(command,env=env,start_new_session=True)
        try:code=worker.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(worker.pid,signal.SIGTERM)
            try:worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid,signal.SIGKILL);worker.wait()
            raise SystemExit(124)
    if code:
        raise SystemExit(code if code > 0 else 128-code)
    if args.output is None or '--help' in sys.argv or '-h' in sys.argv:
        raise SystemExit(0)
    try:
        result = json.loads((args.output/'result.json').read_text())
        passed = result.get('status') in ('passed', 'ready')
        passed &= not (args.output/'failure.txt').exists()
    except (OSError, ValueError):
        passed = False
    if not passed:
        print(f'Runtime failed or incomplete; inspect {args.output}/result.json and failure.txt',
              file=sys.stderr, flush=True)
    raise SystemExit(0 if passed else 1)
