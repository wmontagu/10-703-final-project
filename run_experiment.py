import argparse
import os
import subprocess
import sys
import time

def get_dreamer_cmd(args, run_name):
    """
    Constructs the command line for DreamerV3.
    (https://github.com/danijar/dreamerv3)
    """

    task_name = args.env

    # Build the Dreamer command
    cmd = [
        sys.executable,
        'dreamerv3/dreamerv3/main.py',

        # --- Configs ---
        '--configs', 'defaults',
        '--task', f"gym_{task_name}",

        # --- Logging ---
        '--logdir', f'./logdir/dreamer/{run_name}',
        '--logger.outputs', 'jsonl,scope,wandb',

        '--run.steps', str(args.steps),
        '--run.train_ratio', '1024',
        '--run.envs', '1',
        '--run.eval_envs', '1',
        '--run.episode_timeout', '0',
        '--run.debug', 'False',

        # --- Reproducibility ---
        '--seed', str(args.seed),
    ]

    return cmd


def main():
    parser = argparse.ArgumentParser(description="Run RL Algorithm Comparison")

    parser.add_argument('--algo', type=str, required=True,
                        choices=['mbpo', 'dreamer', 'tdmpc'],
                        help="Algorithm to run.")

    parser.add_argument('--env', type=str, required=True,
                        choices=['Reacher-v4', 'HalfCheetah-v4', 'Humanoid-v4'],
                        help="MuJoCo environment name (using Gymnasium v2).")

    parser.add_argument('--seed', type=int, default=int(time.time() % 1e6),
                        help="Random seed.")

    parser.add_argument('--steps', type=int, default=2000,
                        help="Total Dreamer training steps (for short tests).")

    parser.add_argument('--wandb_project', type=str, default="MBPO_vs_Planning",
                        help="Weights & Biases project name.")

    args = parser.parse_args()

    run_name = f"{args.algo}_{args.env}_seed{args.seed}"

    # Select algorithm
    if args.algo == 'dreamer':
        cmd = get_dreamer_cmd(args, run_name)
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    # Execute
    try:
        subprocess.run(cmd, check=True)
        print("Experiment Finished")

    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("Experiment Interrupted by User")


if __name__ == "__main__":
    main()