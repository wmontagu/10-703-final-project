import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from environment import setup_mppi_for_minari, retrain_dynamics_from_buffer, create_dynamics_function
from mppi import train_mppi

"""
Data structure that allows us to correctly intergrate the Minari dataset with the MPPI information we have.
"""
class MinariEnvWrapper:
    def __init__(self, dataset):
        self.dataset = dataset
        self.current_episode = None
        self.current_step = 0
        self.unwrapped = self

    def reset(self):
        self.current_episode = self.dataset.sample_episodes(n_episodes=1)[0]
        self.current_step = 0

        state = self.current_episode.observations[0]
        if isinstance(state, dict):
            self.state = np.concatenate([v.flatten() for v in state.values()])
        else:
            self.state = state.flatten()
        return self.state

    def step(self):
        self.current_step += 1

        if self.current_step >= len(self.current_episode.observations) - 1:
            done = True
            next_state = self.state
            reward = 0.0
        else:
            next_state = self.current_episode.observations[self.current_step]
            if isinstance(next_state, dict):
                next_state = np.concatenate([v.flatten() for v in next_state.values()])
            else:
                next_state = next_state.flatten()

            reward = self.current_episode.rewards[self.current_step - 1]
            done = self.current_episode.terminations[self.current_step - 1] or self.current_episode.truncations[self.current_step - 1]

        self.state = next_state

        if done:
            self.reset()

        return next_state, reward, done, False, {}
    
def plot_simple_curve(rewards, graph_name="mppi_training", graph_dir="./graphs", window_size=50):
    # plot based on HW1, heavily edited
    # plots the moving avg curve (the avergae of the 50 iterations next to it), and raw rewards at each iteration.
    # we use the moving avg, as it better represents the current performance of our agent. it will be better to change the window_size metric
    # to the same number as the dynamic model retrain frequency.
    os.makedirs(graph_dir, exist_ok=True)

    iterations = list(range(len(rewards)))

    fig, ax = plt.subplots(figsize=(10, 6))

    # plot raw rewards
    ax.plot(iterations, rewards, 'b-', alpha=0.3, linewidth=0.5, label='Raw Rewards')

    if len(rewards) > window_size:
        # moving avg indicates the average reward gained over my window size.
        moving_avg = [np.mean(rewards[i:i+window_size]) for i in range(len(rewards) - window_size + 1)]
        ma_iterations = iterations[window_size-1:]
        ax.plot(ma_iterations, moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window_size})')

    ax.set_title(f'{graph_name}', fontsize=14)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(graph_dir, f'{graph_name}.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Graph saved to: {save_path}")


def plot_mppi_cumulative_reward(cummulative_rewards, graph_name="mppi_training", graph_dir="./graphs"):
    # cumulative Rewards for training iterations
    os.makedirs(graph_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot cumulative reward
    ax.plot(range(len(cummulative_rewards)), cummulative_rewards, 'g-', linewidth=2)
    ax.set_title(f'MPPI Cumulative Reward: {graph_name}', fontsize=14)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total Reward')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(graph_dir, f'{graph_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Graph saved to: {save_path}")

def run_dataset(dataset_name, dataset_label, device):
    # Run MPPI training on a single dataset and generate graphs
    
    print(f"running MPPI on {dataset_label}: {dataset_name}")

    print("setting up MPPI controller...", flush=True)
    mppi_controller, dataset, dynamics_model = setup_mppi_for_minari(
        dataset_name=dataset_name,
        num_samples=1000,
        horizon=20,
        lambda_=1.0,
        noise_sigma_scale=0.5,
        device=device,
        train_dynamics=True,
        num_epochs=100
    )

    env_wrapper = MinariEnvWrapper(dataset)
    env_wrapper.reset()

    # The actual retrain_dynmaics loop for the buffer data
    def retrain_dynamics(data):
        print("\nretraining dynamics model...", flush=True)
        retrain_dynamics_from_buffer(
            model=dynamics_model,
            buffer_tensor=data,
            nx=mppi_controller.nx,
            nu=mppi_controller.nu,
            num_epochs=20,
            batch_size=32,
            lr=0.0003,
            device=device
        )
        new_dynamics_fn = create_dynamics_function(dynamics_model, device=device)
        mppi_controller.F = new_dynamics_fn
        print("\ndynamics model retrained", flush=True)

    print("\nstarting MPPI training...", flush=True)
    total_reward, dataset, rewards_per_iteration = train_mppi(
        mppi=mppi_controller,
        env=env_wrapper,
        retrain_dynamics=retrain_dynamics,
        retrain_after_iter=100, # how many iterations after you retrain the dynamics model
        iter=1000, # how many total iterations
        render=False
    )

    print(f"\ntraining completed for {dataset_label}!")
    print(f"total reward: {total_reward:.2f}")
    print(f"\ngenerating training graphs for {dataset_label}...", flush=True)

    # clean label for filenames
    clean_label = dataset_label.replace(" ", "_").replace("(", "").replace(")", "").lower()

    # simple plots below
    plot_simple_curve(
        rewards=rewards_per_iteration,
        graph_name=f"mppi_rewards_{clean_label}",
        graph_dir="./graphs",
        window_size=50
    )

    cumulative_rewards = np.cumsum(rewards_per_iteration)
    plot_mppi_cumulative_reward(
        cummulative_rewards=cumulative_rewards.tolist(),
        graph_name=f"mppi_cumulative_rewards_{clean_label}",
        graph_dir="./graphs"
    )

    print(f"graphs saved for {dataset_label}\n")
    return total_reward


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}", flush=True)

    datasets = [
        # ("D4RL/door/expert-v2", "Door Expert (Easy)"),
        # ("D4RL/door/cloned-v2", "Door Cloned (Medium)")
        # scrapping the other ones for now the other ones out because they are taking a long time to run
        ("D4RL/door/human-v2", "Door Human (Hard)")
    ]

    results = {}

    # run MPPI on each dataset
    for dataset_name, dataset_label in datasets:
        try:
            total_reward = run_dataset(dataset_name, dataset_label, device)
            results[dataset_label] = total_reward
        except Exception as e:
            print(f"\nerror running {dataset_label}: {e}")
            results[dataset_label] = None


    for dataset_label, reward in results.items():
        if reward is not None:
            print(f"{dataset_label}: Total Reward = {reward:.2f}")
        else:
            print("Failed")

    print("\ngraphs saved to ./graphs/")

    # Here is where we need to write some sort of video logger, I don't know how to do this, but it looks like the training performs correctly.



if __name__ == "__main__":
    main()