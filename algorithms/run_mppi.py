import torch
import numpy as np
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


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}", flush=True)

    print("Setting up MPPI controller...", flush=True)
    mppi_controller, dataset, dynamics_model = setup_mppi_for_minari(
        dataset_name="D4RL/door/human-v2",
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

    # The actual retrain_dynmaics loop for the buffer data.
    def retrain_dynamics(data):
        print("\n [Retraining dynamics model...]", flush=True)
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
        print("\n[Dynamics model retrained]", flush=True)

    print("\nStarting MPPI training...", flush=True)
    total_reward, _ = train_mppi(
        mppi=mppi_controller,
        env=env_wrapper,
        retrain_dynamics=retrain_dynamics,
        retrain_after_iter=100, # how many iterations after you retrain the dynamics model
        iter=1000, # how many total iterations
        render=False
    )

    print("\nTraining completed!")
    print(f"Total reward: {total_reward:.2f}")

    # Here is where we need to write some sort of video logger, I don't know how to do this, but it looks like the training performs correctly.


if __name__ == "__main__":
    main()