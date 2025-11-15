"""
Custom dyanmics model for hard tasks such as human-v2. This model retrains at every n steps in the mppi, so that the 
dynamics stay accurate and up to date.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import minari
from typing import Tuple, Optional


class DynamicsModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.network = nn.Sequential(
            # Need both current state and action to know what happens next, this is important data to keep in our network.
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        delta = self.network(x)
        next_state = state + delta
        return next_state

"""
Trains the original dynamics model.
Very similar to a typical supervised training loop 
using MSE loss (think of action + state as our input, next_state as our output) 
from minari data.
"""
def train_dynamics_model(model: nn.Module, dataset: minari.MinariDataset,
                        num_epochs: int = 100, batch_size: int = 256,
                        lr: float = 0.0003, device: str = None) -> nn.Module:
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Training on device: {device}")
    
    states = []
    actions = []
    next_states = []

    for episode in dataset:
        obs = episode.observations
        acts = episode.actions

        for i in range(len(obs) - 1):
            state = obs[i]
            action = acts[i]
            next_state = obs[i + 1]

            # Since we are using different environments, a state could be a dictionary of vectors or just a vecotr.
            if isinstance(state, dict):
                state = np.concatenate([v.flatten() for v in state.values()])
            if isinstance(next_state, dict):
                next_state = np.concatenate([v.flatten() for v in next_state.values()])
            if isinstance(action, dict):
                action = np.concatenate([v.flatten() for v in action.values()])

            states.append(state)
            actions.append(action)
            next_states.append(next_state)

    states = torch.FloatTensor(np.array(states)).to(device)
    actions = torch.FloatTensor(np.array(actions)).to(device)
    next_states = torch.FloatTensor(np.array(next_states)).to(device)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    dataset_size = len(states)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        indices = torch.randperm(dataset_size)

        for i in range(0, dataset_size, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_next_states = next_states[batch_indices]

            optimizer.zero_grad()
            pred_next_states = model(batch_states, batch_actions)
            loss = loss_fn(pred_next_states, batch_next_states)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.6f}")

    return model

"""
Transferring the network above into a function that can be passed through.
"""
def create_dynamics_function(model: nn.Module, device: str = 'cpu'):
    def dynamics(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            next_state = model(state, action)
            return next_state
    return dynamics


"""
Retrain from data buffer, exactly the same as the dynamics from above, except data is taken from the
buffer rather than the Minari dataset.
"""
def retrain_dynamics_from_buffer(model: nn.Module, buffer_tensor: torch.Tensor, nx: int, nu: int,
                                  num_epochs: int = 20, batch_size: int = 64,
                                  lr: float = 3e-4, device: str = 'cpu'):
    buffer_size = buffer_tensor.shape[0]

    states = buffer_tensor[:, :nx]
    actions = buffer_tensor[:, nx:nx+nu]
    next_states = buffer_tensor[:, nx+nu:]

    states = states.to(device)
    actions = actions.to(device)
    next_states = next_states.to(device)

    model = model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        indices = torch.randperm(buffer_size, device=device)

        for i in range(0, buffer_size, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_next_states = next_states[batch_indices]

            optimizer.zero_grad()
            pred_next_states = model(batch_states, batch_actions)
            loss = loss_fn(pred_next_states, batch_next_states)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        if (epoch + 1) % 10 == 0:
            print(f"  Retrain Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}", flush=True)

    model.eval()
    return model

"""
Creates the running_cost function explained in the MPPI base implementation.
Keeps track of the cost of the action, where MPPI tries to minimize the cost in the trajectory
"""
def create_running_cost_function(goal_state: Optional[np.ndarray] = None,
                                action_penalty: float = 0.01):
    def running_cost(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if goal_state is not None:
            goal = torch.FloatTensor(goal_state).to(state.device)
            state_cost = torch.sum((state - goal) ** 2, dim=-1) # state cost is distance squared from goal
        else:
            state_cost = torch.zeros(state.shape[0], device=state.device)

        action_cost = action_penalty * torch.sum(action ** 2, dim=-1) # can change coefficinet, penelizes large motions. Will be easier to change with seeing videos

        total_cost = state_cost + action_cost

        return total_cost

    return running_cost

"""
Set up for the human-v2 dataset in Minari. Could be generalized to other minari datasets.
If generalizing, may need some if statements for tensors, or vectors, as something can be 1 dimensional.
"""
def setup_mppi_for_minari(dataset_name: str = "D4RL/door/human-v2",
                          num_samples: int = 1000,
                          horizon: int = 20,
                          lambda_: float = 1.0,
                          noise_sigma_scale: float = 0.5,
                          device: str = 'cpu',
                          train_dynamics: bool = True,
                          num_epochs: int = 100):

    dataset = minari.load_dataset(dataset_name, download=True)

    sample_episode = dataset[0]
    sample_obs = sample_episode.observations[0]
    sample_action = sample_episode.actions[0]

    if isinstance(sample_obs, dict):
        state_dim = sum([v.flatten().shape[0] for v in sample_obs.values()])
    else:
        state_dim = sample_obs.flatten().shape[0]

    if isinstance(sample_action, dict):
        action_dim = sum([v.flatten().shape[0] for v in sample_action.values()])
    else:
        action_dim = sample_action.flatten().shape[0]

    print(f"State dimension: {state_dim}", flush=True)
    print(f"Action dimension: {action_dim}", flush=True)

    dynamics_model = DynamicsModel(state_dim, action_dim, hidden_dim=256)

    if train_dynamics:
        print("Training dynamics model...", flush=True)
        dynamics_model = train_dynamics_model(
            dynamics_model,
            dataset,
            num_epochs=num_epochs,
            device=device
        )

    dynamics_fn = create_dynamics_function(dynamics_model, device=device)
    running_cost_fn = create_running_cost_function(action_penalty=0.01)

    noise_sigma = torch.eye(action_dim, dtype=torch.float32) * noise_sigma_scale

    from mppi import MPPI

    mppi_controller = MPPI(
        dynamics=dynamics_fn,
        running_cost=running_cost_fn,
        nx=state_dim,
        noise_sigma=noise_sigma,
        num_samples=num_samples,
        horizon=horizon,
        device=device,
        lambda_=lambda_,
        u_min=torch.FloatTensor([-1.0] * action_dim),
        u_max=torch.FloatTensor([1.0] * action_dim),
    )
    return mppi_controller, dataset, dynamics_model
