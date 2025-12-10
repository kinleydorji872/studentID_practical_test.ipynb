
"""
Problem 2: Deep Q-Network Completion - COMPLETE SOLUTION
Updated for Gymnasium compatibility
"""

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym  # CHANGED: Use gymnasium instead of gym
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt

class DQN(nn.Module):
    """Deep Q-Network architecture"""

    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()

        # Neural network architecture
        # Input: state_dim -> Hidden: 128 -> Hidden: 128 -> Output: action_dim
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        """Forward pass through the network"""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    """Experience replay buffer"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample random mini-batch"""
        # Randomly sample batch_size transitions
        batch = random.sample(self.buffer, batch_size)

        # Unpack and convert to numpy arrays
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent"""

    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        # Initialize Q-network and target network
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)

        self.batch_size = 64

    def select_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.action_dim)
        else:
            # Exploit: best action from Q-network
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def train_step(self, batch_size=64):
        """Training step using experience replay"""
        if len(self.replay_buffer) < batch_size:
            return 0

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Convert to PyTorch tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Compute current Q-values
        # Q(s,a) for the actions that were taken
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Compute target Q-values using target network
        # Target: r + γ * max_a' Q̂(s', a')
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss (MSE between current and target Q-values)
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Backpropagate and update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def update_target_network(self):
        """Copy weights from Q-network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())


def train_cartpole(episodes=500):
    """Train DQN on CartPole-v1"""

    print("="*60)
    print("PROBLEM 2: DQN ON CARTPOLE-V1")
    print("="*60)

    # Create environment with Gymnasium
    env = gym.make('CartPole-v1')  # Gymnasium compatible
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"Environment: CartPole-v1")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print("-" * 60)

    # Create agent
    agent = DQNAgent(state_dim, action_dim)

    # Training metrics
    rewards = []
    losses = []

    for episode in range(episodes):
        # Reset environment - Gymnasium returns (state, info)
        state, _ = env.reset()  # SIMPLIFIED: No need for type checking
        
        total_reward = 0
        episode_losses = []
        steps = 0
        done = False

        while not done:
            # Select action
            action = agent.select_action(state, training=True)

            # Take step in environment - Gymnasium returns 5 values
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition in replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # Train agent
            loss = agent.train_step()
            if loss > 0:
                episode_losses.append(loss)

            # Update state and reward
            state = next_state
            total_reward += reward
            steps += 1

        # Update target network every 10 episodes
        if (episode + 1) % 10 == 0:
            agent.update_target_network()

        # Track rewards
        rewards.append(total_reward)
        if episode_losses:
            losses.append(np.mean(episode_losses))

        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
            print(f"Episode {episode + 1}/{episodes}")
            print(f"  Reward: {total_reward:.1f} | Avg (100): {avg_reward:.1f}")
            print(f"  Steps: {steps} | Epsilon: {agent.epsilon:.3f}")
            if losses:
                print(f"  Loss: {losses[-1]:.4f}")
            print("-" * 60)

        # Check if solved
        if len(rewards) >= 100 and np.mean(rewards[-100:]) >= 475:
            print(f"\n🎉 CartPole solved in {episode + 1} episodes!")
            print(f"Average reward (100 episodes): {np.mean(rewards[-100:]):.2f}")
            break

    env.close()

    # Plot results
    plot_results(rewards, losses)

    return agent, rewards


def plot_results(rewards, losses):
    """Plot training results"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot rewards
    axes[0].plot(rewards, alpha=0.6, label='Episode Reward')
    if len(rewards) >= 100:
        moving_avg = [np.mean(rewards[max(0, i-99):i+1]) for i in range(len(rewards))]
        axes[0].plot(moving_avg, linewidth=2, label='Moving Avg (100)')
    axes[0].axhline(y=475, color='r', linestyle='--', label='Solved Threshold')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Training Rewards')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot losses
    if losses:
        axes[1].plot(losses, alpha=0.6, label='Loss')
        if len(losses) >= 50:
            moving_avg = [np.mean(losses[max(0, i-49):i+1]) for i in range(len(losses))]
            axes[1].plot(moving_avg, linewidth=2, label='Moving Avg (50)')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Training Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('problem2_results.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'problem2_results.png'")
    plt.close()


def test_agent(agent, episodes=10):
    """Test trained agent"""
    env = gym.make('CartPole-v1')  # Gymnasium compatible

    print(f"\nTesting agent for {episodes} episodes...")
    test_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()  # Gymnasium returns (state, info)

        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward

        test_rewards.append(total_reward)
        print(f"Test Episode {episode + 1}: Reward = {total_reward:.1f}")

    env.close()

    print(f"\nTest Results:")
    print(f"  Average: {np.mean(test_rewards):.2f}")
    print(f"  Std Dev: {np.std(test_rewards):.2f}")
    print(f"  Min/Max: {np.min(test_rewards):.1f} / {np.max(test_rewards):.1f}")


# Main execution
if __name__ == "__main__":
    # Train agent
    agent, rewards = train_cartpole(episodes=500)

    # Test agent
    test_agent(agent, episodes=10)

    # Final statistics
    print("\n" + "="*60)
    print("TRAINING STATISTICS")
    print("="*60)
    print(f"Total episodes: {len(rewards)}")
    print(f"Final avg reward (100): {np.mean(rewards[-100:]):.2f}")
    print(f"Best episode reward: {np.max(rewards):.1f}")
    print(f"Solved: {'Yes' if np.mean(rewards[-100:]) >= 475 else 'No'}")
    print("="*60)
    print("PROBLEM 2 COMPLETE!")
    print("="*60)