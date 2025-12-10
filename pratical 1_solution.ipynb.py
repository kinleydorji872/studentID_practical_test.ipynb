#%%
"""
Problem 1: Q-Learning on GridWorld - COMPLETE SOLUTION

This solution implements Q-learning to solve a 5x5 GridWorld environment.
"""

import numpy as np
import matplotlib.pyplot as plt

class GridWorld:
    """
    5x5 GridWorld Environment

    S . . . .
    . # . # .
    . . . # .
    . # . . .
    . . . . G

    S = Start (0,0)
    G = Goal (4,4)
    # = Wall (cannot pass)
    . = Empty cell
    """

    def __init__(self):
        self.grid_size = 5
        self.walls = [(1,1), (1,3), (2,3), (3,1)]
        self.start = (0, 0)
        self.goal = (4, 4)
        self.current_pos = self.start

        # Create mapping of valid positions to state indices
        self.valid_positions = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i, j) not in self.walls:
                    self.valid_positions.append((i, j))

        self.n_states = len(self.valid_positions)
        print(f"GridWorld initialized with {self.n_states} valid states")

    def reset(self):
        """Reset to start position. Return initial state."""
        self.current_pos = self.start
        return self.state_to_index(self.start)

    def step(self, action):
        """
        Take action and return (next_state, reward, done)
        Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        """
        row, col = self.current_pos

        # Calculate new position based on action
        if action == 0:  # UP
            new_pos = (row - 1, col)
        elif action == 1:  # DOWN
            new_pos = (row + 1, col)
        elif action == 2:  # LEFT
            new_pos = (row, col - 1)
        else:  # RIGHT (action == 3)
            new_pos = (row, col + 1)

        # Check if new position is valid
        new_row, new_col = new_pos

        # Hit wall or boundary - stay in place, negative reward
        if (new_pos in self.walls or
            new_row < 0 or new_row >= self.grid_size or
            new_col < 0 or new_col >= self.grid_size):
            reward = -1
            new_pos = self.current_pos
        # Move to goal - positive reward
        elif new_pos == self.goal:
            reward = 10
            self.current_pos = new_pos
            return self.state_to_index(new_pos), reward, True
        # Normal move - small negative reward (encourages efficiency)
        else:
            reward = -0.1
            self.current_pos = new_pos

        next_state = self.state_to_index(self.current_pos)
        done = (self.current_pos == self.goal)

        return next_state, reward, done

    def state_to_index(self, state):
        """Convert (row, col) to unique state index"""
        try:
            return self.valid_positions.index(state)
        except ValueError:
            # Invalid state (wall)
            return -1

    def index_to_state(self, index):
        """Convert state index back to (row, col)"""
        if 0 <= index < len(self.valid_positions):
            return self.valid_positions[index]
        return None


def train_qlearning(env, episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    """
    Train Q-learning agent

    Args:
        env: GridWorld environment
        episodes: Number of training episodes
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate

    Returns:
        Q: Q-table (numpy array)
        rewards: List of total rewards per episode
    """
    n_states = env.n_states
    n_actions = 4

    # Initialize Q-table with zeros
    Q = np.zeros((n_states, n_actions))

    # Track rewards
    rewards_per_episode = []

    print(f"Training Q-learning for {episodes} episodes...")
    print(f"Parameters: alpha={alpha}, gamma={gamma}, epsilon={epsilon}")
    print("-" * 60)

    for episode in range(episodes):
        # Reset environment
        state = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 100  # Prevent infinite loops

        while steps < max_steps:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                # Explore: random action
                action = np.random.randint(n_actions)
            else:
                # Exploit: best action from Q-table
                action = np.argmax(Q[state])

            # Take action
            next_state, reward, done = env.step(action)

            # Q-learning update rule:
            # Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state, best_next_action]
            td_error = td_target - Q[state, action]
            Q[state, action] = Q[state, action] + alpha * td_error

            total_reward += reward
            state = next_state
            steps += 1

            if done:
                break

        rewards_per_episode.append(total_reward)

        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}: Avg Reward (100) = {avg_reward:.2f}")

    print("-" * 60)
    print(f"Training complete!")
    print(f"Final average reward (last 100): {np.mean(rewards_per_episode[-100:]):.2f}")

    return Q, rewards_per_episode


def visualize_policy(env, Q):
    """Visualize learned policy as arrows"""
    action_symbols = ['↑', '↓', '←', '→']

    print("\nLearned Policy:")
    print("-" * 25)

    for i in range(env.grid_size):
        row_display = []
        for j in range(env.grid_size):
            if (i, j) == env.start:
                row_display.append('S')
            elif (i, j) == env.goal:
                row_display.append('G')
            elif (i, j) in env.walls:
                row_display.append('#')
            else:
                state_idx = env.state_to_index((i, j))
                if state_idx >= 0:
                    best_action = np.argmax(Q[state_idx])
                    row_display.append(action_symbols[best_action])
                else:
                    row_display.append('?')

        print('  '.join(row_display))

    print("-" * 25)


def plot_results(rewards):
    """Plot training rewards"""
    plt.figure(figsize=(12, 5))

    # Plot 1: Raw rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.6)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards per Episode')
    plt.grid(True, alpha=0.3)

    # Plot 2: Moving average
    plt.subplot(1, 2, 2)
    window = 50
    if len(rewards) >= window:
        moving_avg = [np.mean(rewards[max(0, i-window+1):i+1])
                     for i in range(len(rewards))]
        plt.plot(moving_avg, linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title(f'Moving Average ({window} episodes)')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('problem1_results.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'problem1_results.png'")
    plt.close()


def test_agent(env, Q, num_tests=10):
    """Test the trained agent"""
    print(f"\nTesting agent for {num_tests} episodes...")

    successes = 0
    total_steps = []

    for test in range(num_tests):
        state = env.reset()
        steps = 0
        max_steps = 100

        while steps < max_steps:
            # Use greedy policy (no exploration)
            action = np.argmax(Q[state])
            state, reward, done = env.step(action)
            steps += 1

            if done:
                successes += 1
                total_steps.append(steps)
                break

    print(f"Success rate: {successes}/{num_tests} ({100*successes/num_tests:.1f}%)")
    if total_steps:
        print(f"Average steps to goal: {np.mean(total_steps):.1f}")


# Main execution
if __name__ == "__main__":
    print("="*60)
    print("PROBLEM 1: Q-LEARNING ON GRIDWORLD")
    print("="*60)

    # Create environment
    env = GridWorld()

    # Train agent
    Q, rewards = train_qlearning(
        env,
        episodes=1000,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1
    )

    # Visualize policy
    visualize_policy(env, Q)

    # Plot results
    plot_results(rewards)

    # Test agent
    test_agent(env, Q, num_tests=20)

    # Print Q-table statistics
    print(f"\nQ-table Statistics:")
    print(f"  Shape: {Q.shape}")
    print(f"  Non-zero values: {np.count_nonzero(Q)}/{Q.size}")
    print(f"  Max Q-value: {np.max(Q):.2f}")
    print(f"  Min Q-value: {np.min(Q):.2f}")

    print("\n" + "="*60)
    print("PROBLEM 1 COMPLETE!")
    print("="*60)
# %%
