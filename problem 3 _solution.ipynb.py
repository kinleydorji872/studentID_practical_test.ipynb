"""
Problem 3: Analysis and Debugging - COMPLETE SOLUTION

This solution identifies and fixes bugs in the FrozenLake Q-learning implementation
"""

import gymnasium as gym  # CHANGED: Fix for np.bool8 error
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# BUG ANALYSIS
# ============================================================================

"""
BUG 1: Q-table initialization
Location: Line 15
Description: Q-table is initialized but might have dimension issues
Impact: Incorrect Q-table shape could cause indexing errors
Fix: Ensure proper dimensions match state and action spaces

BUG 2: Action selection (Line 33)
Location: Line 33 (np.argmin instead of np.argmax)
Description: Using argmin (minimum Q-value) instead of argmax (maximum Q-value)
Impact: Agent chooses WORST actions instead of BEST actions - completely wrong!
Fix: Change np.argmin(Q[state]) to np.argmax(Q[state])

BUG 3: Q-learning update rule (Line 40)
Location: Line 40
Description: Missing learning rate (alpha) in the update rule
             Current: Q[state, action] = reward + gamma * np.max(Q[next_state])
             Correct: Q[state, action] += alpha * (target - current_q)
Impact: Overwrites Q-values instead of gradually updating them
Fix: Use proper temporal difference update with learning rate

BUG 4: Epsilon decay (Line 51)
Location: No epsilon decay implemented
Description: Epsilon stays constant at 0.3, never decreases
Impact: Agent never shifts from exploration to exploitation
Fix: Add epsilon decay: epsilon = max(epsilon_min, epsilon * decay_rate)
"""

# ============================================================================
# BUGGY CODE (For Reference)
# ============================================================================

def buggy_train_frozenlake(episodes=10000):
    """
    BUGGY VERSION - For demonstration purposes
    """
    env = gym.make('FrozenLake-v1', is_slippery=True)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Bug 1: Q-table initialization (this is actually OK, but could note dimensions)
    Q = np.zeros((n_states, n_actions))

    alpha = 0.8  # Learning rate
    gamma = 0.95  # Discount factor
    epsilon = 0.3  # Exploration rate

    rewards = []

    for episode in range(episodes):
        result = env.reset()
        state = result[0] if isinstance(result, tuple) else result
        total_reward = 0
        done = False

        while not done:
            # Bug 2: Action selection issue - using argmin instead of argmax!
            if np.random.random() < epsilon:
                action = np.random.randint(n_actions)
            else:
                action = np.argmin(Q[state])  # BUG: Should be argmax!

            result = env.step(action)
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result

            # Bug 3: Update rule issue - missing learning rate and current Q-value
            Q[state, action] = reward + gamma * np.max(Q[next_state])  # BUG: Wrong update!

            state = next_state
            total_reward += reward

        rewards.append(total_reward)

        # Bug 4: Epsilon handling - no epsilon decay!

    success_rate = np.mean(rewards[-100:])
    print(f"Buggy version success rate: {success_rate}")

    return Q, rewards


# ============================================================================
# FIXED CODE
# ============================================================================

def fixed_train_frozenlake(episodes=10000):
    """
    FIXED VERSION - All bugs corrected
    """
    env = gym.make('FrozenLake-v1', is_slippery=True)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    print(f"FrozenLake Environment:")
    print(f"  States: {n_states}")
    print(f"  Actions: {n_actions}")
    print("-" * 60)

    # FIX 1: Q-table initialization (this was OK, but adding comment)
    Q = np.zeros((n_states, n_actions))  # Properly shaped Q-table

    # Hyperparameters
    alpha = 0.1  # Learning rate (lowered from 0.8 for more stable learning)
    gamma = 0.99  # Discount factor (increased for better long-term planning)
    epsilon = 1.0  # Initial exploration rate (start high)
    epsilon_min = 0.01  # Minimum exploration rate
    epsilon_decay = 0.9995  # Epsilon decay rate

    rewards = []

    print(f"Training with:")
    print(f"  Learning rate (alpha): {alpha}")
    print(f"  Discount factor (gamma): {gamma}")
    print(f"  Initial epsilon: {epsilon}")
    print(f"  Epsilon decay: {epsilon_decay}")
    print("-" * 60)

    for episode in range(episodes):
        result = env.reset()
        state = result[0] if isinstance(result, tuple) else result
        total_reward = 0
        done = False
        steps = 0
        max_steps = 100  # Prevent infinite loops

        while not done and steps < max_steps:
            # FIX 2: Action selection - use argmax for exploitation!
            if np.random.random() < epsilon:
                action = np.random.randint(n_actions)  # Explore
            else:
                action = np.argmax(Q[state])  # Exploit (FIXED: was argmin)

            result = env.step(action)
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result

            # FIX 3: Proper Q-learning update rule with learning rate
            # Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
            current_q = Q[state, action]
            max_next_q = np.max(Q[next_state])
            target_q = reward + gamma * max_next_q
            Q[state, action] = current_q + alpha * (target_q - current_q)

            state = next_state
            total_reward += reward
            steps += 1

        rewards.append(total_reward)

        # FIX 4: Epsilon decay - gradually reduce exploration
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Print progress
        if (episode + 1) % 1000 == 0:
            success_rate = np.mean(rewards[-100:])
            print(f"Episode {episode + 1}: Success rate (100) = {success_rate:.3f}, Epsilon = {epsilon:.3f}")

    env.close()

    success_rate = np.mean(rewards[-100:])
    print(f"\nFinal success rate: {success_rate:.3f}")

    return Q, rewards


def compare_buggy_vs_fixed():
    """
    Compare buggy and fixed implementations
    """
    print("="*60)
    print("PROBLEM 3: DEBUGGING Q-LEARNING")
    print("="*60)
    print("\nBUG ANALYSIS:")
    print("-" * 60)
    print("Bug 1: Q-table initialization")
    print("  Issue: Actually OK, but should verify dimensions")
    print("  Impact: Low")
    print("  Fix: Ensure Q = np.zeros((n_states, n_actions))")
    print()
    print("Bug 2: Action selection (CRITICAL)")
    print("  Issue: Using np.argmin instead of np.argmax")
    print("  Impact: Agent chooses WORST actions, not BEST")
    print("  Fix: Change to np.argmax(Q[state])")
    print()
    print("Bug 3: Q-learning update rule (CRITICAL)")
    print("  Issue: Missing learning rate, overwrites Q-values")
    print("  Impact: No gradual learning, unstable updates")
    print("  Fix: Q[s,a] += alpha * (target - Q[s,a])")
    print()
    print("Bug 4: Epsilon decay (IMPORTANT)")
    print("  Issue: Epsilon never decreases")
    print("  Impact: Agent never exploits learned knowledge")
    print("  Fix: epsilon = max(epsilon_min, epsilon * decay)")
    print("-" * 60)

    episodes = 5000

    # Train buggy version
    print("\n" + "="*60)
    print("Training BUGGY version...")
    print("="*60)
    Q_buggy, rewards_buggy = buggy_train_frozenlake(episodes=episodes)

    # Train fixed version
    print("\n" + "="*60)
    print("Training FIXED version...")
    print("="*60)
    Q_fixed, rewards_fixed = fixed_train_frozenlake(episodes=episodes)

    # Compare results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"\nBuggy Implementation:")
    print(f"  Success rate: {np.mean(rewards_buggy[-100:]):.3f}")
    print(f"  Total successes: {sum(rewards_buggy)}")
    print(f"\nFixed Implementation:")
    print(f"  Success rate: {np.mean(rewards_fixed[-100:]):.3f}")
    print(f"  Total successes: {sum(rewards_fixed)}")
    print(f"\nImprovement: {(np.mean(rewards_fixed[-100:]) - np.mean(rewards_buggy[-100:])):.3f}")

    # Plot comparison
    plot_comparison(rewards_buggy, rewards_fixed)

    print("\n" + "="*60)
    print("PROBLEM 3 COMPLETE!")
    print("="*60)


def plot_comparison(rewards_buggy, rewards_fixed):
    """Plot comparison of buggy vs fixed implementation"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    window = 100

    # Moving averages
    ma_buggy = [np.mean(rewards_buggy[max(0, i-window+1):i+1])
                for i in range(len(rewards_buggy))]
    ma_fixed = [np.mean(rewards_fixed[max(0, i-window+1):i+1])
                for i in range(len(rewards_fixed))]

    # Plot 1: Success rates over time
    axes[0].plot(ma_buggy, label='Buggy Version', alpha=0.7, linewidth=2)
    axes[0].plot(ma_fixed, label='Fixed Version', alpha=0.7, linewidth=2)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Success Rate (100-episode window)')
    axes[0].set_title('Learning Progress Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Final performance comparison
    final_buggy = np.mean(rewards_buggy[-100:])
    final_fixed = np.mean(rewards_fixed[-100:])

    axes[1].bar(['Buggy', 'Fixed'], [final_buggy, final_fixed],
                color=['red', 'green'], alpha=0.7)
    axes[1].set_ylabel('Success Rate')
    axes[1].set_title('Final Performance (last 100 episodes)')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    axes[1].text(0, final_buggy + 0.02, f'{final_buggy:.3f}',
                ha='center', fontweight='bold')
    axes[1].text(1, final_fixed + 0.02, f'{final_fixed:.3f}',
                ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('problem3_comparison.png', dpi=300, bbox_inches='tight')
    print("\nComparison plot saved as 'problem3_comparison.png'")
    plt.close()


# Main execution
if __name__ == "__main__":
    compare_buggy_vs_fixed()