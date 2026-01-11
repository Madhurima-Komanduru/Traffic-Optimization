import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import random
import pickle
import os

# ============================================================================
# TRAFFIC SIMULATION ENVIRONMENT
# ============================================================================

class TrafficEnvironment:
    """
    Custom 4-way intersection traffic simulation environment.
    Each direction (North, South, East, West) has incoming lanes.
    """
    
    def __init__(self, max_steps=1000):
        # 4 directions: North, South, East, West
        self.num_directions = 4
        self.max_steps = max_steps
        self.current_step = 0
        
        # Traffic light phases: 0=NS_green, 1=EW_green
        self.num_phases = 2
        self.current_phase = 0
        self.phase_duration = 0  # Steps since last phase change
        self.min_phase_duration = 5  # Minimum green light duration
        
        # Vehicle counts per direction
        self.vehicle_counts = np.zeros(self.num_directions)
        self.waiting_times = np.zeros(self.num_directions)
        
        # Ambulance status
        self.ambulance_present = False
        self.ambulance_direction = None
        
        # Traffic generation parameters
        self.base_arrival_rate = 0.3  # Base probability of vehicle arrival
        self.traffic_density = 1.0  # Multiplier for traffic load
        
        # Statistics
        self.total_waiting_time = 0
        self.total_vehicles_processed = 0
        self.episode_rewards = []
        
    def reset(self, traffic_density=1.0):
        """Reset environment for new episode"""
        self.current_step = 0
        self.current_phase = 0
        self.phase_duration = 0
        self.vehicle_counts = np.random.randint(0, 10, size=self.num_directions)
        self.waiting_times = np.zeros(self.num_directions)
        self.ambulance_present = False
        self.ambulance_direction = None
        self.traffic_density = traffic_density
        self.total_waiting_time = 0
        self.total_vehicles_processed = 0
        
        return self._get_state()
    
    def _get_state(self):
        """
        State representation:
        - Vehicle count in each direction (4 values)
        - Waiting time in each direction (4 values)
        - Current phase (1 value)
        - Phase duration (1 value)
        - Ambulance present (1 value)
        - Ambulance direction (1 value, -1 if no ambulance)
        Total: 12 features
        """
        state = np.concatenate([
            self.vehicle_counts / 50.0,  # Normalize
            self.waiting_times / 100.0,  # Normalize
            [self.current_phase],
            [self.phase_duration / 20.0],
            [1.0 if self.ambulance_present else 0.0],
            [self.ambulance_direction / 3.0 if self.ambulance_present else -1.0]
        ])
        return state.astype(np.float32)
    
    def _generate_vehicles(self):
        """Simulate vehicle arrivals"""
        for direction in range(self.num_directions):
            if np.random.random() < self.base_arrival_rate * self.traffic_density:
                self.vehicle_counts[direction] += 1
        
        # Random ambulance arrival (2% chance per step)
        if not self.ambulance_present and np.random.random() < 0.02:
            self.ambulance_present = True
            self.ambulance_direction = np.random.randint(0, self.num_directions)
            self.vehicle_counts[self.ambulance_direction] += 1
    
    def _process_traffic(self):
        """Process vehicles based on current light phase"""
        # Determine which directions have green light
        if self.current_phase == 0:  # North-South green
            green_directions = [0, 1]  # North, South
        else:  # East-West green
            green_directions = [2, 3]  # East, West
        
        vehicles_processed = 0
        
        for direction in range(self.num_directions):
            if direction in green_directions:
                # Process vehicles (3-5 vehicles per step with green light)
                processed = min(self.vehicle_counts[direction], np.random.randint(3, 6))
                self.vehicle_counts[direction] -= processed
                vehicles_processed += processed
                
                # Reset waiting time for this direction
                if processed > 0:
                    self.total_waiting_time += self.waiting_times[direction] * processed
                    self.waiting_times[direction] = 0
                
                # Check if ambulance was processed
                if self.ambulance_present and direction == self.ambulance_direction:
                    if processed > 0:
                        self.ambulance_present = False
                        self.ambulance_direction = None
            else:
                # Increment waiting time for red light directions
                if self.vehicle_counts[direction] > 0:
                    self.waiting_times[direction] += 1
        
        self.total_vehicles_processed += vehicles_processed
        return vehicles_processed
    
    def step(self, action):
        """
        Execute action and return next state, reward, done
        Actions: 0 = keep current phase, 1 = switch phase
        """
        self.current_step += 1
        self.phase_duration += 1
        
        # Generate new vehicles
        self._generate_vehicles()
        
        # Handle action (phase switching)
        reward = 0
        phase_switched = False
        
        if action == 1 and self.phase_duration >= self.min_phase_duration:
            # Switch phase
            self.current_phase = 1 - self.current_phase
            self.phase_duration = 0
            phase_switched = True
            reward -= 1  # Small penalty for switching (represents transition time)
        
        # Emergency: Ambulance priority
        if self.ambulance_present:
            required_phase = 0 if self.ambulance_direction in [0, 1] else 1
            if self.current_phase != required_phase:
                # Large penalty for not giving ambulance priority
                reward -= 50
            else:
                # Reward for giving ambulance priority
                reward += 20
        
        # Process traffic
        vehicles_processed = self._process_traffic()
        
        # Calculate reward based on waiting times and vehicle processing
        total_waiting = np.sum(self.waiting_times * self.vehicle_counts)
        avg_waiting = total_waiting / max(np.sum(self.vehicle_counts), 1)
        
        # Reward: negative of average waiting time + vehicles processed
        reward += vehicles_processed * 2  # Reward for processing vehicles
        reward -= avg_waiting * 0.5  # Penalty for long waiting times
        reward -= np.sum(self.vehicle_counts) * 0.1  # Penalty for queue length
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        next_state = self._get_state()
        
        return next_state, reward, done, {
            'vehicles_processed': vehicles_processed,
            'avg_waiting_time': avg_waiting,
            'total_vehicles': np.sum(self.vehicle_counts)
        }
    
    def get_average_waiting_time(self):
        """Calculate average waiting time per vehicle"""
        if self.total_vehicles_processed > 0:
            return self.total_waiting_time / self.total_vehicles_processed
        return 0


# ============================================================================
# DEEP Q-NETWORK (DQN) AGENT
# ============================================================================

class DQN(nn.Module):
    """Deep Q-Network for traffic control"""
    
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class DQNAgent:
    """DQN Agent with Experience Replay and Target Network"""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.memory = deque(maxlen=10000)
        
        # Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Initialize target network
        self.update_target_network()
        
    def update_target_network(self):
        """Copy weights from model to target_model"""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()
    
    def replay(self):
        """Train on batch from replay memory"""
        if len(self.memory) < self.batch_size:
            return 0
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filename):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)
        print(f"Model saved to {filename}")
    
    def load(self, filename):
        """Load model"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.update_target_network()
        print(f"Model loaded from {filename}")


# ============================================================================
# STATIC TIMER BASELINE
# ============================================================================

class StaticTimerController:
    """Baseline: Fixed-timer traffic signal controller"""
    
    def __init__(self, phase_duration=20):
        self.phase_duration = phase_duration
        self.current_phase = 0
        self.steps_in_phase = 0
    
    def reset(self):
        self.current_phase = 0
        self.steps_in_phase = 0
    
    def act(self, state):
        """Switch phase every fixed interval"""
        self.steps_in_phase += 1
        if self.steps_in_phase >= self.phase_duration:
            self.steps_in_phase = 0
            self.current_phase = 1 - self.current_phase
            return 1  # Switch action
        return 0  # Keep current phase


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_dqn_agent(episodes=500, max_steps=1000, save_path='traffic_dqn_model.pth'):
    """Train DQN agent on traffic environment"""
    
    env = TrafficEnvironment(max_steps=max_steps)
    state_size = 12
    action_size = 2
    agent = DQNAgent(state_size, action_size)
    
    episode_rewards = []
    episode_avg_waiting_times = []
    losses = []
    
    print("Starting DQN Training...")
    print(f"Device: {agent.device}")
    print(f"Episodes: {episodes}, Max Steps: {max_steps}\n")
    
    for episode in range(episodes):
        # Vary traffic density across episodes
        traffic_density = 0.5 + (episode / episodes) * 1.5  # 0.5 to 2.0
        state = env.reset(traffic_density=traffic_density)
        total_reward = 0
        episode_loss = []
        
        for step in range(max_steps):
            # Choose action
            action = agent.act(state, training=True)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            loss = agent.replay()
            if loss > 0:
                episode_loss.append(loss)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Update target network every 10 episodes
        if episode % 10 == 0:
            agent.update_target_network()
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Record metrics
        episode_rewards.append(total_reward)
        avg_waiting = env.get_average_waiting_time()
        episode_avg_waiting_times.append(avg_waiting)
        
        if episode_loss:
            losses.append(np.mean(episode_loss))
        
        # Print progress
        if (episode + 1) % 50 == 0:
            recent_rewards = np.mean(episode_rewards[-50:])
            recent_waiting = np.mean(episode_avg_waiting_times[-50:])
            print(f"Episode {episode + 1}/{episodes}")
            print(f"  Avg Reward (last 50): {recent_rewards:.2f}")
            print(f"  Avg Waiting Time (last 50): {recent_waiting:.2f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            if losses:
                print(f"  Loss: {losses[-1]:.4f}")
            print()
    
    # Save trained model
    agent.save(save_path)
    
    # Save training history
    history = {
        'episode_rewards': episode_rewards,
        'episode_avg_waiting_times': episode_avg_waiting_times,
        'losses': losses
    }
    
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    
    print(f"\nTraining completed! Model saved to {save_path}")
    
    return agent, history


# ============================================================================
# EVALUATION AND COMPARISON
# ============================================================================

def evaluate_controller(controller, controller_type='DQN', num_episodes=50, 
                       max_steps=1000, traffic_densities=[0.5, 1.0, 1.5, 2.0]):
    """Evaluate controller performance across different traffic densities"""
    
    env = TrafficEnvironment(max_steps=max_steps)
    results = {density: [] for density in traffic_densities}
    
    print(f"\nEvaluating {controller_type} Controller...")
    
    for density in traffic_densities:
        density_waiting_times = []
        
        for episode in range(num_episodes):
            state = env.reset(traffic_density=density)
            
            if controller_type == 'Static':
                controller.reset()
            
            for step in range(max_steps):
                if controller_type == 'DQN':
                    action = controller.act(state, training=False)
                else:
                    action = controller.act(state)
                
                next_state, reward, done, info = env.step(action)
                state = next_state
                
                if done:
                    break
            
            avg_waiting = env.get_average_waiting_time()
            density_waiting_times.append(avg_waiting)
        
        results[density] = density_waiting_times
        avg = np.mean(density_waiting_times)
        std = np.std(density_waiting_times)
        print(f"  Density {density}: Avg Wait Time = {avg:.2f} ± {std:.2f}")
    
    return results


def compare_controllers(dqn_agent, save_plots=True):
    """Compare DQN agent vs Static Timer controller"""
    
    static_controller = StaticTimerController(phase_duration=20)
    
    # Evaluate both controllers
    dqn_results = evaluate_controller(dqn_agent, 'DQN')
    static_results = evaluate_controller(static_controller, 'Static')
    
    # Plot comparison
    densities = list(dqn_results.keys())
    dqn_means = [np.mean(dqn_results[d]) for d in densities]
    static_means = [np.mean(static_results[d]) for d in densities]
    dqn_stds = [np.std(dqn_results[d]) for d in densities]
    static_stds = [np.std(static_results[d]) for d in densities]
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Average Wait Time vs Traffic Density
    plt.subplot(1, 2, 1)
    plt.errorbar(densities, dqn_means, yerr=dqn_stds, marker='o', 
                 label='DQN Agent', capsize=5, linewidth=2)
    plt.errorbar(densities, static_means, yerr=static_stds, marker='s', 
                 label='Static Timer', capsize=5, linewidth=2)
    plt.xlabel('Traffic Density', fontsize=12)
    plt.ylabel('Average Waiting Time (steps)', fontsize=12)
    plt.title('Performance Comparison: DQN vs Static Timer', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Improvement Percentage
    plt.subplot(1, 2, 2)
    improvement = [(static_means[i] - dqn_means[i]) / static_means[i] * 100 
                   for i in range(len(densities))]
    colors = ['green' if imp > 0 else 'red' for imp in improvement]
    plt.bar(densities, improvement, color=colors, alpha=0.7)
    plt.xlabel('Traffic Density', fontsize=12)
    plt.ylabel('Improvement (%)', fontsize=12)
    plt.title('DQN Improvement over Static Timer', fontsize=14, fontweight='bold')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
        print("\nPerformance comparison plot saved as 'performance_comparison.png'")
    
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    for i, density in enumerate(densities):
        print(f"\nTraffic Density: {density}")
        print(f"  DQN Agent:      {dqn_means[i]:.2f} ± {dqn_stds[i]:.2f} steps")
        print(f"  Static Timer:   {static_means[i]:.2f} ± {static_stds[i]:.2f} steps")
        print(f"  Improvement:    {improvement[i]:.1f}%")
    
    print("\n" + "="*60)
    
    return dqn_results, static_results


def plot_training_history(history, save_plots=True):
    """Plot training metrics"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Episode Rewards
    axes[0, 0].plot(history['episode_rewards'], alpha=0.6, linewidth=0.8)
    axes[0, 0].plot(np.convolve(history['episode_rewards'], 
                                np.ones(50)/50, mode='valid'), 
                   linewidth=2, label='50-episode MA')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Episode Rewards During Training', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Average Waiting Time
    axes[0, 1].plot(history['episode_avg_waiting_times'], alpha=0.6, linewidth=0.8)
    axes[0, 1].plot(np.convolve(history['episode_avg_waiting_times'], 
                                np.ones(50)/50, mode='valid'), 
                   linewidth=2, label='50-episode MA', color='orange')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Avg Waiting Time (steps)')
    axes[0, 1].set_title('Average Waiting Time During Training', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training Loss
    if history['losses']:
        axes[1, 0].plot(history['losses'], linewidth=1.5, color='red')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Reward Distribution
    axes[1, 1].hist(history['episode_rewards'], bins=30, alpha=0.7, 
                    color='green', edgecolor='black')
    axes[1, 1].axvline(np.mean(history['episode_rewards']), 
                      color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {np.mean(history["episode_rewards"]):.1f}')
    axes[1, 1].set_xlabel('Total Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Episode Rewards', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print("Training history plot saved as 'training_history.png'")
    
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("AUTONOMOUS TRAFFIC OPTIMIZATION USING REINFORCEMENT LEARNING")
    print("="*70)
    print("\nThis system uses Deep Q-Network (DQN) to optimize traffic flow")
    print("at a 4-way intersection with adaptive ambulance priority.\n")
    
    # Train the DQN agent
    print("[1/3] Training DQN Agent...")
    print("-" * 70)
    agent, training_history = train_dqn_agent(
        episodes=500,
        max_steps=1000,
        save_path='traffic_dqn_model.pth'
    )
    
    # Plot training history
    print("\n[2/3] Visualizing Training Progress...")
    print("-" * 70)
    plot_training_history(training_history)
    
    # Compare with baseline
    print("\n[3/3] Comparing with Static Timer Baseline...")
    print("-" * 70)
    dqn_results, static_results = compare_controllers(agent)
    
    print("\n" + "="*70)
    print("TRAINING AND EVALUATION COMPLETED!")
    print("="*70)
    print("\nGenerated Files:")
    print("  • traffic_dqn_model.pth - Trained DQN model")
    print("  • training_history.pkl - Training metrics")
    print("  • training_history.png - Training visualization")
    print("  • performance_comparison.png - Performance comparison plots")
    print("\n" + "="*70)
