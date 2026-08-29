import numpy as np
from pg.hyperparameters import HyperParameters
from .memory_multiaction import MemoryMultiAction


class MultiActionAgent:
    def __init__(self, policy_network, hyperparams: HyperParameters):
        self.memory = MemoryMultiAction()
        self.hyperparams = hyperparams
        self.policy_network = policy_network
        self.previous_frame = None
        self.episode_count = 0
        self.total_rewards = []

        # Exploration parameters
        self.initial_temperature = 2.0  # Start with high exploration
        self.min_temperature = 0.5  # Minimum temperature
        self.temperature_decay = 0.9999  # Slow decay

        # Reward tracking
        self.reward_history = []
        self.baseline_reward = 0.0

        # Temperature tracking
        self.current_temperature = self.initial_temperature

    def sample_action(self, state: np.ndarray) -> int:
        action_probs, _ = self.policy_network.forward_pass(state)

        # Adaptive temperature based on performance
        if len(self.total_rewards) > 10:
            recent_avg = np.mean(self.total_rewards[-10:])
            if recent_avg < self.baseline_reward:
                self.current_temperature = max(
                    self.min_temperature,
                    self.initial_temperature
                    * (self.temperature_decay**self.episode_count),
                )
            else:
                self.current_temperature = self.min_temperature
        else:
            self.current_temperature = self.initial_temperature

        # Apply temperature for exploration
        logits = np.log(action_probs + 1e-8) / self.current_temperature
        exp_logits = np.exp(logits - np.max(logits))
        action_probs = exp_logits / np.sum(exp_logits)

        # Add some random exploration
        if np.random.random() < 0.1:  # 10% random actions
            action = np.random.randint(0, len(action_probs))
        else:
            action = np.random.choice(len(action_probs), p=action_probs)

        return action

    def sample_and_record_action(self, state: np.ndarray) -> int:
        action_probs, hidden_layer = self.policy_network.forward_pass(state)
        action = self.sample_action(state)

        # todo(shehio): Next bug might be here!!

        # Store the log-prob gradient for the taken action
        dlogp = np.zeros_like(action_probs)
        dlogp[action] = 1 - action_probs[action]  # encourage the taken action

        # Add entropy regularization to encourage exploration
        entropy = -np.sum(action_probs * np.log(action_probs + 1e-8))
        entropy_bonus = 0.005 * entropy  # Reduced entropy bonus

        self.memory.states.append(state)
        self.memory.hidden_layers.append(hidden_layer)
        self.memory.dlogps.append(dlogp)
        self.memory.actions.append(action)
        self.memory.entropies.append(entropy_bonus)

        return action

    def reap_reward(self, reward: float) -> None:
        self.memory.rewards.append(reward)

    def make_episode_end_updates(self, episode_number: int) -> None:
        episode_reward = np.sum(self.memory.rewards)
        self.total_rewards.append(episode_reward)

        # Update baseline reward
        if len(self.total_rewards) > 0:
            self.baseline_reward = np.mean(self.total_rewards)

        # Print temperature every 10 episodes
        if episode_number % 10 == 0:
            print(
                f"Temperature: {self.current_temperature:.3f}, Episode: {episode_number}"
            )

        # Only train if we have meaningful rewards
        if episode_reward > -100:  # Don't train on very bad episodes
            self.__accumulate_gradient()
            self.__train_policy_network(episode_number)

        self.__save_policy_network(episode_number)
        self.memory = MemoryMultiAction()
        self.episode_count += 1

    def __accumulate_gradient(self) -> None:
        if len(self.memory.rewards) == 0:
            print("WARNING: No rewards received - skipping training")
            return

        if np.sum(self.memory.rewards) == 0:
            print("WARNING: Zero total reward - skipping training")
            return

        episode_states = np.vstack(self.memory.states)
        episode_hidden_layers = np.vstack(self.memory.hidden_layers)
        episode_dlogps = np.vstack(self.memory.dlogps)
        episode_rewards = np.vstack(self.memory.rewards)

        # Add entropy bonuses to rewards (reduced)
        if hasattr(self.memory, "entropies") and self.memory.entropies:
            episode_entropies = np.vstack(self.memory.entropies)
            episode_rewards += episode_entropies * 0.1  # Reduced entropy bonus

        # Better reward normalization
        episode_discounted_rewards = self.__discount_and_normalize_rewards(
            episode_rewards, self.hyperparams.gamma
        )

        # Clip rewards to prevent extreme values
        episode_discounted_rewards = np.clip(episode_discounted_rewards, -10, 10)

        episode_dlogps *= episode_discounted_rewards

        if np.any(np.isnan(episode_dlogps)) or np.any(np.isinf(episode_dlogps)):
            print("🚨 CRITICAL: NaN or infinite gradients detected! Skipping update.")
            return

        # Better gradient clipping
        gradient_magnitude = np.mean(np.abs(episode_dlogps))
        if gradient_magnitude > 0.5:  # More conservative clipping
            episode_dlogps = episode_dlogps * (0.5 / gradient_magnitude)
            print(f"Clipped gradients from {gradient_magnitude:.3f} to 0.5")

        self.policy_network.backward_pass(
            episode_hidden_layers, episode_dlogps, episode_states
        )

    def __train_policy_network(self, episode_number: int) -> None:
        if episode_number % self.hyperparams.batch_size == 0:
            # Adaptive learning rate
            if len(self.total_rewards) > 20:
                recent_avg = np.mean(self.total_rewards[-20:])
                if recent_avg < self.baseline_reward:
                    lr = (
                        self.hyperparams.learning_rate * 0.5
                    )  # Reduce LR if doing poorly
                else:
                    lr = self.hyperparams.learning_rate
            else:
                lr = self.hyperparams.learning_rate

            self.policy_network.train(lr, self.hyperparams.decay_rate)

    def __save_policy_network(self, episode_number: int) -> None:
        if episode_number % self.hyperparams.save_interval == 0:
            self.policy_network.save_network(episode_number)

    def __discount_and_normalize_rewards(self, rewards, gamma):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            # NOTE: pg/agent.py resets running_add on any non-zero reward. That
            # is correct for Pong, where scoring ends the rally and is therefore
            # a real episode boundary ("pong specific!" in its comment). In
            # MsPacman every pellet scores, so resetting here truncated the
            # return to a single step and destroyed all credit assignment.
            running_add = running_add * gamma + rewards[t]
            discounted_rewards[t] = running_add

        # Better normalization
        if np.std(discounted_rewards) > 1e-8:
            discounted_rewards -= np.mean(discounted_rewards)
            discounted_rewards /= np.std(discounted_rewards) + 1e-8

        return discounted_rewards
