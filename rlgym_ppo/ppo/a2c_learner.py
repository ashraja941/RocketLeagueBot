import os
import time

import numpy as np
import torch

from rlgym_ppo.ppo import ContinuousPolicy, DiscreteFF, MultiDiscreteFF, ValueEstimator
class A2CLearner(object):
    def __init__(
        self,
        obs_space_size,
        act_space_size,
        policy_type,
        policy_layer_sizes,
        critic_layer_sizes,
        continuous_var_range,
        batch_size,
        policy_lr,
        critic_lr,
        ent_coef,
        device,
    ):
        self.device = device

        if policy_type == 2:
            self.policy = ContinuousPolicy(
                obs_space_size,
                act_space_size * 2,
                policy_layer_sizes,
                device,
                var_min=continuous_var_range[0],
                var_max=continuous_var_range[1],
            ).to(device)
        elif policy_type == 1:
            self.policy = MultiDiscreteFF(
                obs_space_size, policy_layer_sizes, device
            ).to(device)
        else:
            self.policy = DiscreteFF(
                obs_space_size, act_space_size, policy_layer_sizes, device
            ).to(device)

        self.value_net = ValueEstimator(obs_space_size, critic_layer_sizes, device).to(
            device
        )

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=critic_lr
        )
        self.value_loss_fn = torch.nn.MSELoss()

        self.ent_coef = ent_coef
        self.batch_size = batch_size

    def learn(self, exp):
        """
        Compute A2C updates with an experience buffer.

        Args:
            exp (ExperienceBuffer): Experience buffer containing training data.

        Returns:
            dict: Dictionary containing training report metrics.
        """
        # Get all shuffled batches from the experience buffer
        batches = exp.get_all_batches_shuffled(self.batch_size)

        # Initialize metrics
        n_iterations = 0
        mean_entropy = 0
        mean_val_loss = 0
        mean_policy_loss = 0
        mean_clip = 0  # Not used in A2C
        mean_divergence = 0  # Not used in A2C
        t1 = time.time()  # Start timing

        # Save parameters before updates for magnitude tracking
        policy_before = torch.nn.utils.parameters_to_vector(self.policy.parameters()).cpu()
        critic_before = torch.nn.utils.parameters_to_vector(self.value_net.parameters()).cpu()

        for batch in batches:
            (
                batch_acts,
                batch_old_probs,  # Not used in A2C but kept for compatibility
                batch_obs,
                batch_target_values,
                batch_advantages,
            ) = batch

            batch_acts = batch_acts.to(self.device)
            batch_obs = batch_obs.to(self.device)
            batch_advantages = batch_advantages.to(self.device)
            batch_target_values = batch_target_values.to(self.device)

            # Zero gradients
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()

            # Compute value estimates
            vals = self.value_net(batch_obs).view_as(batch_target_values)

            # Compute policy log probs & entropy
            log_probs, entropy = self.policy.get_backprop_data(batch_obs, batch_acts)

            # Compute policy and value losses
            policy_loss = -(log_probs * batch_advantages).mean()
            value_loss = self.value_loss_fn(vals, batch_target_values)
            entropy_loss = -self.ent_coef * entropy.mean()

            # Total loss
            total_loss = policy_loss + value_loss + entropy_loss

            # Backpropagation and optimization step
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)

            self.policy_optimizer.step()
            self.value_optimizer.step()

            # Update metrics
            n_iterations += 1
            mean_policy_loss += policy_loss.item()
            mean_val_loss += value_loss.item()
            mean_entropy += entropy.mean().item()

        # Save parameters after updates for magnitude tracking
        policy_after = torch.nn.utils.parameters_to_vector(self.policy.parameters()).cpu()
        critic_after = torch.nn.utils.parameters_to_vector(self.value_net.parameters()).cpu()

        # Compute update magnitudes
        policy_update_magnitude = (policy_before - policy_after).norm().item()
        critic_update_magnitude = (critic_before - critic_after).norm().item()

        # Timing
        elapsed_time = time.time() - t1

        # Average metrics over all iterations
        mean_policy_loss /= n_iterations
        mean_val_loss /= n_iterations
        mean_entropy /= n_iterations

        # Report metrics
        report = {
            "PPO Batch Consumption Time": elapsed_time / n_iterations,
            "Cumulative Model Updates": n_iterations,  # Equivalent to the number of batches processed
            "Policy Entropy": mean_entropy,
            "Mean KL Divergence": mean_divergence,  # A2C does not use KL divergence
            "Value Function Loss": mean_val_loss,
            "SB3 Clip Fraction": mean_clip,  # A2C does not use clipping
            "Policy Update Magnitude": policy_update_magnitude,
            "Value Function Update Magnitude": critic_update_magnitude,
        }

        return report


    def save_to(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(folder_path, "A2C_POLICY.pt"))
        torch.save(
            self.value_net.state_dict(), os.path.join(folder_path, "A2C_VALUE_NET.pt")
        )
        torch.save(
            self.policy_optimizer.state_dict(),
            os.path.join(folder_path, "A2C_POLICY_OPTIMIZER.pt"),
        )
        torch.save(
            self.value_optimizer.state_dict(),
            os.path.join(folder_path, "A2C_VALUE_NET_OPTIMIZER.pt"),
        )

    def load_from(self, folder_path):
        assert os.path.exists(folder_path), "A2C LEARNER CANNOT FIND FOLDER {}".format(
            folder_path
        )

        self.policy.load_state_dict(
            torch.load(os.path.join(folder_path, "A2C_POLICY.pt"))
        )
        self.value_net.load_state_dict(
            torch.load(os.path.join(folder_path, "A2C_VALUE_NET.pt"))
        )
        self.policy_optimizer.load_state_dict(
            torch.load(os.path.join(folder_path, "A2C_POLICY_OPTIMIZER.pt"))
        )
        self.value_optimizer.load_state_dict(
            torch.load(os.path.join(folder_path, "A2C_VALUE_NET_OPTIMIZER.pt"))
        )
