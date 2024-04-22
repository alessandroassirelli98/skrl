import itertools

import torch
import torch.nn as nn


class Pretrainer:
    def __init__(self, agent, transitions, lr, epochs, batch_size, value_train=False):
        """
        Initialize the Pretrainer object.

        Parameters:
        - agent: The agent object being trained.
        - transitions: The dataset of transitions for pretraining.
        - type transitions: list({'states': torch.tensor,
                                'actions': torch.tensor,
                                'reward': torch.tensor,
                                'next_states': torch.tensor,
                                'terminated': torch.tensor}, {...}, {...})
        - lr: The learning rate for the optimizer.
        - epochs: The number of training epochs.
        - batch_size: The batch size for training.
        - value_train: Whether to train the value function or not (default False).
        """
        self._agent = agent
        self._learning_rate = lr
        self.dataset = transitions

        # Setting up optimizer based on whether policy and value are separate or not
        if self._agent.policy is self._agent.value:
            self.optimizer = torch.optim.Adam(self._agent.policy.parameters(), lr=self._learning_rate)
        else:
            self.optimizer = torch.optim.Adam(
                itertools.chain(self._agent.policy.parameters(), self._agent.value.parameters()),
                lr=self._learning_rate
            )

        self.epochs = epochs
        self.batch_size = batch_size
        self.train_value = value_train
        self.log_policy_loss = []
        self.log_value_loss = []
        self.log_std = []
        self.log_mse = []
        self.dataset_length = len(self.dataset)
        self._split_dataset()

    def get_batch(self, dataset, indices):
        """
        Retrieve a batch of data from the dataset.

        Parameters:
        - dataset: The dataset from which to retrieve the batch.
        - indices: The indices of the data points in the batch.

        Returns:
        - s: The states in the batch.
        - a: The actions in the batch.
        - r: The rewards in the batch.
        - ns: The next states in the batch.
        - t: The termination flags in the batch.
        """
        s = []
        a = []
        r = []
        ns = []
        t = []
        batch = [dataset[i] for i in indices]
        for transition in batch:
            s.append(transition["states"])
            a.append(transition["actions"])
            r.append(transition["reward"])
            ns.append(transition["next_states"])
            t.append(transition["terminated"])
        s = torch.stack(s).squeeze()
        a = torch.stack(a).squeeze()
        r = torch.stack(r)
        ns = torch.stack(ns).squeeze()
        t = torch.stack(t)
        return s, a, r, ns, t

    def _split_dataset(self):
        """
        Split the dataset into training and testing sets.
        """
        self.train_len = int(0.8 * self.dataset_length)
        self.test_len = int(0.2 * self.dataset_length)
        train_indices = torch.arange(self.dataset_length)[: self.train_len]
        test_indices = torch.arange(self.dataset_length)[self.test_len:]
        self.train_dataset = [self.dataset[i] for i in train_indices]
        self.test_dataset = [self.dataset[i] for i in test_indices]

    def train_bc(self):
        """
        Train the behavior cloning (BC) model.
        """
        print("-----------------Pretraining Policy --------------")
        value_loss = 0.
        bc_loss = 0.
        for epoch in range(self.epochs):
            print(f'Epoch: {epoch}/{self.epochs}')
            iter_range = self.train_len // self.batch_size
            cumulative_mse = torch.zeros(22, device=self._agent.device)
            cumulative_std = torch.zeros(22, device=self._agent.device)
            cumulative_policy_loss = torch.zeros(1, device=self._agent.device)
            cumulative_value_loss = torch.zeros(1, device=self._agent.device)
            for iter in range(iter_range):
                rnd_indices = torch.arange(iter * self.batch_size, (iter + 1) * self.batch_size, dtype=int)
                s, a, r, ns, t = self.get_batch(self.train_dataset, rnd_indices)
                ns = self._agent._state_preprocessor(ns, train=not epoch)
                s = self._agent._state_preprocessor(s, train=not epoch)

                _, log_prob, mean_a = self._agent.policy.act({"states": s, "taken_actions": a}, role="policy")
                mean_a = mean_a["mean_actions"]
                bc_loss = torch.mean((mean_a - a)**2)  # Minimize MSE

                if self.train_value:
                    rtgo, v = self._compute_rtgo(s, r, ns, t)
                    rtgo = self._agent._value_preprocessor(rtgo, train=not epoch)
                    value_loss = self._agent._value_loss_scale * torch.functional.F.mse_loss(rtgo, v)

                self.optimizer.zero_grad()
                (bc_loss + value_loss).backward()
                if self._agent._grad_norm_clip > 0:
                    if self._agent.policy is self._agent.value:
                        nn.utils.clip_grad_norm_(self._agent.policy.parameters(), self._agent._grad_norm_clip)
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(self._agent.policy.parameters(), self._agent.value.parameters()),
                            self._agent._grad_norm_clip
                        )
                self.optimizer.step()
                cumulative_policy_loss += (bc_loss.detach())
                cumulative_value_loss += value_loss.detach() if self.train_value else value_loss
                cumulative_std += (torch.exp(self._agent.policy.log_std_parameter.detach()))
                cumulative_mse += (torch.mean((a - mean_a)**2, dim=0).detach())

            epoch_loss = cumulative_policy_loss / (iter + 1)
            self.log_std.append(cumulative_std / (iter + 1))
            self.log_mse.append(cumulative_mse / (iter + 1))
            self.log_policy_loss.append(epoch_loss)
            self.log_value_loss.append(cumulative_value_loss / (iter + 1))

        self.log_policy_loss = torch.tensor(self.log_policy_loss)
        self.log_value_loss = torch.tensor(self.log_value_loss)
        self.log_std = torch.stack(self.log_std)
        self.log_mse = torch.stack(self.log_mse)
        print(f'Terminal Loss: {self.log_policy_loss[-1]}')

    def _compute_rtgo(self, states, rewards, next_states, terminateds):
        """
        Compute the return (RTGO) for each state in the batch.

        Parameters:
        - states: The states at each time step.
        - rewards: The rewards received at each time step.
        - next_states: The next states at each time step.
        - terminateds: Flags indicating whether each episode terminated.

        Returns:
        - rtgo: The return (RTGO) for each time state.
        - v: The estimated value function for each time state.
        """
        not_t = (~terminateds.clone()).long()
        rtgo = torch.zeros_like(rewards)
        adv = 0
        v, _, _ = self._agent.value.act({"states": states}, role="value")
        v_next, _, _ = self._agent.value.act({"states": next_states}, role="value")
        v_next = self._agent._value_preprocessor(v_next, inverse=True)
        for i in reversed(range(self.batch_size)):
            adv = rewards[i] - v[i] + self._agent._discount_factor * not_t[i] * (v_next[i] + adv * self._agent._lambda)
            rtgo[i] = adv + v[i]
        return rtgo, v

    def test_bc(self):
        """
        Test the behavior cloning (BC) model.
        """
        self.test_policy_loss = []
        self.test_value_loss = []
        iter_range = self.test_len // self.batch_size
        for iter in range(iter_range):
            rnd_indices = torch.arange(iter * self.batch_size, (iter + 1) * self.batch_size, dtype=int)
            s, a, r, ns, t = self.get_batch(self.test_dataset, rnd_indices)
            s = self._agent._state_preprocessor(s)
            r = r.squeeze()
            t = t.squeeze()
            _, log_prob, mean_a = self._agent.policy.act({"states": s, "taken_actions": a}, role="policy")
            mean_a = mean_a["mean_actions"]
            bc_loss = torch.mean((mean_a - a)**2).detach()
            rtgo, v = self._compute_rtgo(s, r, ns, t)
            rtgo = self._agent._value_preprocessor(rtgo)
            value_loss = self._agent._value_loss_scale * torch.functional.F.mse_loss(rtgo, v)
            self.test_policy_loss.append(bc_loss)
            self.test_value_loss.append(value_loss)
        self.test_policy_loss = torch.tensor(self.test_policy_loss)
        self.test_value_loss = torch.tensor(self.test_value_loss)
        print(f'Test Mean Loss: {torch.mean(self.test_policy_loss)}')
        print(f'Test Mean Value Loss: {torch.mean(self.test_value_loss)}')
        return 0
