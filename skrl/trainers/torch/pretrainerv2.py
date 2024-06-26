import itertools

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset, random_split
import random
import copy

class ExpertDataSet(Dataset):
    def __init__(self, expert_observations, expert_actions):
        self.observations = expert_observations
        self.actions = expert_actions

    def __getitem__(self, index):
        return (self.observations[index], self.actions[index])

    def __len__(self):
        return len(self.observations)

class PretrainerV2:
    def __init__(self, agent, transitions, lr, epochs, batch_size, log_interval=100):
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
        """
        self._agent = agent
        self._learning_rate = lr
        self.log_interval = log_interval

        self.optimizer = torch.optim.Adam(self._agent.policy.parameters(), lr=self._learning_rate)

        self.epochs = epochs
        self.batch_size = batch_size
        self.log_policy_loss = []
        self.log_std = []
        self.log_mse = []

        s = []
        a = []
        for transition in transitions:
            s.append(transition["states"].squeeze(0))
            a.append(transition["actions"].squeeze(0))

        expert_dataset = ExpertDataSet(s, a)

        self.train_size = int(0.8 * len(expert_dataset))

        self.test_size = len(expert_dataset) - self.train_size

        self.train_expert_dataset, self.test_expert_dataset = random_split(
            expert_dataset, [self.train_size, self.test_size]
        )

        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_expert_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.test_expert_dataset, batch_size=batch_size, shuffle=True)
        
        # a_t = torch.stack(expert_dataset.actions)
        # expert_log_std = torch.log(torch.std(a_t, dim=0) + 1e-12)
        # with torch.no_grad():
        #     self._agent.policy.log_std_parameter.copy_(expert_log_std)

    def train_bc(self):
        
        """
        Train the behavior cloning (BC) model.
        """
        print("-----------------Pretraining Policy --------------")
        criterion = nn.MSELoss()

        def train_batch():
            self._agent.policy.train()

            for batch_idx, (state, action_target) in enumerate(self.train_loader):
                state = self._agent._state_preprocessor(state, train = not batch_idx)
                self.optimizer.zero_grad()
                action, _, mean_a = self._agent.policy.act({"states": state}, role="policy")
                # action_prediction = mean_a["mean_actions"]
                action_prediction = action

                loss = criterion(action_prediction, action_target)
                loss.backward()
                self.optimizer.step()
                if batch_idx % self.log_interval == 0:
                    print(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch,
                            batch_idx * len(state),
                            len(self.train_loader.dataset),
                            100.0 * batch_idx / len(self.train_loader),
                            loss.item(),
                        )
                    )
            return loss.item()
        
        def test_batch():
            self._agent.policy.eval()
            test_loss = 0
            with torch.no_grad():
                for state, action_target in self.test_loader:
                    state = self._agent._state_preprocessor(state, train=False)

                    self.optimizer.zero_grad()
                    action, _, mean_a = self._agent.policy.act({"states": state}, role="policy")
                    # action_prediction = mean_a["mean_actions"]
                    action_prediction = action


                    test_loss = criterion(action_prediction, action_target)

            # test_loss /= len(self.test_loader.dataset)
            print(f"Test set: Average loss: {test_loss:.4f}")
            return test_loss.item()

        self.log_policy_loss = []
        self.log_policy_test_loss = []
        for epoch in range(1, self.epochs + 1):
            self.log_policy_loss.append(train_batch())
            self.log_policy_test_loss.append(test_batch())