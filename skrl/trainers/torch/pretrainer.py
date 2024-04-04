import torch

class Pretrainer:
    def __init__ (self, agent, lr, epochs, batch_size):
        self._agent = agent
        self._learning_rate = lr
        self._demo_memory = self._agent._demonstration_memory
        self.optimizer = torch.optim.Adam(self._agent.policy.parameters(), lr=self._learning_rate)
        self.epochs = epochs
        self.batch_size = batch_size

        self.log_loss = []
        self._evaluate_expert()

    def _evaluate_expert(self):
        terminateds = self._demo_memory._tensor_terminated.squeeze().squeeze()
        rewards = self._demo_memory._tensor_rewards.squeeze().squeeze()
        ep_rw = []
        eps_rw = []
        for t, r in enumerate(rewards):
            if terminateds[t]:
                eps_rw.append(ep_rw)
                ep_rw = []
            ep_rw.append(r)
        
        print(f'Average expert reward: {torch.mean(torch.tensor([sum(ep) for ep in eps_rw]))}')
        print(f'Max expert episode reward: {torch.max(torch.tensor([sum(ep) for ep in eps_rw]))}')
        print(f'Min expert episode reward: {torch.min(torch.tensor([sum(ep) for ep in eps_rw]))}')

    def train_bc(self):
        print("-----------------Pretraining Policy --------------")
        dataset_length = len(self._demo_memory._tensor_terminated.squeeze().squeeze())
        for epoch in range(self.epochs):
            print(f'Epoch: {epoch}/{self.epochs}')
            for iter in range(dataset_length // self.batch_size):
                demo_states, y, _, _ = self._agent._demonstration_memory.sample(names=self._agent._demonstration_tensors_names, batch_size=self.batch_size)[0]
                x = self._agent._state_preprocessor(demo_states)
                act, _, _ = self._agent.policy.act({"states": x}, role="policy")
                bc_loss = torch.norm(act - y, p=2)

                self.optimizer.zero_grad()
                bc_loss.backward()
                self.optimizer.step()

                self.log_loss.append(bc_loss)
        self.log_loss = torch.tensor(self.log_loss)
        print(f'Terminal Loss: {self.log_loss[-1]}')
