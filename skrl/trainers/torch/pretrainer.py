import torch

class Pretrainer:
    def __init__ (self, agent, transitions, lr, epochs, batch_size):
        self._agent = agent
        self._learning_rate = lr
        self.dataset = transitions
        self.optimizer = torch.optim.Adam(self._agent.policy.parameters(), lr=self._learning_rate)
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_policy_loss = []
        self.log_value_loss = []

        self.log_std = []
        self.log_mse = []

        self.dataset_length = len(self.dataset)

    # def _create_dataset(self):
    #     self.dataset = []
    #     for i in range(self.dataset_length):
    #         actions = self._demo_memory._tensor_actions.squeeze()[i, :]
    #         states = self._demo_memory._tensor_states.squeeze()[i, :]
    #         reward = self._demo_memory._tensor_rewards.squeeze(1)[i, :]
    #         next_states = self._demo_memory._tensor_next_states.squeeze()[i, :]
    #         terminated = self._demo_memory._tensor_terminated.squeeze(1)[i, :]

    #         # states = self._agent._state_preprocessor(states)
    #         # next_states = self._agent._state_preprocessor(next_states)

    #         dict = {}
    #         dict["states"] = states
    #         dict["actions"] = actions
    #         dict["reward"] = reward
    #         dict["next_states"] = next_states
    #         dict["terminated"] = terminated
    #         self.dataset.append(dict)

    #     # Fit preprocessor
    #     s, _,_,_,_ = self.get_batch(self.dataset, range(len(self.dataset)))
    #     v, _, _ = self._agent.value.act({"states": s}, role="value")
    #     self._agent._value_preprocessor(v, train = True)

    def get_batch(self, dataset, indices):
        s =[]
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

        return s,a,r,ns,t

    def _split_dataset(self):
        self.train_len = int(0.8 * self.dataset_length)
        self.test_len = int(0.2 * self.dataset_length)

        train_indices = torch.randperm(self.dataset_length)[: self.train_len]
        test_indices = torch.randperm(self.dataset_length)[self.test_len :]

        self.train_dataset = [self.dataset[i] for i in train_indices]
        self.test_dataset = [self.dataset[i] for i in test_indices]


    # def _evaluate_expert(self):
    #     terminateds = self._demo_memory._tensor_terminated.squeeze().squeeze()
    #     rewards = self._demo_memory._tensor_rewards.squeeze().squeeze()
    #     ep_rw = []
    #     eps_rw = []
    #     for t, r in enumerate(rewards):
    #         if terminateds[t]:
    #             eps_rw.append(ep_rw)
    #             ep_rw = []
    #         ep_rw.append(r)
        
    #     self.mean_exp_rew = torch.mean(torch.tensor([sum(ep) for ep in eps_rw]))
    #     self.max_exp_rew = torch.max(torch.tensor([sum(ep) for ep in eps_rw]))
    #     self.min_exp_rew = torch.min(torch.tensor([sum(ep) for ep in eps_rw]))
    #     print(f'Average expert reward: {self.mean_exp_rew}')
    #     print(f'Max expert episode reward: {self.max_exp_rew}')
    #     print(f'Min expert episode reward: {self.min_exp_rew}')


    def train_bc(self):
        self._split_dataset()
        print("-----------------Pretraining Policy --------------")

        for epoch in range(self.epochs):
            print(f'Epoch: {epoch}/{self.epochs}')
            iter_range = self.train_len // self.batch_size
            cumulative_mse = torch.zeros(22, device=self._agent.device)
            cumulative_std = torch.zeros(22, device = self._agent.device)
            cumulative_policy_loss = torch.zeros(1, device = self._agent.device)
            cumulative_value_loss = torch.zeros(1, device = self._agent.device)
            best_loss = torch.ones(1, device=self._agent.device) * 1000
            for iter in range(iter_range):
                rnd_indices = torch.randperm(self.train_len)[: self.batch_size]
                # rnd_indices = torch.arange(iter * self.batch_size, (iter + 1) * self.batch_size, dtype=int)
                s, a , r, ns, t = self.get_batch(self.train_dataset, rnd_indices)

                ns = self._agent._state_preprocessor(ns, train = not epoch)
                s = self._agent._state_preprocessor(s, train = not epoch)

                _, log_prob, mean_a = self._agent.policy.act({"states": s, "taken_actions": a}, role="policy")
                mean_a = mean_a["mean_actions"]

                # bc_loss = torch.norm(mean_a - a, p=2) # This minimizes the difference, but we want to make the demo actions more likely instead
                bc_loss = - log_prob.mean()
                
                # rtgo, v = self._compute_rtgo(s, r, ns, t)
                # tdt, v = self._compute_td_error(s, r, ns, t)
                
                # tdt = self._agent._value_preprocessor(tdt, train=True)

                # if self._agent._clip_predicted_values:
                #     v = tdt + torch.clip(v - tdt,
                #                         min=-self._agent._value_clip,
                #                         max=self._agent._value_clip)
                
                # value_loss = self._agent._value_loss_scale * torch.functional.F.mse_loss(tdt, v)

                self.optimizer.zero_grad()
                (bc_loss).backward()
                self.optimizer.step()

                cumulative_policy_loss += (bc_loss.detach())
                # self.log_value_loss.append(value_loss)
                cumulative_std += (torch.exp(self._agent.policy.log_std_parameter.detach()))
                cumulative_mse += (torch.mean((a - mean_a)**2, dim=0).detach())

            epoch_loss = cumulative_policy_loss / (iter + 1)
            self.log_std.append(cumulative_std / (iter + 1))
            self.log_mse.append(cumulative_mse / (iter + 1))
            self.log_policy_loss.append(epoch_loss)

            # if (epoch_loss < best_loss):
            #     print(f'Saving best model, loss: {epoch_loss[0]}')
            #     best_loss = epoch_loss.clone()
            #     best_model = self._agent.policy.state_dict()

            
        self.log_policy_loss = torch.tensor(self.log_policy_loss)
        self.log_value_loss = torch.tensor(self.log_value_loss)
        self.log_std = torch.stack(self.log_std)
        self.log_mse = torch.stack(self.log_mse)
        print(f'Terminal Loss: {self.log_policy_loss[-1]}')


    
    def _compute_td_error(self, s, r, ns, t):
        # with torch.no_grad():
        not_t = t.logical_not()
        v_next, _, _ = self._agent.value.act({"states": ns}, role="value")
        v_next = self._agent._value_preprocessor(v_next, inverse=True)


        v, _, _ = self._agent.value.act({"states": s}, role="value")
        # td = reward(s_t,a_t) if s is terminal, 
        # td = reward(s_t,a_t) + gamma * v(s_{t+1}) if s_t not terminal
        # with torch.no_grad():
        tdt = r + self._agent._discount_factor * v_next * not_t
        return tdt , v 

    def _compute_rtgo(self, states, rewards, next_states, terminateds):
        not_t = (~terminateds.clone()).long()
        rtgo = torch.zeros_like(rewards)
        adv = 0
        v, _, _ = self._agent.value.act({"states": states}, role="value")
        v_next, _, _ = self._agent.value.act({"states": next_states}, role="value")

        # v = self._agent._value_preprocessor(v, inverse=True)
        # v_next = self._agent._value_preprocessor(v_next, inverse=True)
        for i in reversed(range(self.batch_size)):  
            adv = rewards[i] - v[i] + self._agent._discount_factor * not_t[i] * (v_next[i] + adv * self._agent._lambda)
            rtgo[i] = adv + v[i]

        return rtgo, v
    
    def test_bc(self):
        self.test_loss = []
        replay_action = []
        for iter in range(len(self.dataset)):
            s, a, _,_,_ = self.get_batch(self.dataset, [iter])
            s = self._agent._state_preprocessor(s)
            with torch.no_grad():
                _, log_prob, mean_a = self._agent.policy.act({"states": s, "taken_actions": a}, role="policy")
                mean_a = mean_a["mean_actions"]


                bc_loss = (a - mean_a) ** 2
            # bc_loss = - log_prob.mean()
            
            self.test_loss.append(bc_loss)
            replay_action.append(mean_a)


        self.test_loss = torch.stack(self.test_loss)
        replay_action = torch.stack(replay_action)
        print(f'Test Mean Loss: {torch.mean(self.test_loss)}')
        return replay_action

        # self.test_loss = []
        # for iter in range(self.test_len // self.batch_size):
        #     rnd_indices = torch.randperm(self.test_len)[: self.batch_size]
        #     s, a, _,_,_ = self.get_batch(self.test_dataset, rnd_indices)
        #     s = self._agent._state_preprocessor(s)

        #     _, log_prob, mean_a = self._agent.policy.act({"states": s, "taken_actions": a}, role="policy")
        #     mean_a = mean_a["mean_actions"]

        #     bc_loss = torch.norm( mean_a - a, p=2)
        #     # bc_loss = - log_prob.mean()
            
        #     self.test_loss.append(bc_loss)

        # self.test_loss = torch.tensor(self.test_loss)
        # print(f'Test Mean Loss: {torch.mean(self.test_loss)}')

