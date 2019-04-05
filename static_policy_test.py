import torch
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import gym
import gym_aero
import sys
import config as cfg
import numpy as np

class MDN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, skills):
        super(MDN,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.skills = skills

        self.gamma = 0.99
        self.lmbd = 0.92
        self.eps = 0.05

        #self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc1 = torch.nn.Linear(input_dim+int(len(skills)*output_dim), hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.mu = torch.nn.Linear(hidden_dim, output_dim)
        self.logvar = torch.nn.Linear(hidden_dim, output_dim)
        self.pi = torch.nn.Linear(hidden_dim, int(len(skills)+1))
        self.value = torch.nn.Linear(hidden_dim, 1)

        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        mus = self.mu(x)
        logvars = self.logvar(x)
        pi = F.softmax(self.pi(x), dim=-1)
        value = self.value(x)
        return mus, logvars, pi, value
    
    """
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        mus = self.mu(x)
        logvars = self.logvar(x)
        value = self.value(x)
        return mus, logvars, value
    """
    
    def update(self, optim, trajectory):
        states = torch.stack(trajectory["states"]).float()
        actions = torch.stack(trajectory["actions"]).float()
        beta_log_probs = torch.stack(trajectory["log_probs"]).float()
        rewards = torch.stack(trajectory["rewards"]).float()
        values = torch.stack(trajectory["values"]).float()
        masks = torch.stack(trajectory["masks"])
        returns = Tensor(rewards.size(0),1)
        deltas = Tensor(rewards.size(0),1)
        advantages = Tensor(rewards.size(0),1)
        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for _ in range(4):
            for i in reversed(range(rewards.size(0))):
                returns[i] = rewards[i]+self.gamma*prev_return*masks[i]
                deltas[i] = rewards[i]+self.gamma*prev_value*masks[i]-values.data[i]
                advantages[i] = deltas[i]+self.gamma*self.lmbd*prev_advantage*masks[i]
                prev_return = returns[i, 0]
                prev_value = values.data[i, 0]
                prev_advantage = advantages[i, 0]
            advantages = (advantages-advantages.mean())/(advantages.std())
            returns = (returns-returns.mean())/(returns.std())

            # get action probabilities under current policy
            mus = []
            sigmas = []
            for _, s in enumerate(self.skills):
                s_mu, s_logvar = s.beta(states)
                mus.append(s_mu.detach())
                sigmas.append(s_logvar.exp().sqrt().detach())
            augmented_states = torch.cat([states]+[Tensor(mu) for mu in mus], dim=-1)
            mu_pi, logvar_pi, pi, _ = self.forward(augmented_states)
            mus.append(mu_pi)
            sigmas.append(logvar_pi.exp().sqrt())
            
            probs = 0.
            for i, (m, s) in enumerate(zip(mus, sigmas)):
                dist = Normal(m, s)
                norm_probs = torch.exp(dist.log_prob(actions))
                probs += pi[:,i].unsqueeze(1)*norm_probs
            pi_log_probs = torch.log(probs)

            #dist_pi = Normal(mu_pi, logvar_pi.exp().sqrt())
            #pi_log_probs = dist_pi.log_prob(actions)
            ratio = (pi_log_probs-beta_log_probs.detach()).sum(dim=1, keepdim=True).exp()
            optim.zero_grad()
            actor_loss = -torch.min(ratio*advantages, torch.clamp(ratio, 1-self.eps, 1+self.eps)*advantages).mean()
            critic_loss = F.smooth_l1_loss(values, returns)
            loss = actor_loss+critic_loss
            loss.backward(retain_graph=True)
            optim.step()
    
    """
    def update(self, optim, trajectory):
        states = torch.stack(trajectory["states"]).float()
        actions = torch.stack(trajectory["actions"]).float()
        beta_log_probs = torch.stack(trajectory["log_probs"]).float()
        rewards = torch.stack(trajectory["rewards"]).float()
        values = torch.stack(trajectory["values"]).float()
        masks = torch.stack(trajectory["masks"])
        returns = Tensor(rewards.size(0),1)
        deltas = Tensor(rewards.size(0),1)
        advantages = Tensor(rewards.size(0),1)
        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for _ in range(4):
            for i in reversed(range(rewards.size(0))):
                returns[i] = rewards[i]+self.gamma*prev_return*masks[i]
                deltas[i] = rewards[i]+self.gamma*prev_value*masks[i]-values.data[i]
                advantages[i] = deltas[i]+self.gamma*self.lmbd*prev_advantage*masks[i]
                prev_return = returns[i, 0]
                prev_value = values.data[i, 0]
                prev_advantage = advantages[i, 0]
            advantages = (advantages-advantages.mean())/(advantages.std())
            returns = (returns-returns.mean())/(returns.std())
            mu_pi, logvar_pi, _ = self.forward(states)
            dist_pi = Normal(mu_pi, logvar_pi.exp().sqrt())
            pi_log_probs = dist_pi.log_prob(actions)
            ratio = (pi_log_probs-beta_log_probs.detach()).sum(dim=1, keepdim=True).exp()
            optim.zero_grad()
            actor_loss = -torch.min(ratio*advantages, torch.clamp(ratio, 1-self.eps, 1+self.eps)*advantages).mean()
            critic_loss = F.smooth_l1_loss(values, returns)
            loss = actor_loss+critic_loss
            loss.backward(retain_graph=True)
            optim.step()
    """

    def select_action(self, x):
        mus = []
        sigmas = []
        for _, s in enumerate(self.skills):
            s_mu, s_logvar = s.beta(x)
            mus.append(s_mu.detach())
            sigmas.append(s_logvar.exp().sqrt().detach())
        state = torch.cat([x]+[Tensor(mu) for mu in mus], dim=-1)
        mu, logvar, pi, value = self.forward(state)
        #print(logvar.exp().sqrt())
        dist = Normal(mu, logvar.exp().sqrt())
        a = dist.sample()
        candidate_actions = mus+[a]
        mus.append(mu)
        sigmas.append(logvar.exp().sqrt())
        action = 0.
        for i, a in enumerate(candidate_actions):
            action += pi[i]*a
        prob = 0.
        for i, (m, s) in enumerate(zip(mus, sigmas)):
            dist = Normal(m, s)
            norm_prob = torch.exp(dist.log_prob(action))
            prob += pi[i]*norm_prob
        log_prob = torch.log(prob)
        return action, value, log_prob
    
    """
    def select_action(self, x):
        mu, logvar, value = self.forward(x)
        sigma = logvar.exp().sqrt()
        dist = Normal(mu, sigma)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, value, log_prob
    """
    
    
"""
# create first policy, train on env 1
env_name = "One-v0"
params = cfg.trpo
import algs.ind.trpo_peb as trpo_peb
print("---Initializing TRPO-PEB in env: "+env_name+"---")
trpo_peb.Trainer(env_name, params)

# run second environment
env_name = "Two-v0"
params = cfg.trpo
import algs.ind.trpo_peb as trpo_peb
print("---Initializing TRPO-PEB in env: "+env_name+"---")
trpo_peb.Trainer(env_name, params)
"""

"""
env_name = "Three-v0"
params = cfg.trpo
import algs.ind.trpo_peb as trpo_peb
print("---Initializing TRPO-PEB in env: "+env_name+"---")
trpo_peb.Trainer(env_name, params)
"""

# run composite policy
log_interval = 1
env = gym.make("Three-v0")
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
policy_1 = torch.load("/home/seanny/quadrotor/saved_policies/trpo-1-One-v0.pth.tar")
policy_2 = torch.load("/home/seanny/quadrotor/saved_policies/trpo-1-Two-v0.pth.tar")
if cfg.trpo["cuda"]:
    policy_1 = policy_1.cuda()
    policy_2 = policy_2.cuda()
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.Tensor
skills = [policy_1, policy_2]
iterations = 300
batch_size = 4096

for _ in range(5):
    agent  = MDN(state_dim, 128, action_dim, skills)
    agent = agent.cuda()
    crit_opt = torch.optim.Adam(agent.parameters(),lr=1e-4)
    count = 1
    interval_avg = []
    avg = 0
    for ep in range(1, iterations+1):
        s_, a_, ns_, r_, v_, lp_, masks = [], [], [], [], [], [], []
        num_steps = 1
        reward_batch = 0
        num_episodes = 0
        while num_steps < batch_size+1:
            state = env.reset()
            state = Tensor(state)
            reward_sum = 0
            t = 0
            done = False
            while not done:
                #if ep % log_interval == 0:
                    #env.render()
                action, value, log_prob = agent.select_action(state)
                next_state, reward, done, _ = env.step(action.cpu().data.numpy())
                reward_sum += reward
                next_state = Tensor(next_state)
                reward = Tensor([reward])
                s_.append(state)
                ns_.append(next_state)
                a_.append(action)
                r_.append(reward)
                v_.append(value)
                lp_.append(log_prob)
                masks.append(Tensor([not done]))
                state = next_state
                t += 1
            num_steps += t
            num_episodes += 1
            reward_batch += reward_sum
        reward_batch /= num_episodes
        interval_avg.append(reward_batch)
        avg = (avg*(ep-1)+reward_batch)/ep        
        trajectory = {
                    "states": s_,
                    "actions": a_,
                    "rewards": r_,
                    "next_states": ns_,
                    "masks": masks,
                    "values": v_,
                    "log_probs": lp_
                    }

        agent.update(crit_opt, trajectory)
        if ep % log_interval == 0:
            interval = float(sum(interval_avg))/float(len(interval_avg))
            print('Episode {}\t Interval average: {:.3f}\t Average reward: {:.3f}'.format(ep, interval, avg))
            interval_avg = []