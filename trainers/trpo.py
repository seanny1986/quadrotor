import environments.envs as envs 
import policies.trpo as trpo
import argparse
import torch
import torch.nn.functional as F
import math
import utils
import numpy as np

class Trainer:
    def __init__(self, env_name, params):
        self.env = envs.make(env_name)
        self.params = params

        self.iterations = params["iterations"]
        self.gamma = params["gamma"]
        self.tau = params["tau"]
        self.l2_reg = params["l2_reg"]
        self.max_kl = params["max_kl"]
        self.damping = params["damping"]
        self.seed = params["seed"]
        self.batch_size = params["batch_size"]
        self.render = params["render"]
        self.log_interval = params["log_interval"]
        self.save = params["save"]
        
        cuda = params["cuda"]
        state_dim = self.env.observation_space
        action_dim = self.env.action_space
        hidden_dim = params["hidden_dim"]

        self.pi = fmis.Actor(state_dim, hidden_dim, action_dim)
        self.beta = fmis.Actor(state_dim, hidden_dim, action_dim)
        self.phi = fmis.Dynamics(state_dim+action_dim, hidden_dim, state_dim)
        self.critic = fmis.Critic(state_dim, hidden_dim, 1)
        self.agent = fmis.FMIS(self.pi, self.beta, self.critic, self.phi, self.env, GPU=cuda)

        self.pi_optim = torch.optim.Adam(self.pi.parameters())
        self.phi_optim = torch.optim.Adam(self.phi.parameters())

        if cuda:
            self.Tensor = torch.cuda.FloatTensor
        else:
            self.Tensor = torch.Tensor
        
        if self.render:
            self.env.init_rendering()
            
        self.train()
    
    def train(self):
        for i_episode in count(1):
            memory = Memory()

            num_steps = 0
            reward_batch = 0
            num_episodes = 0
            while num_steps < args.batch_size:
                state = env.reset()
                state = running_state(state)

                reward_sum = 0
                for t in range(10000): # Don't infinite loop while learning
                    action = select_action(state)
                    action = action.data[0].numpy()
                    next_state, reward, done, _ = env.step(action)
                    reward_sum += reward

                    next_state = running_state(next_state)

                    mask = 1
                    if done:
                        mask = 0

                    memory.push(state, np.array([action]), mask, next_state, reward)

                    if args.render:
                        env.render()
                    if done:
                        break

                    state = next_state
                num_steps += (t-1)
                num_episodes += 1
                reward_batch += reward_sum

            reward_batch /= num_episodes
            batch = memory.sample()
            update_params(batch)

            if i_episode % args.log_interval == 0:
                print('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(i_episode, reward_sum, reward_batch))


num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

env.seed(args.seed)
torch.manual_seed(args.seed)

policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)

def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def update_params(batch):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    values = value_net(Variable(states))

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    # Original code uses the same LBFGS to optimize the value loss
    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values_ = value_net(Variable(states))

        value_loss = (values_ - targets).pow(2).mean()

        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * args.l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy()[0], get_flat_grad_from(value_net).data.double().numpy())

    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(value_net).double().numpy(), maxiter=25)
    set_flat_params_to(value_net, torch.Tensor(flat_params))

    advantages = (advantages - advantages.mean()) / advantages.std()

    action_means, action_log_stds, action_stds = policy_net(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    def get_loss(volatile=False):
        action_means, action_log_stds, action_stds = policy_net(Variable(states, volatile=volatile))
        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
        action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
        return action_loss.mean()


    def get_kl():
        mean1, log_std1, std1 = policy_net(Variable(states))

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    trpo_step(policy_net, get_loss, get_kl, args.max_kl, args.damping)

running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)

