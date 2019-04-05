import torch
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import gym
import gym_aero
import sys
import config as cfg
import numpy as np

class Actor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_skills):
        super(Actor, self).__init__()
        self.__fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.__fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.__mu = torch.nn.Linear(hidden_dim, output_dim)
        self.__logvar = torch.nn.Linear(hidden_dim, output_dim)
        self.__phi = torch.nn.Linear(hidden_dim, num_skills)
        self.__mu.weight.data.mul_(0.1)
        self.__mu.bias.data.mul_(0.)

        #self.__logvar.weight.data.mul_(10)
        #self.__logvar.bias.data.mul_(0.)

    def forward(self, x):
        x = F.tanh(self.__fc1(x.float()))
        x = F.tanh(self.__fc2(x))
        mu = self.__mu(x)
        logvar = self.__logvar(x)
        phi = F.softmax(self.__phi(x),dim=-1)
        return mu, logvar, phi

class Critic(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Critic, self).__init__()
        self.__fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.__value = torch.nn.Linear(hidden_dim, output_dim)
        self.__value.weight.data.mul_(0.1)
        self.__value.bias.data.mul_(0.)

    def forward(self, x):
        x = F.tanh(self.__fc1(x))
        state_values = self.__value(x)
        return state_values

class MDN(torch.nn.Module):
    def __init__(self, pi, beta, critic, skills, params, GPU=False):
        super(MDN, self).__init__()
        self.pi = pi
        self.beta = beta
        self.hard_update(self.beta, self.pi)
        self.critic = critic
        self.skills = skills
        self.__gamma = params["gamma"]
        self.__tau = params["tau"]
        self.__max_kl = params["max_kl"]
        self.__damping = params["damping"]
        self.__GPU = GPU
        if GPU:
            self.__Tensor = torch.cuda.FloatTensor
            self.pi = self.pi.cuda()
            self.beta = self.beta.cuda()
            self.critic = self.critic.cuda()
        else:
            self.__Tensor = torch.Tensor
    
    def conjugate_gradient(self, Avp, b, n_steps=10, residual_tol=1e-10):
        """
        Estimate the function Fv = g, where F is the FIM, and g is the gradient.
        Since dx ~= F^{-1}g for a stochastic process, v is dx. The CG algorithm 
        assumes the function is locally quadratic. In order to ensure our step 
        actually improves the policy, we need to do a linesearch after this.
        """
        x = torch.zeros(b.size())
        if self.__GPU: x = x.cuda()
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(n_steps):
            _Avp = Avp(p)
            alpha = rdotr/p.dot(_Avp)
            x += alpha*p
            r -= alpha*_Avp
            new_rdotr = r.dot(r)
            beta = new_rdotr/rdotr
            p = r+beta*p
            rdotr = new_rdotr
            if rdotr <= residual_tol:
                break
        return x

    def linesearch(self, model, func, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=.1):
        """
        Conducts an exponentially decaying linesearch to guarantee that our update step improves the
        model. 
        """
        fval = func(x).data
        steps = 0.5**torch.arange(max_backtracks).float()
        if self.__GPU: steps = steps.cuda()
        for (n, stepfrac) in enumerate(steps):
            xnew = x+stepfrac*fullstep
            newfval = func(xnew).data
            actual_improve = fval-newfval
            expected_improve = expected_improve_rate*stepfrac
            ratio = actual_improve/expected_improve
            if ratio.item() > accept_ratio and actual_improve.item() > 0:
                return True, xnew
        return False, x

    def get_flat_params_from(self, model):
        """
        Get flattened parameters from a network. Returns a single-column vector of network weights.
        """
        params = []
        for param in model.parameters():
            params.append(param.data.view(-1))
        flat_params = torch.cat(params)
        return flat_params

    def set_flat_params_to(self, model, flat_params):
        """
        Take a single-column vector of network weights, and manually set the weights of a given network
        to those contained in the vector.
        """
        prev_ind = 0
        for param in model.parameters():
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(flat_params[prev_ind:prev_ind+flat_size].view(param.size()))
            prev_ind += flat_size

    def hard_update(self, target, source):
        """
        Updates a target network based on a source network. I.e. it makes N* == N for two networks
        N* and N.
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def select_action(self, x):
        """
        Actions are taken under beta. Beta is the same as pi when doing trajectory rollouts,
        but when we optimize, we keep beta fixed, and maximize the advantage of pi over beta.
        From there, we set params of beta to be the same as those of pi.
        """
        
        mus = []
        sigmas = []
        for _, s in enumerate(self.skills):
            s_mu, s_logvar = s.beta(x)
            mus.append(s_mu.detach())
            sigmas.append(s_logvar.exp().sqrt().detach())
        state = torch.cat([x]+[Tensor(mu) for mu in mus], dim=-1)
        mu, logvar, phi = self.beta(state)
        #print(logvar.exp().sqrt())
        dist = Normal(mu, logvar.exp().sqrt())
        a = dist.sample()
        candidate_actions = mus+[a]
        mus.append(mu)
        sigmas.append(logvar.exp().sqrt())
        action = 0.
        for i, a in enumerate(candidate_actions):
            action += phi[i]*a
        prob = 0.
        for i, (m, s) in enumerate(zip(mus, sigmas)):
            dist = Normal(m, s)
            norm_prob = torch.exp(dist.log_prob(action))
            prob += phi[i]*norm_prob
        log_prob = torch.log(prob)
        return action, log_prob, state

    def update(self, crit_opt, trajectory):
        def policy_loss(params=None):
            """
            Compute the loss of the current policy relative to the old policy. See
            Schulman, 2015, Eqns. 1-4, 12-14.
            """
            def get_loss():
                mus = []
                sigmas = []
                for _, s in enumerate(self.skills):
                    s_mu, s_logvar = s.beta(states)
                    mus.append(s_mu.detach())
                    sigmas.append(s_logvar.exp().sqrt().detach())
                augmented_states = torch.cat([states]+[Tensor(mu) for mu in mus], dim=-1)
                mu_pi, logvar_pi, phi = self.pi(augmented_states)
                mus.append(mu_pi)
                sigmas.append(logvar_pi.exp().sqrt())
            
                probs = 0.
                for i, (m, s) in enumerate(zip(mus, sigmas)):
                    dist = Normal(m, s)
                    norm_probs = torch.exp(dist.log_prob(actions))
                    probs += phi[:,i].unsqueeze(1)*norm_probs
                pi_log_probs = torch.log(probs)

                ratio = (pi_log_probs-fixed_log_probs.detach()).sum(dim=1, keepdim=True).exp()
                action_loss = -ratio*advantages
                return action_loss.mean()
            
            if params is None:
                return get_loss()
            else:
                self.set_flat_params_to(self.pi, params)
                return get_loss()
        
        def fvp(vec):
            """
            Compute mean Fisher Vector Product (Schulman, 2015; see Appendix C.1). Returns the vector
            product Fv = g. To do this, we compute:

                grad_{kl} pi_{theta}
                grad_{kl} (grad_{kl} pi_{theta} * v)
            
            Which gives us F*v
            """

            mus = []
            sigmas = []
            for _, s in enumerate(self.skills):
                s_mu, s_logvar = s.beta(states)
                mus.append(s_mu.detach())
                sigmas.append(s_logvar.exp().sqrt().detach())
            augmented_states = torch.cat([states]+[Tensor(mu) for mu in mus], dim=-1)
            mu_pi, logvar_pi, phi = self.pi(augmented_states)
            mus.append(mu_pi)
            sigmas.append(logvar_pi.exp().sqrt())
            
            probs = 0.
            for i, (m, s) in enumerate(zip(mus, sigmas)):
                dist = Normal(m, s)
                norm_probs = torch.exp(dist.log_prob(actions))
                probs += phi[:,i].unsqueeze(1)*norm_probs
            pi_log_probs = torch.log(probs)

            ratio = (pi_log_probs-fixed_log_probs.detach()).exp()
            kl = (ratio.detach()*torch.log(ratio)).mean(dim=0, keepdim=True)
            #kl = pi_log_probs.exp()*ratio.log()
            kl = kl.sum()
            grads = torch.autograd.grad(kl, self.pi.parameters(), create_graph=True)
            flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
            kl_v = flat_grad_kl.dot(vec)
            grads = torch.autograd.grad(kl_v, self.pi.parameters())
            flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data
            return flat_grad_grad_kl+vec*self.__damping

        # get trajectory batch data
        rewards = torch.stack(trajectory["rewards"])
        masks = torch.stack(trajectory["masks"])
        actions = torch.stack(trajectory["actions"])
        fixed_log_probs = torch.stack(trajectory["log_probs"])
        states = torch.stack(trajectory["states"])
        actions = torch.stack(trajectory["actions"])
        #next_states = torch.stack(trajectory["next_states"])
        augmented_states = torch.stack(trajectory["augmented_states"])

        # calculate empirical advantage using trajectory rollouts
        values = self.critic(augmented_states)
        returns = self.__Tensor(actions.size(0),1)
        deltas = self.__Tensor(actions.size(0),1)
        advantages = self.__Tensor(actions.size(0),1)
        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(rewards.size(0))):
            #if masks[i] == 0:
            #    next_val = self.critic(next_states[i,:])
            #    prev_return = next_val
            #    prev_value = next_val
            returns[i] = rewards[i]+self.__gamma*prev_return*masks[i]
            deltas[i] = rewards[i]+self.__gamma*prev_value*masks[i]-values.data[i]
            advantages[i] = deltas[i]+self.__gamma*self.__tau*prev_advantage*masks[i]
            prev_return = returns[i, 0]
            prev_value = values.data[i, 0]
            prev_advantage = advantages[i, 0]
        returns = (returns-returns.mean())/(returns.std()+1e-10)
        advantages = (advantages-advantages.mean())/(advantages.std()+1e-10)
        
        # update critic using Adam
        crit_opt.zero_grad()
        crit_loss = F.smooth_l1_loss(values, returns.detach())
        crit_loss.backward(retain_graph=True)
        crit_opt.step()
        
        # trust region policy update. We update pi by maximizing it's advantage over beta,
        # and then set beta to the policy parameters of pi.
        pol_loss = policy_loss()
        grads = torch.autograd.grad(pol_loss, self.pi.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        stepdir = self.conjugate_gradient(fvp, -loss_grad)
        shs = 0.5*(stepdir.dot(fvp(stepdir)))
        lm = torch.sqrt(self.__max_kl/shs)
        fullstep = stepdir*lm
        expected_improve = -loss_grad.dot(fullstep)
        old_params = self.get_flat_params_from(self.pi)
        _, params = self.linesearch(self.pi, policy_loss, old_params, fullstep, expected_improve)
        self.set_flat_params_to(self.pi, params)
        self.hard_update(self.beta, self.pi)
    
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
log_interval = 5
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
batch_size = 2048

for _ in range(5):
    pi = Actor(state_dim+int(len(skills)*action_dim), 128, action_dim, int(len(skills)+1))
    beta = Actor(state_dim+int(len(skills)*action_dim), 128, action_dim, int(len(skills)+1))
    critic = Critic(state_dim+int(len(skills)*action_dim), 128, 1)
    agent  = MDN(pi, beta, critic, skills, cfg.trpo["network_settings"], GPU=True)
    agent = agent.cuda()
    crit_opt = torch.optim.Adam(critic.parameters(),lr=1e-5)
    count = 1
    interval_avg = []
    avg = 0
    for ep in range(1, iterations+1):
        s_, a_, ns_, r_, lp_, masks, as_ = [], [], [], [], [], [], []
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
                action, log_prob, augmented_state = agent.select_action(state)
                next_state, reward, done, _ = env.step(action.cpu().data.numpy())
                reward_sum += reward
                next_state = Tensor(next_state)
                reward = Tensor([reward])
                s_.append(state)
                ns_.append(next_state)
                a_.append(action)
                r_.append(reward)
                lp_.append(log_prob)
                masks.append(Tensor([not done]))
                as_.append(augmented_state)
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
                    "log_probs": lp_,
                    "augmented_states": as_
                    }

        agent.update(crit_opt, trajectory)
        if ep % log_interval == 0:
            interval = float(sum(interval_avg))/float(len(interval_avg))
            print('Episode {}\t Interval average: {:.3f}\t Average reward: {:.3f}'.format(ep, interval, avg))
            interval_avg = []