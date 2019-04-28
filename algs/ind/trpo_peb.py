import torch
import torch.autograd as autograd
from torch.distributions import Normal, Categorical
import torch.nn as nn
import torch.nn.functional as F
import scipy.optimize
from math import pi, log
import numpy as np
from collections import namedtuple
import gym
import gym_aero
import utils
import numpy as np
import csv
import os

"""
    Clean implementation of Trust Region Policy Optimization (J. Schulman, 2015). This
    is a second-order method that solves the constrained problem:
    
        max Advantage_{beta}(pi)
        s.t. DKL (theta_beta||theta_pi) < eps
    
    Where Advantage_{beta} is the empirical Advantage estimate generated by rollouts
    under some initial policy beta, pi is a new policy that we are maximizing with respect
    to beta, and epsilon is some constant that we choose. DKL is the Kullback-Leibler
    divergence, and the goal is to minimize the KL distance of the two policies whilst ensuring
    that the policy still improves. The theory implies that this gives us a monotonic improvement
    guarantee (i.e. the policy will improve at each iteration, until convergence). We can 
    approximate the KL divergence as:

        KL ~ 0.5 * (theta' - theta) * F * (theta' - theta) 

    Where F is the Fisher information matrix (the negative Hessian of the log probability).
    We use the Conjugate Gradient method to solve for: 
    
        F * (theta' - theta) = g 
    
    Where is g is the standard policy gradient (using advantage). This is justified because:

        dtheta = (theta' - theta) ~= F^{-1} g 

    I.e. this is our update rule using the Hessian (where here F substitutes for the Hessian).
    We can calculate F using auto-differentiation through the network (using the mean KL distance
    between the two policies), and we can iteratively compute a vector dtheta. We can then use 
    this vector dtheta to solve for the maximum step size that we need to take, by assuming:

        max_KL ~ 0.5 * beta^{2} (theta' - theta) * F * (theta' - theta)
    
    Where max_KL is the maximum KL divergence allowed (i.e. our trust region). Since the CG
    algorithm assumes (by definition) that the function being solved is quadratic, we need 
    to do a line search to ensure that  our quadratic approximation is locally valid, and that
    our update does in fact improve the policy.

    Since theta' and theta are different, we need to use importance sampling to correct for the
    fact that the advantage is calculated using rollouts under theta, and not theta'.

    This implementation is fairly consistent with the implementation described in the paper. It 
    uses a first order update for the critic, unlike many other implementations which tend to use a 
    second order update (typically L-BFGS using the scipy.optimize library). I opted for a first 
    order critic update because it was easier to implement, cleaner, and the algorithm still performed 
    well.

    The algorithm is:

    repeat:
    1. set beta = pi.
    2. conduct trajectory rollouts under beta.
    3. compute empirical advantage estimate A_{beta}(s,a).
    4. find grad_{pi} A_{beta}(s,a). Remember, pi and beta are still the same here,
       so this is effectively the same as finding the standard policy gradient, and
       cuts down on code when we want to do off-policy correction using importance
       sampling.
    5. use the conjugate gradient method to find Fv = g, where F is the FIM, and v
       is our estimate of dtheta (starting with grad log(pi) A_{beta}(s,a)).
    6. find the quadratic approximation to (theta_beta||theta_pi) using dtheta^{T} F dtheta.
    7. set KL_max = 0.5 * beta * dtheta^{T} * F * beta * dtheta, and solve for beta
    8. conduct a linesearch to ensure that the quadratic approximation to KL holds. Since
       the policy parameters theta have changed, it's necessary to use an importance
       sampling estimator to get the correct advantage of the new policy over the old one.
    9. update the parameters of pi

    The reason I'm using two policies here is to make calculating the KL divergence
    a bit cleaner and easier to read. It's an extra step, but the clarity is worth it imo.
"""

class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Actor, self).__init__()
        self.__fc1 = nn.Linear(input_dim, hidden_dim)
        self.__fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.__mu = nn.Linear(hidden_dim, output_dim)
        self.__logvar = nn.Linear(hidden_dim, output_dim)
        self.__mu.weight.data.mul_(0.1)
        self.__mu.bias.data.mul_(0.)

    def forward(self, x):
        x = F.elu(self.__fc1(x.float()))
        x = F.elu(self.__fc2(x))
        mu = self.__mu(x)
        logvar = self.__logvar(x)
        return mu, logvar

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Critic, self).__init__()
        self.__fc1 = nn.Linear(input_dim, hidden_dim)
        self.__value = nn.Linear(hidden_dim, output_dim)
        self.__value.weight.data.mul_(0.1)
        self.__value.bias.data.mul_(0.)

    def forward(self, x):
        x = F.tanh(self.__fc1(x))
        state_values = self.__value(x)
        return state_values

class TRPO(nn.Module):
    def __init__(self, pi, beta, critic, params, GPU=False):
        super(TRPO, self).__init__()
        self.pi = pi
        self.beta = beta
        self.hard_update(self.beta, self.pi)
        self.critic = critic
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

    def select_action(self, state, deterministic=False):
        """
        Actions are taken under beta. Beta is the same as pi when doing trajectory rollouts,
        but when we optimize, we keep beta fixed, and maximize the advantage of pi over beta.
        From there, we set params of beta to be the same as those of pi.
        """
        mu, logvar = self.beta(state)
        if deterministic:
            return mu
        else:
            sigma = logvar.exp().sqrt()+1e-10
            dist = Normal(mu, sigma)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action, log_prob

    def update(self, crit_opt, trajectory):
        def policy_loss(params=None):
            """
            Compute the loss of the current policy relative to the old policy. See
            Schulman, 2015, Eqns. 1-4, 12-14.
            """
            def get_loss():
                mu_pi, logvar_pi = self.pi(states)
                sigma_pi = logvar_pi.exp().sqrt()+1e-10
                dist = Normal(mu_pi, sigma_pi)
                log_probs = dist.log_prob(actions)
                ratio = (log_probs-fixed_log_probs.detach()).sum(dim=1, keepdim=True).exp()
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
            mu_pi, logvar_pi = self.pi(states)
            mu_beta, logvar_beta = self.beta(states)
            var_pi = logvar_pi.exp()
            var_beta = logvar_beta.exp()
            kl = var_pi.sqrt().log()-var_beta.sqrt().log()+(var_beta+(mu_beta-mu_pi).pow(2))/(2.*var_pi)-0.5
            kl = torch.sum(kl, dim=1).mean()
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
        next_states = torch.stack(trajectory["next_states"])

        # calculate empirical advantage using trajectory rollouts
        values = self.critic(states)
        returns = self.__Tensor(actions.size(0),1)
        deltas = self.__Tensor(actions.size(0),1)
        advantages = self.__Tensor(actions.size(0),1)
        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(rewards.size(0))):
            if masks[i] == 0:
                next_val = self.critic(next_states[i,:])
                prev_return = next_val
                prev_value = next_val
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


class Trainer:
    def __init__(self, env_name, params, ident=1):
        self.__id = str(ident)
        self.__env = gym.make(env_name)
        self.__env_name = env_name
        self.__params = params
        self.__iterations = params["iterations"]
        self.__seed = params["seed"]
        self.__batch_size = params["batch_size"]
        self.__render = params["render"]
        self.__log_interval = params["log_interval"]
        self.__save = params["save"]
        cuda = params["cuda"]
        state_dim = self.__env.observation_space.shape[0]
        action_dim = self.__env.action_space.shape[0]
        hidden_dim = params["hidden_dim"]
        pi = Actor(state_dim, hidden_dim, action_dim)
        beta = Actor(state_dim, hidden_dim, action_dim)
        critic = Critic(state_dim, hidden_dim, 1)
        self.__agent = TRPO(pi, beta, critic, params["network_settings"], GPU=cuda)
        self.__crit_opt = torch.optim.Adam(critic.parameters())
        if cuda:
            self.__Tensor = torch.cuda.FloatTensor
            self.__agent = self.__agent.cuda()
        else:
            self.__Tensor = torch.Tensor
        self.__best = None

        # initialize experiment logging
        self.__logging = params["logging"]
        self.__directory = os.getcwd()
        if self.__logging:
            filename = self.__directory + "/data/trpo-"+self.__id+"-"+self.__env_name+".csv"
            with open(filename, "w") as csvfile:
                self.__writer = csv.writer(csvfile)
                self.__writer.writerow(["episode", "interval", "reward"])
                self.run_algo()
        else:
            self.run_algo()

    def run_algo(self):
        interval_avg = []
        avg = 0
        for ep in range(1, self.__iterations+1):
            s_, a_, ns_, r_, lp_, masks = [], [], [], [], [], []
            num_steps = 1
            reward_batch = 0
            num_episodes = 0
            action_mags = []
            while num_steps < self.__batch_size+1:
                state = self.__env.reset()
                state = self.__Tensor(state)
                reward_sum = 0
                t = 0
                done = False
                while not done:
                    if ep % self.__log_interval == 0 and self.__render:
                        self.__env.render()
                    action, log_prob = self.__agent.select_action(state)
                    next_state, reward, done, _ = self.__env.step(action.cpu().data.numpy())
                    reward_sum += reward
                    next_state = self.__Tensor(next_state)
                    reward = self.__Tensor([reward])
                    s_.append(state)
                    ns_.append(next_state)
                    a_.append(action)
                    r_.append(reward)
                    lp_.append(log_prob)

                    action_mags.append(action)

                    masks.append(self.__Tensor([not done]))
                    state = next_state
                    t += 1
                num_steps += t
                num_episodes += 1
                reward_batch += reward_sum
            reward_batch /= num_episodes
            interval_avg.append(reward_batch)
            avg = (avg*(ep-1)+reward_batch)/ep
            
            if (self.__best is None or reward_batch > self.__best) and self.__save:
                print("---Saving best TRPO policy---")
                self.__best = reward_batch
                fname = self.__directory + "/saved_policies/trpo-"+self.__id+"-"+self.__env_name+".pth.tar"
                utils.save(self.__agent, fname)
            
            trajectory = {
                        "states": s_,
                        "actions": a_,
                        "rewards": r_,
                        "next_states": ns_,
                        "masks": masks,
                        "log_probs": lp_
                        }
            self.__agent.update(self.__crit_opt, trajectory)
            if ep % self.__log_interval == 0:
                interval = float(sum(interval_avg))/float(len(interval_avg))
                print('Episode {}\t Interval average: {:.3f}\t Average reward: {:.3f}'.format(ep, interval, avg))
                interval_avg = []
                if self.__logging:
                    self.__writer.writerow([ep, interval, avg])
        fname = self.__directory + "/saved_policies/trpo-"+self.__id+"-"+self.__env_name+"-final.pth.tar"
        utils.save(self.__agent, fname)