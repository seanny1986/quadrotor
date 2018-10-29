import torch
import torch.autograd as autograd
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F
from math import pi, log, sqrt
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
        x = F.tanh(self.__fc1(x.float()))
        x = F.tanh(self.__fc2(x))
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
        self.__pi = pi
        self.__beta = beta
        self.hard_update(self.__beta, self.__pi)
        self.__critic = critic
        self.__gamma = params["gamma"]
        self.__tau = params["tau"]
        self.__max_kl = params["max_kl"]
        self.__damping = params["damping"]
        self.__GPU = GPU
        if GPU:
            self.__Tensor = torch.cuda.FloatTensor
            self.__pi = self.__pi.cuda()
            self.__beta = self.__beta.cuda()
            self.__critic = self.__critic.cuda()
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
        steps = 0.5**torch.arange(max_backtracks)
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

    def select_action(self, state):
        """
        Actions are taken under beta. Beta is the same as pi when doing trajectory rollouts,
        but when we optimize, we keep beta fixed, and maximize the advantage of pi over beta.
        From there, we set params of beta to be the same as those of pi.
        """
        mu, logvar = self.__beta(state)
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
                mu_pi, logvar_pi = self.__pi(states)
                sigma_pi = logvar_pi.exp().sqrt()+1e-10
                dist = Normal(mu_pi, sigma_pi)
                log_probs = dist.log_prob(actions)
                ratio = (log_probs-fixed_log_probs.detach()).sum(dim=1, keepdim=True).exp()
                action_loss = -ratio*advantages
                return action_loss.mean()
            if params is None:
                return get_loss()
            else:
                self.set_flat_params_to(self.__pi, params)
                return get_loss()
        
        def fvp(vec):
            """
            Compute mean Fisher Vector Product (Schulman, 2015; see Appendix C.1). Returns the vector
            product Fv = g. To do this, we compute:

                grad_{kl} pi_{theta}
                grad_{kl} (grad_{kl} pi_{theta} * v)
            
            Which gives us F*v
            """
            mu_pi, logvar_pi = self.__pi(states)
            mu_beta, logvar_beta = self.__beta(states)
            var_pi = logvar_pi.exp()
            var_beta = logvar_beta.exp()
            kl = var_pi.sqrt().log()-var_beta.sqrt().log()+(var_beta+(mu_beta-mu_pi).pow(2))/(2.*var_pi)-0.5
            kl = torch.sum(kl, dim=1).mean()
            grads = torch.autograd.grad(kl, self.__pi.parameters(), create_graph=True)
            flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
            kl_v = flat_grad_kl.dot(vec)
            grads = torch.autograd.grad(kl_v, self.__pi.parameters())
            flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data
            return flat_grad_grad_kl+vec*self.__damping
        
        def closure():
            crit_opt.zero_grad()
            values = self.__critic(states)
            returns = self.__Tensor(actions.size(0),1)
            prev_return = 0
            for i in reversed(range(rewards.size(0))):
                returns[i] = rewards[i]+self.__gamma*prev_return*masks[i]
                prev_return = returns[i, 0]
            returns = (returns-returns.mean())/(returns.std()+1e-10)
            crit_loss = F.smooth_l1_loss(values, returns)
            crit_loss.backward()
            return crit_loss

        # get trajectory batch data
        rewards = torch.stack(trajectory["rewards"])
        masks = torch.stack(trajectory["masks"])
        actions = torch.stack(trajectory["actions"])
        fixed_log_probs = torch.stack(trajectory["log_probs"])
        states = torch.stack(trajectory["states"])

        # hacky line search for second order update of the critic
        old_params = self.get_flat_params_from(self.__critic)
        curr_loss = closure()
        for lr in self.lr * .5**np.arange(10):
            self.optimizer = torch.optim.LBFGS(self.__critic.parameters(), lr=lr)
            self.optimizer.step(closure)
            current_params = self.get_flat_params_from(self.__critic)
            loss = closure()
            if loss < curr_loss:
                curr_loss = loss
                old_params = current_params
            if any(np.isnan(current_params.data.cpu().numpy())):
                print("LBFGS optimization diverged. Rolling back update...")
                self.set_flat_params_to(self.__critic, old_params)
        self.set_flat_params_to(self.__critic, old_params)

        # calculate empirical advantage using trajectory rollouts
        values = self.__critic(states)
        returns = self.__Tensor(actions.size(0),1)
        deltas = self.__Tensor(actions.size(0),1)
        advantages = self.__Tensor(actions.size(0),1)
        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i]+self.__gamma*prev_return*masks[i]
            deltas[i] = rewards[i]+self.__gamma*prev_value*masks[i]-values.data[i]
            advantages[i] = deltas[i]+self.__gamma*self.__tau*prev_advantage*masks[i]
            prev_return = returns[i, 0]
            prev_value = values.data[i, 0]
            prev_advantage = advantages[i, 0]
        returns = (returns-returns.mean())/(returns.std()+1e-10)
        advantages = (advantages-advantages.mean())/(advantages.std()+1e-10)
        
        # trust region policy update. We update pi by maximizing it's advantage over beta,
        # and then set beta to the policy parameters of pi.
        pol_loss = policy_loss()
        grads = torch.autograd.grad(pol_loss, self.__pi.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        stepdir = self.conjugate_gradient(fvp, -loss_grad)
        shs = 0.5*(stepdir.dot(fvp(stepdir)))
        lm = torch.sqrt(self.__max_kl/shs)
        fullstep = stepdir*lm
        expected_improve = -loss_grad.dot(fullstep)
        old_params = self.get_flat_params_from(self.__pi)
        _, params = self.linesearch(self.__pi, policy_loss, old_params, fullstep, expected_improve)
        self.set_flat_params_to(self.__pi, params)
        self.hard_update(self.__beta, self.__pi)

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
        planner_state_dim = self.__env.planner_observation_space.shape[0]
        planner_action_dim = self.__env.planner_action_space.shape[0]
        hidden_dim = params["hidden_dim"]

        # low level policy
        pi = Actor(state_dim, hidden_dim, action_dim)
        beta = Actor(state_dim, hidden_dim, action_dim)
        critic = Critic(state_dim, hidden_dim, 1)
        self.__agent = TRPO(pi, beta, critic, params["network_settings"], GPU=cuda)
        self.__crit_opt = LBFGS(critic.parameters(), lr=1, history_size=10, line_search='Wolfe', debug=True)

        # high level policy
        pi_pl = Actor(planner_state_dim, hidden_dim, planner_action_dim)
        beta_pl = Actor(planner_state_dim, hidden_dim, planner_action_dim)
        critic_pl = Critic(planner_state_dim, hidden_dim, 1)
        self.__agent_pl = TRPO(pi_pl, beta_pl, critic_pl, params["network_settings"], GPU=cuda)
        self.__crit_opt_pl = LBFGS(critic_pl.parameters(), lr=1, history_size=10, line_search='Wolfe', debug=True)

        if cuda:
            self.__Tensor = torch.cuda.FloatTensor
            self.__agent = self.__agent.cuda()
            self.__agent_pl = self.__agent_pl.cuda()
        else:
            self.__Tensor = torch.Tensor
        self.__best = None
        self.__best_planner = None

        # initialize experiment logging
        self.__logging = params["logging"]
        self.__directory = os.getcwd()
        if self.__logging:
            filename = self.__directory + "/data/trpo-h-"+self.__id+"-"+"-"+self.__env_name+".csv"
            with open(filename, "w") as csvfile:
                self.__writer = csv.writer(csvfile)
                self.__writer.writerow(["episode", "interval", "reward", "planner_interval", "planner_reward"])
                self.run_algo()
        else:
            self.run_algo()

    def run_planner(self, wp_state, render=True):
        _wp_state, _wp, _wp_lp, _wp_rew, _wp_masks = [], [], [], [], []
        wp_state = self.__Tensor(wp_state)
        wp_reward_sum = 0         
        done = False
        while not done:
            wp, wp_lp  = self.__agent_pl.select_action(wp_state)
            next_wp_state, wp_rew, done, _ = self.__env.planner_step(wp.cpu().numpy())
            wp_reward_sum += wp_rew
            self.__env.add_waypoint(np.array(next_wp_state)[0:3].reshape((3,1)))
            next_wp_state = self.__Tensor(next_wp_state)
            wp_rew = self.__Tensor([wp_rew])
            _wp_state.append(wp_state)
            _wp.append(wp)
            _wp_lp.append(wp_lp)
            _wp_rew.append(wp_rew)
            _wp_masks.append(self.__Tensor([not done]))
            wp_state = next_wp_state
        return wp_reward_sum, {"states": _wp_state,
                                "actions": _wp,
                                "log_probs": _wp_lp,
                                "masks": _wp_masks,
                                "rewards": _wp_rew}

    def run_algo(self):
        print("----STARTING TRAINING ALGORITHM---")
        interval_avg = []
        wp_interval_avg = []
        avg = 0
        wp_avg = 0
        for ep in range(1, self.__iterations+1):
            s_, a_, r_, lp_, masks = [], [], [], [], []
            wp_s_, wp_a_, wp_r_, wp_lp_, wp_masks = [], [], [], [], []
            num_steps = 1
            reward_batch = 0
            wp_reward_batch = 0
            num_episodes = 0
            while num_steps < self.__batch_size+1:
                planner_state, aircraft_state = self.__env.planner_reset()
                wp_reward_sum, wp_trajectory = self.run_planner(planner_state, ep % self.__log_interval == 0 and self.__render)
                wp_s_.extend(wp_trajectory["states"])
                wp_a_.extend(wp_trajectory["actions"])
                wp_r_.extend(wp_trajectory["rewards"])
                wp_lp_.extend(wp_trajectory["log_probs"])
                wp_masks.extend(wp_trajectory["masks"])
                self.__env.init_waypoints()
                policy_state = self.__env.policy_reset(aircraft_state)
                policy_state = self.__Tensor(policy_state)
                reward_sum = 0
                t = 0
                if ep % self.__log_interval == 0 and self.__render:
                    self.__env.render()
                done = False
                while not done:
                    action, log_prob = self.__agent.select_action(policy_state)
                    next_state, reward, done, _ = self.__env.policy_step(action.cpu().data.numpy())
                    reward_sum += reward
                    next_state = self.__Tensor(next_state)
                    reward = self.__Tensor([reward])
                    s_.append(policy_state)
                    a_.append(action)
                    r_.append(reward)
                    lp_.append(log_prob)
                    masks.append(self.__Tensor([not done]))
                    t += 1
                    if ep % self.__log_interval == 0 and self.__render:
                        self.__env.render()
                    if done:
                        break
                    policy_state = next_state
                num_steps += (t-1)
                num_episodes += 1
                reward_batch += reward_sum
                wp_reward_batch += wp_reward_sum
            reward_batch /= num_episodes
            wp_reward_batch /= num_episodes
            interval_avg.append(reward_batch)
            wp_interval_avg.append(wp_reward_batch)
            avg = (avg*(ep-1)+reward_batch)/ep
            wp_avg = (wp_avg*(ep-1)+wp_reward_batch)/ep

            if (self.__best is None or reward_batch > self.__best) and self.__save:
                print("---Saving best TRPO-H policy---")
                self.__best = reward_batch
                fname = self.__directory + "/saved_policies/trpo-h-"+self.__id+"-"+"-"+self.__env_name+".pth.tar"
                utils.save(self.__agent, fname)
            if (self.__best_planner is None or wp_reward_batch > self.__best_planner) and self.__save:
                print("---Saving best TRPO-H planner---")
                self.__best_planner = wp_reward_batch
                fname = self.__directory + "/saved_policies/trpo-h-"+"planner-"+self.__id+"-"+"-"+self.__env_name+".pth.tar"
                utils.save(self.__agent_pl, fname)
            
            print("Updating planner")
            wp_trajectory = {
                        "states": wp_s_,
                        "actions": wp_a_,
                        "rewards": wp_r_,
                        "masks": wp_masks,
                        "log_probs": wp_lp_
                        }
            self.__agent_pl.update(self.__crit_opt_pl, wp_trajectory)

            print("Updating policy")
            trajectory = {
                        "states": s_,
                        "actions": a_,
                        "rewards": r_,
                        "masks": masks,
                        "log_probs": lp_
                        }

            self.__agent.update(self.__crit_opt, trajectory)
            if ep % self.__log_interval == 0:
                interval = float(sum(interval_avg))/float(len(interval_avg))
                wp_interval = float(sum(wp_interval_avg))/float(len(wp_interval_avg))
                print("Planner Return:")
                print("Episode {}:".format(ep)) 
                print("Planner Interval average: {:.3f}\t Planner Average reward: {:.3f}".format(wp_interval, wp_avg))
                print("Policy Interval average: {:.3f}\t Policy Average reward: {:.3f}".format(interval, avg))
                interval_avg = []
                wp_interval_avg = []
                if self.__logging:
                    self.__writer.writerow([ep, interval, avg, wp_interval, wp_avg])
        fname = self.__directory + "/saved_policies/trpo-h-"+self.__id+"-"+"-"+self.__env_name+"-final.pth.tar"
        utils.save(self.__agent, fname)