import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import scipy.optimize
from math import pi, log


class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Actor, self).__init__()
        self.affine1 = nn.Linear(input_dim, hidden_dim)
        self.action_mean = nn.Linear(hidden_dim, output_dim)
        self.action_log_std = nn.Parameter(torch.zeros(1, output_dim))

        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.saved_actions = []
        self.rewards = []
        self.final_value = 0

    def forward(self, x):
        x = F.tanh(self.affine1(x))
        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        return action_mean, action_log_std, action_std

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Critic, self).__init__()
        self.affine1 = nn.Linear(input_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, output_dim)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.tanh(self.affine1(x))
        state_values = self.value_head(x)
        return state_values

class TRPO(nn.Module):
    def __init__(self, actor, critic):
        super(TRPO, self).__init__()
        self.actor = actor
        self.critic = critic

    def select_action(self, state):
        state = torch.from_numpy(state).unsqueeze(0)
        action_mean, _, action_std = self.actor(Variable(state))
        action = torch.normal(action_mean, action_std)
        return action

    def update(self, batch):
        
        # Original code uses the same LBFGS to optimize the value loss
        def get_value_loss(flat_params):
            set_flat_params_to(self.critic, torch.Tensor(flat_params))
            for param in self.critic.parameters():
                if param.grad is not None:
                    param.grad.data.fill_(0)
            values_ = self.critic(Variable(states))
            value_loss = (values_-targets).pow(2).mean()
            
            # weight decay
            for param in self.critic.parameters():
                value_loss += param.pow(2).sum()*args.l2_reg
            value_loss.backward()
            return (value_loss.data.double().numpy()[0], get_flat_grad_from(self.critic).data.double().numpy())

        def get_loss(volatile=False):
            action_means, action_log_stds, action_stds = policy_net(Variable(states, volatile=volatile))
            log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
            action_loss = -Variable(advantages)*torch.exp(log_prob-Variable(fixed_log_prob))
            return action_loss.mean()

        def get_kl():
            mean1, log_std1, std1 = self.actor(Variable(states))
            mean0 = Variable(mean1.data)
            log_std0 = Variable(log_std1.data)
            std0 = Variable(std1.data)
            kl = log_std1-log_std0+(std0.pow(2)+(mean0-mean1).pow(2))/(2.*std1.pow(2))-0.5
            return kl.sum(1, keepdim=True)

        rewards = torch.Tensor(batch.reward)
        masks = torch.Tensor(batch.mask)
        actions = torch.Tensor(np.concatenate(batch.action, 0))
        states = torch.Tensor(batch.state)
        values = self.critic(Variable(states))
        returns = torch.Tensor(actions.size(0),1)
        deltas = torch.Tensor(actions.size(0),1)
        advantages = torch.Tensor(actions.size(0),1)
        prev_return = 0
        prev_value = 0
        prev_advantage = 0

        # calculate advantages
        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i]+args.gamma*prev_return*masks[i]
            deltas[i] = rewards[i]+args.gamma*prev_value*masks[i]-values.data[i]
            advantages[i] = deltas[i]+args.gamma*args.tau*prev_advantage*masks[i]
            prev_return = returns[i, 0]
            prev_value = values.data[i, 0]
            prev_advantage = advantages[i, 0]
        
        targets = Variable(returns)
        flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(self.critic).double().numpy(), maxiter=25)
        set_flat_params_to(self.critic, torch.Tensor(flat_params))
        advantages = (advantages-advantages.mean())/advantages.std()
        action_means, action_log_stds, action_stds = policy_net(Variable(states))
        fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()
        trpo_step(self.actor, get_loss, get_kl, args.max_kl, args.damping)

def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr/torch.dot(p, _Avp)
        x += alpha*p
        r -= alpha*_Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr/rdotr
        p = r + betta*p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x

def linesearch(model, f, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=.1):
    fval = f(True).data
    print("fval before", fval[0])
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x+stepfrac*fullstep
        set_flat_params_to(model, xnew)
        newfval = f(True).data
        actual_improve = fval-newfval
        expected_improve = expected_improve_rate*stepfrac
        ratio = actual_improve / expected_improve
        print("a/e/r", actual_improve[0], expected_improve[0], ratio[0])
        if ratio[0] > accept_ratio and actual_improve[0] > 0:
            print("fval after", newfval[0])
            return True, xnew
    return False, x

def trpo_step(model, get_loss, get_kl, max_kl, damping):
    def Fvp(v):
        kl = get_kl()
        kl = kl.mean()
        grads = torch.autograd.grad(kl, model.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
        kl_v = (flat_grad_kl*Variable(v)).sum()
        grads = torch.autograd.grad(kl_v, model.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data
        return flat_grad_grad_kl + v * damping

    loss = get_loss()
    grads = torch.autograd.grad(loss, model.parameters())
    loss_grad = torch.cat([grad.view(-1) for grad in grads]).data
    stepdir = conjugate_gradients(Fvp, -loss_grad, 10)
    shs = 0.5 * (stepdir*Fvp(stepdir)).sum(0, keepdim=True)
    lm = torch.sqrt(shs/max_kl)
    fullstep = stepdir/lm[0]
    neggdotstepdir = (-loss_grad*stepdir).sum(0, keepdim=True)
    print(("lagrange multiplier:", lm[0], "grad_norm:", loss_grad.norm()))
    prev_params = get_flat_params_from(model)
    success, new_params = linesearch(model, get_loss, prev_params, fullstep, neggdotstepdir/lm[0])
    set_flat_params_to(model, new_params)
    return loss

def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5+0.5*torch.log(2*var*pi)
    return entropy.sum(1, keepdim=True)

def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x-mean).pow(2)/(2*var)-0.5*log(2*pi)-log_std
    return log_density.sum(1, keepdim=True)

def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    flat_params = torch.cat(params)
    return flat_params

def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(flat_params[prev_ind:prev_ind+flat_size].view(param.size()))
        prev_ind += flat_size

def get_flat_grad_from(net, grad_grad=False):
    grads = []
    for param in net.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            grads.append(param.grad.view(-1))
    flat_grad = torch.cat(grads)
    return flat_grad



running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)