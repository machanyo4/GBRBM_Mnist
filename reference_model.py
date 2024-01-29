import torch
import torch.nn as nn
import torch.nn.functional as F

class GBRBM(nn.Module):
    def __init__(self, visible_size, hidden_size, CD_step=1, CD_burnin=0, init_var=1e-0):
        super(GBRBM, self).__init__()

        self.visible_size = visible_size
        self.hidden_size = hidden_size
        self.CD_step = CD_step
        self.CD_burnin = CD_burnin
        self.init_var = init_var

        self.W = nn.Parameter(torch.Tensor(visible_size, hidden_size))
        self.b = nn.Parameter(torch.Tensor(hidden_size))
        self.mu = nn.Parameter(torch.Tensor(visible_size))
        self.log_var = nn.Parameter(torch.Tensor(visible_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.W, std=1.0 * self.init_var / torch.sqrt(self.visible_size + self.hidden_size))
        nn.init.constant_(self.b, 0.0)
        nn.init.constant_(self.mu, 0.0)
        nn.init.constant_(self.log_var, torch.log(torch.Tensor([self.init_var])))

    def get_var(self):
        return self.log_var.exp().clamp(min=1e-8)

    def energy(self, v, h):
        var = self.get_var()
        eng = 0.5 * ((v - self.mu)**2 / var).sum(dim=1)
        eng -= ((v / var).mm(self.W) * h).sum(dim=1) + h.mv(self.b)
        return eng.mean()

    def energy_grad_param(self, v, h):
        var = self.get_var()
        grad = {}
        grad['W'] = -torch.einsum("bi,bj->ij", v / var, h) / v.shape[0]
        grad['b'] = -h.mean(dim=0)
        grad['mu'] = ((self.mu - v) / var).mean(dim=0)
        grad['log_var'] = (-0.5 * (v - self.mu)**2 / var + ((v / var) * h.mm(self.W.T))).mean(dim=0)
        return grad

    def Gibbs_sampling_vh(self, v, num_steps=1, burn_in=0):
        samples = []
        var = self.get_var()
        h = torch.bernoulli(self.prob_h_given_v(v, var))
        for _ in range(num_steps):
            mu = self.prob_v_given_h(h)
            v = mu + torch.randn_like(mu) * var.sqrt()
            h = torch.bernoulli(self.prob_h_given_v(v, var))
            samples.append((v, h))
        return samples[burn_in:]

    def prob_h_given_v(self, v, var):
        return torch.sigmoid((v / var).mm(self.W) + self.b)

    def prob_v_given_h(self, h):
        return h.mm(self.W.T) + self.mu

    def CD_grad(self, v):
        v = v.view(v.shape[0], -1)
        # positive gradient
        pos_samples = self.Gibbs_sampling_vh(v, num_steps=self.CD_step, burn_in=self.CD_burnin)
        grad_pos = self.energy_grad_param(pos_samples[-1][0], pos_samples[-1][1])

        # negative gradient
        v_neg = torch.randn_like(v)
        neg_samples = self.Gibbs_sampling_vh(v_neg, num_steps=self.CD_step, burn_in=self.CD_burnin)
        grad_neg = self.energy_grad_param(neg_samples[-1][0], neg_samples[-1][1])

        # compute update
        for name, param in self.named_parameters():
            param.grad = grad_pos[name] - grad_neg[name]