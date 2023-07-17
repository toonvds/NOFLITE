# Taken from FCCN authors
import sys
import numpy as np # linear algebra
from scipy.stats import randint
import matplotlib.pyplot as plt # this is used for the plot the graph
from tqdm import notebook

import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions import bernoulli, normal
import torch.distributions
from collections import defaultdict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init_qz(qz, pz, y, t, x):
    """
    Initialize qz towards outputting standard normal distributions
    - with standard torch init of weights the gradients tend to explode after first update step
    """
    idx = list(range(x.shape[0]))
    np.random.shuffle(idx)

    optimizer = optim.Adam(qz.parameters(), lr=0.001)

    for i in range(50):
        batch = np.random.choice(idx, 1)
        x_train, y_train, t_train = (x[batch]), (y[batch]), (t[batch])
        xy = torch.cat((x_train, y_train), 1)

        z_infer = qz(xy=xy, t=t_train)

        # KL(q_z|p_z) mean approx, to be minimized
        # KLqp = (z_infer.log_prob(z_infer.mean) - pz.log_prob(z_infer.mean)).sum(1)
        # Analytic KL
        KLqp = (-torch.log(z_infer.stddev) + 1 / 2 * (z_infer.variance + z_infer.mean ** 2 - 1)).sum(1)

        objective = KLqp
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

        if KLqp != KLqp:
            raise ValueError('KL(pz,qz) contains NaN during init')

    return qz


class p_x_z(nn.Module):

    def __init__(self, dim_in=20, nh=3, dim_h=20, dim_out_bin=19, dim_out_con=6):
        super().__init__()
        # save required vars
        self.nh = nh
        self.dim_out_bin = dim_out_bin
        self.dim_out_con = dim_out_con

        # dim_in is dim of latent space z
        self.input = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh-1)])
        # output layer defined separate for continuous and binary outputs
        self.output_bin = nn.Linear(dim_h, dim_out_bin)
        # for each output an mu and sigma are estimated
        self.output_con_mu = nn.Linear(dim_h, dim_out_con)
        self.output_con_sigma = nn.Linear(dim_h, dim_out_con)
        self.softplus = nn.Softplus()

    def forward(self, z_input):
        z = F.elu(self.input(z_input))
        for i in range(self.nh-1):
            z = F.elu(self.hidden[i](z))
        # for binary outputs:
        x_bin_p = torch.sigmoid(self.output_bin(z))
        x_bin = bernoulli.Bernoulli(x_bin_p)
        # for continuous outputs
        mu, sigma = self.output_con_mu(z), self.softplus(self.output_con_sigma(z))
        x_con = normal.Normal(mu, sigma)

        if (z != z).all():
            raise ValueError('p(x|z) forward contains NaN')

        return  x_con,x_bin


class p_t_z(nn.Module):

    def __init__(self, dim_in=20, nh=1, dim_h=20, dim_out=1):
        super().__init__()
        # save required vars
        self.nh = nh
        self.dim_out = dim_out

        # dim_in is dim of latent space z
        self.input = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        self.output = nn.Linear(dim_h, dim_out)

    def forward(self, x):
        x = F.elu(self.input(x))
        for i in range(self.nh):
            x = F.elu(self.hidden[i](x))
        # for binary outputs:
        out_p = torch.sigmoid(self.output(x))

        out = bernoulli.Bernoulli(out_p)
        return out


class p_y_zt(nn.Module):

    def __init__(self, dim_in=20, nh=3, dim_h=20, dim_out=1):
        super().__init__()
        # save required vars
        self.nh = nh
        self.dim_out = dim_out

        # Separated forwards for different t values, TAR

        self.input_t0 = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden_t0 = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        self.mu_t0 = nn.Linear(dim_h, dim_out)

        self.input_t1 = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden_t1 = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        self.mu_t1 = nn.Linear(dim_h, dim_out)

    def forward(self, z, t):
        # Separated forwards for different t values, TAR

        x_t0 = F.elu(self.input_t0(z))
        for i in range(self.nh):
            x_t0 = F.elu(self.hidden_t0[i](x_t0))
        mu_t0 = F.elu(self.mu_t0(x_t0))

        x_t1 = F.elu(self.input_t1(z))
        for i in range(self.nh):
            x_t1 = F.elu(self.hidden_t1[i](x_t1))
        mu_t1 = F.elu(self.mu_t1(x_t1))
        # set mu according to t value
        y = normal.Normal((1-t)*mu_t0 + t * mu_t1, 1)

        return y


####### Inference model / Encoder #######
class q_t_x(nn.Module):

    def __init__(self, dim_in=25, nh=1, dim_h=20, dim_out=1):
        super().__init__()
        # save required vars
        self.nh = nh
        self.dim_out = dim_out

        # dim_in is dim of data x
        self.input = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        self.output = nn.Linear(dim_h, dim_out)

    def forward(self, x):
        x = F.elu(self.input(x))
        for i in range(self.nh):
            x = F.elu(self.hidden[i](x))
        # for binary outputs:
        out_p = torch.sigmoid(self.output(x))
        out = bernoulli.Bernoulli(out_p)

        return out


class q_y_xt(nn.Module):

    def __init__(self, dim_in=25, nh=3, dim_h=20, dim_out=1):
        super().__init__()
        # save required vars
        self.nh = nh
        self.dim_out = dim_out

        # dim_in is dim of data x
        self.input = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        # separate outputs for different values of t
        self.mu_t0 = nn.Linear(dim_h, dim_out)
        self.mu_t1 = nn.Linear(dim_h, dim_out)

    def forward(self, x, t):
        # Unlike model network, shared parameters with separated heads
        x = F.elu(self.input(x))
        for i in range(self.nh):
            x = F.elu(self.hidden[i](x))
        # only output weights separated
        mu_t0 = self.mu_t0(x)
        mu_t1 = self.mu_t1(x)
        # set mu according to t, sigma set to 1
        y = normal.Normal((1-t)*mu_t0 + t * mu_t1, 1)
        return y


class q_z_tyx(nn.Module):

    def __init__(self, dim_in=25+1, nh=3, dim_h=20, dim_out=20):
        super().__init__()
        # dim in is dim of x + dim of y
        # dim_out is dim of latent space z
        # save required vars
        self.nh = nh

        # Shared layers with separated output layers

        self.input = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])

        self.mu_t0 = nn.Linear(dim_h, dim_out)
        self.mu_t1 = nn.Linear(dim_h, dim_out)
        self.sigma_t0 = nn.Linear(dim_h, dim_out)
        self.sigma_t1 = nn.Linear(dim_h, dim_out)
        self.softplus = nn.Softplus()

    def forward(self, xy, t):
        # Shared layers with separated output layers
        # print('before first linear z_infer')
        # print(xy)
        x = F.elu(self.input(xy))
        # print('first linear z_infer')
        # print(x)
        for i in range(self.nh):
            x = F.elu(self.hidden[i](x))

        mu_t0 = self.mu_t0(x)
        mu_t1 = self.mu_t1(x)
        sigma_t0 = self.softplus(self.sigma_t0(x))
        sigma_t1 = self.softplus(self.sigma_t1(x))

        # Set mu and sigma according to t
        z = normal.Normal((1-t)*mu_t0 + t * mu_t1, (1-t)*sigma_t0 + t * sigma_t1)
        return z


def cevae(xtrain, trttrain, ytrain, xtest, trttest, dim_bin, dim_cont,
          lr=1e-4, decay=1e-4, batch_size=100, iters=7000, n_h=64, n_samples=500):
    ym, ys = torch.mean(ytrain), torch.std(ytrain)
    ytr = (ytrain - ym) / ys

    # init networks (overwritten per replication)
    p_x_z_dist = p_x_z(dim_in=20, nh=3, dim_h=n_h, dim_out_bin=dim_bin, dim_out_con=dim_cont).to(device)
    p_t_z_dist = p_t_z(dim_in=20, nh=1, dim_h=n_h, dim_out=1).to(device)
    p_y_zt_dist = p_y_zt(dim_in=20, nh=3, dim_h=n_h, dim_out=1).to(device)
    q_t_x_dist = q_t_x(dim_in=dim_bin + dim_cont, nh=1, dim_h=n_h, dim_out=1).to(device)

    # t is not feed into network, therefore not increasing input size (y is fed).
    q_y_xt_dist = q_y_xt(dim_in=dim_bin + dim_cont, nh=3, dim_h=n_h, dim_out=1).to(device)
    q_z_tyx_dist = q_z_tyx(dim_in=dim_bin + dim_cont + 1, nh=3, dim_h=n_h, dim_out=20).to(device)
    p_z_dist = normal.Normal(torch.zeros(20).to(device), torch.ones(20).to(device))

    # Create optimizer
    params = list(p_x_z_dist.parameters()) + \
             list(p_t_z_dist.parameters()) + \
             list(p_y_zt_dist.parameters()) + \
             list(q_t_x_dist.parameters()) + \
             list(q_y_xt_dist.parameters()) + \
             list(q_z_tyx_dist.parameters())

    # In paper Adamax is suggested
    optimizer = optim.Adamax(params, lr=lr, weight_decay=decay)

    # init q_z inference
    q_z_tyx_dist = init_qz(q_z_tyx_dist, p_z_dist, ytr, trttrain, xtrain)

    #####training################################################################
    loss = []
    for _ in range(iters):
        print(end="\r|%-10s|" % ("=" * int(10 * _ / (iters - 1))))

        i = np.random.choice(xtrain.shape[0], size=batch_size, replace=False)
        y_train = ytr[i, :]
        x_train = xtrain[i, :]
        trt_train = trttrain[i, :]

        # inferred distribution over z
        xy = torch.cat((x_train, y_train), 1)
        z_infer = q_z_tyx_dist(xy=xy, t=trt_train)
        # use a single sample to approximate expectation in lowerbound
        z_infer_sample = z_infer.sample()

        # RECONSTRUCTION LOSS
        # p(x|z)
        x_con, x_bin = p_x_z_dist(z_infer_sample)
        l1 = x_bin.log_prob(x_train[:, dim_cont:]).sum(1)

        l2 = x_con.log_prob(x_train[:, :dim_cont]).sum(1)

        # p(t|z)
        t = p_t_z_dist(z_infer_sample)
        l3 = t.log_prob(trt_train).squeeze()

        # p(y|t,z)
        # for training use trt_train, in out-of-sample prediction this becomes t_infer
        y = p_y_zt_dist(z_infer_sample, trt_train)
        l4 = y.log_prob(y_train).squeeze()

        # REGULARIZATION LOSS
        # p(z) - q(z|x,t,y)
        # approximate KL
        l5 = (p_z_dist.log_prob(z_infer_sample) - z_infer.log_prob(z_infer_sample)).sum(1)

        # AUXILIARY LOSS
        # q(t|x)
        t_infer = q_t_x_dist(x_train)
        l6 = t_infer.log_prob(trt_train).squeeze()

        # q(y|x,t)
        y_infer = q_y_xt_dist(x_train, trt_train)
        l7 = y_infer.log_prob(y_train).squeeze()

        # Total objective
        # inner sum to calculate loss per item, torch.mean over batch
        loss_mean = torch.mean(l1 + l2 + l3 + l4 + l5 + l6 + l7)
        loss.append(loss_mean.cpu().detach().numpy())
        objective = -loss_mean

        optimizer.zero_grad()
        # Calculate gradients
        objective.backward()
        # Update step
        optimizer.step()

    # evaluation
    out0 = []
    out1 = []

    t_infer = q_t_x_dist(xtest)

    for q in range(n_samples):
        ttmp = t_infer.sample()
        y_infer = q_y_xt_dist(xtest, ttmp)

        xy = torch.cat((xtest, y_infer.sample()), 1)
        z_infer = q_z_tyx_dist(xy=xy, t=trttest).sample()
        # Manually input zeros and ones
        y0 = p_y_zt_dist(z_infer, torch.zeros(z_infer.shape[0], 1).to(device)).sample()
        y1 = p_y_zt_dist(z_infer, torch.ones(z_infer.shape[0], 1).to(device)).sample()
        y0, y1 = y0 * ys + ym, y1 * ys + ym
        out0.append(y0.detach().cpu().numpy().ravel())
        out1.append(y1.detach().cpu().numpy().ravel())

    # the sample for the treated and control group
    out1 = np.array(out1)
    out0 = np.array(out0)

    return out1 - out0