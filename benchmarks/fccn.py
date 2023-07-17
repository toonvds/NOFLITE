# Based on: https://github.com/thuizhou/Collaborating-Networks

import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
import torch
from torch import optim, nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class cn_g(nn.Module):
    def __init__(self, hidden_size=25):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.k1 = 100
        self.k2 = 80

        self.fc1 = nn.Linear(hidden_size*2+2, self.k1)
        self.fc2 = nn.Linear(self.k1, self.k2)
        self.fc3 = nn.Linear(self.k2, 1)

    def forward(self, y, x):
        data = torch.cat([y, x], dim=1)
        h1 = self.fc1(data)
        h1 = F.elu(h1)
        h2 = self.fc2(h1)
        h2 = F.elu(h2)
        h3 = self.fc3(h2)
        g_logit = h3
        return g_logit


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight, 0.0, .001)


#critic for latent representation
class critic(nn.Module):
    def __init__(self, hidden_size=25):
        super().__init__()
        self.k1 = 50
        self.fc1 = nn.Linear(hidden_size, self.k1)
        self.fc2 = nn.Linear(self.k1, 1)

    def forward(self, s):
        h1 = F.elu(self.fc1(s))
        critic_out = self.fc2(h1)
        return critic_out


# generator for latent representation
class gen(nn.Module):
    def __init__(self, input_size, hidden_size=25):
        super().__init__()

        self.k1 = 100

        self.fc1 = nn.Linear(input_size, self.k1)
        self.fc2 = nn.Linear(self.k1, hidden_size)

    def forward(self, x):
        h1 = F.elu(self.fc1(x))
        gen_out=self.fc2(h1)
        return gen_out


# predictor for propensity model
class prop_pred(nn.Module):
    def __init__(self, hidden_size=25):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 1)

    def forward(self, s):
        prop_out = self.fc1(s)
        return prop_out


class FCCN(nn.Module):
    def __init__(self, input_size, hidden_size=25, alpha=5e-4, beta=1e-5, EDU=False):
        super(FCCN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.alpha = alpha
        self.beta = beta
        self.EDU = EDU

        self.g0 = cn_g(hidden_size).to(self.device)
        self.g1 = cn_g(hidden_size).to(self.device)
        self.critic_ipm = critic(hidden_size).to(self.device)

        if not(self.EDU):
            self.g0.apply(weights_init)
            self.g1.apply(weights_init)

        # domain invariant and domain specific
        self.gen_lat_i = gen(input_size).to(self.device)
        self.gen_lat_s = gen(input_size).to(self.device)

        self.prop_est = prop_pred().to(self.device)

        self.gloss = nn.BCELoss()
        self.proploss = nn.BCELoss()
        self.poss_vals = 3000

    def train(self, xtrain, ytrain, trttrain, iters=20000, batch_size=128):

        xtrain = torch.from_numpy(xtrain).float()
        ytrain = torch.from_numpy(ytrain).float()
        trttrain = torch.from_numpy(trttrain.reshape(-1, 1)).float()

        gparams = list(self.g0.parameters()) + list(self.g1.parameters())
        optimizer_g = optim.Adam(gparams, lr=1e-4)
        if not self.EDU:
            optimizer_critic = optim.RMSprop(self.critic_ipm.parameters(), lr=1e-3)
        else:
            optimizer_critic = optim.RMSprop(self.critic_ipm.parameters(), lr=5e-4)
        optimizer_gen = optim.RMSprop(list(self.gen_lat_i.parameters()) + list(self.gen_lat_s.parameters()), lr=1e-4)
        optimizer_prop = optim.Adam(self.prop_est.parameters(), lr=1e-4)

        if not self.EDU:
            my_lr_scheduler_critic = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_critic, gamma=0.998)
            my_lr_scheduler_prop = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_prop, gamma=0.998)
            my_lr_scheduler_gen = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_gen, gamma=0.998)
        else:
            my_lr_scheduler_critic = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_critic, gamma=0.999)

        g_loss = []
        critic_loss = []
        gen_loss = []
        prop_loss = []

        n_critic = 5
        clip_value = torch.tensor(0.01, device=self.device)
        ipm_weight = torch.tensor(self.alpha, device=self.device)
        prop_weight = torch.tensor(self.beta, device=self.device)

        # Get ranges for prediction:
        self.y_poss = torch.linspace(ytrain.min() - 1, ytrain.max() + 1, self.poss_vals).reshape(-1, 1)
        self.y_possnp = np.linspace(ytrain.min() - 1, ytrain.max() + 1, self.poss_vals)

        for iter in range(iters):
            print(end="\r|%-10s|" % ("=" * int(10 * iter / (iters - 1))))

            # train g-network and latent representation ###########################
            i = np.random.choice(xtrain.shape[0], size=batch_size, replace=False)
            ys = ytrain[i, :].to(self.device)
            xs = xtrain[i, :].to(self.device)
            trts = trttrain[i, :].to(self.device)
            yhat = torch.rand_like(ys).to(self.device) * (ytrain.max() + 2 - ytrain.min()) + ytrain.min() - 1

            ntrt = torch.mean(trts)
            ncon = 1 - ntrt

            optimizer_g.zero_grad()
            optimizer_critic.zero_grad()
            optimizer_gen.zero_grad()
            optimizer_prop.zero_grad()

            with torch.no_grad():
                ylt = ys < yhat
                ylt = ylt.float()

            lats_i = self.gen_lat_i(xs)
            lats_s = self.gen_lat_s(xs)
            lats = torch.cat([lats_i, lats_s], dim=1)

            proplogit = self.prop_est(lats_s)

            props = torch.sigmoid(proplogit)

            lats1 = torch.cat([lats, props], dim=1)

            propl = self.proploss(props, trts)

            qhat_logit0 = self.g0(yhat, lats1)
            qhat_logit1 = self.g1(yhat, lats1)
            qhat_logit = qhat_logit0 * (1 - trts) + qhat_logit1 * trts

            gl = self.gloss(torch.sigmoid(qhat_logit), ylt)
            ipms = self.critic_ipm(lats_i)
            pos_ipm_loss = torch.mean(ipms * trts) / ntrt - torch.mean(ipms * (1. - trts)) / ncon

            combined_loss = gl + torch.mul(pos_ipm_loss, ipm_weight) + torch.mul(propl, prop_weight)
            combined_loss.backward(retain_graph=False)

            optimizer_gen.step()
            optimizer_prop.step()
            optimizer_g.step()

            if not self.EDU:
                my_lr_scheduler_prop.step()
                my_lr_scheduler_gen.step()

            g_loss.append(gl.cpu().item())
            prop_loss.append(propl.cpu().item())

            ##train critic #########################
            for j in range(n_critic):
                optimizer_g.zero_grad()
                optimizer_critic.zero_grad()
                optimizer_gen.zero_grad()

                i = np.random.choice(xtrain.shape[0], size=batch_size, replace=False)
                xs = xtrain[i, :].to(self.device)
                trts = trttrain[i, :].to(self.device)

                lats_i = self.gen_lat_i(xs)
                ipms = self.critic_ipm(lats_i)
                ntrt = torch.mean(trts)
                ncon = 1 - ntrt
                criticl = -(torch.mean(ipms * trts) / ntrt - torch.mean(ipms * (1. - trts)) / ncon)
                criticl.backward(retain_graph=False)
                optimizer_critic.step()

                # Clip critic weights
                for p in self.critic_ipm.parameters():
                    p.data.clamp_(-clip_value, clip_value)

            my_lr_scheduler_critic.step()
            critic_loss.append(criticl.cpu().item())

        self.g0.eval()
        self.g1.eval()
        self.critic_ipm.eval()
        self.gen_lat_i.eval()
        self.gen_lat_s.eval()

        return [g_loss, critic_loss, gen_loss, prop_loss]

    def predict(self, xtest, n_samples):
        xtest = torch.from_numpy(xtest).float()

        # mu_est = np.zeros(len(xtest))
        mus_est = np.zeros((len(xtest), n_samples))

        xtest = xtest.to(self.device)

        stmp_i = self.gen_lat_i(xtest.to(self.device))
        stmp_s = self.gen_lat_s(xtest.to(self.device))
        proplogitte = self.prop_est(stmp_s)
        propste = torch.sigmoid(proplogitte)
        stest = torch.cat([stmp_i, stmp_s, propste], dim=1)

        # use 1d interpolation for mean estimate
        for i in range(len(xtest)):

            probs0 = torch.sigmoid(self.g0(self.y_poss.to(self.device), stest[i].repeat(self.poss_vals, 1).to(
                self.device))).detach().cpu().numpy().ravel()

            probs0[0] = 0
            probs0[-1] = 1
            if not self.EDU:
                mus_tmp0 = interp1d(probs0, self.y_possnp)(np.random.uniform(0.005, 0.995, n_samples))
            else:
                mus_tmp0 = interp1d(probs0, self.y_possnp)(np.random.uniform(0.002, 0.998, n_samples))
            # mu_tmp0 = mus_tmp0.mean()

            probs1 = torch.sigmoid(self.g1(self.y_poss.to(self.device), stest[i].repeat(self.poss_vals, 1).to(
                self.device))).detach().cpu().numpy().ravel()

            probs1[0] = 0
            probs1[-1] = 1
            if not self.EDU:
                mus_tmp1 = interp1d(probs1, self.y_possnp)(np.random.uniform(0.005, 0.995, n_samples))
            else:
                mus_tmp1 = interp1d(probs1, self.y_possnp)(np.random.uniform(0.002, 0.998, n_samples))
            # mu_tmp1 = mus_tmp1.mean()

            mus_est[i, :] = mus_tmp1 - mus_tmp0
            # mu_est[i] = mu_tmp1 - mu_tmp0

        return mus_est
