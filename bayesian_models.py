import torch
import torch.nn as nn
#import geoopt as gt
import numpy as np
import torch.nn.functional as F
import math
import pickle

torch.autograd.set_detect_anomaly(True)

PI = torch.from_numpy(np.asarray(np.pi))
EPS = 1.e-5


class Bayesian_MuRP(nn.Module):
    def __init__(self, device, latent_dim=50, mure_dim=100, prior_path='', 
                num_rel=4, beta=1., dropout=0.5, hidden_dim=300, input_dim=1024, reg_type='kl'):
        super().__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mure_dim = mure_dim
        self.num_rel = num_rel
        self.beta = beta
        self.regularization_type = reg_type
        
        self.fc1 = nn.Linear(2*self.mure_dim, self.hidden_dim, bias=True)
        self.mu = nn.Linear(self.hidden_dim, self.latent_dim, bias=True)
        self.sigma = nn.Linear(self.hidden_dim, self.latent_dim, bias=True)
        self.mu_bn = nn.BatchNorm1d(self.latent_dim)
        self.var_bn = nn.BatchNorm1d(self.latent_dim)
        self.manifold = gt.Stereographic(-1)

        self.event_map_layer = nn.Linear(self.input_dim, self.mure_dim, bias=True)
        self.dropout = dropout
        if self.dropout > 0.:
            self.event_drop = nn.Dropout(self.dropout)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.mure_layer = nn.Linear(self.latent_dim, self.num_rel*self.mure_dim*2, bias=True)

        nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_uniform_(self.mure_layer.weight, nn.init.calculate_gain("tanh"))
        nn.init.xavier_uniform_(self.mu.weight, nn.init.calculate_gain("linear"))
        nn.init.xavier_uniform_(self.sigma.weight, nn.init.calculate_gain("linear"))
        nn.init.xavier_uniform_(self.event_map_layer.weight, nn.init.calculate_gain("linear"))

        self.batch_size = 1
        if len(prior_path) > 0:
            self.use_prior = True
            self.prior = torch.tensor(pickle.load(open(prior_path, 'rb'))).to(self.device)
            self.prior_layer = nn.Linear(self.prior.size(0), self.latent_dim)
            nn.init.xavier_uniform_(self.prior_layer.weight, nn.init.calculate_gain("tanh"))
        else:
            self.use_prior = False
            self.prior = None
        
    def forward(self, lm_hidden_state, epos_1, epos_2, rel, drop=True, get_z=False):
        # hidden_state [batch, length, 1024]
        self.batch_size = lm_hidden_state.size(0)

        e_1 = torch.gather(lm_hidden_state, dim=1,
                index=epos_1.unsqueeze(dim=1).expand(lm_hidden_state.size(0), lm_hidden_state.size(2)).unsqueeze(dim=1))
        e_2 = torch.gather(lm_hidden_state, dim=1,
                index=epos_2.unsqueeze(dim=1).expand(lm_hidden_state.size(0), lm_hidden_state.size(2)).unsqueeze(dim=1))

        e_1 = self.event_map_layer(e_1.squeeze(1))
        e_2 = self.event_map_layer(e_2.squeeze(1))

        event_reps = torch.cat((e_1, e_2), dim=1)

        event_reps = self.relu(self.fc1(event_reps))

        if self.dropout > 0. and drop:
            event_reps = self.event_drop(event_reps)
        
        mu = self.mu(event_reps)
        mu = self.mu_bn(mu)

        logvar = self.sigma(event_reps)
        logvar = self.var_bn(logvar)

        z = self.reparameterize(mu, logvar)

        if get_z:
            return [self.trans(z, e_1, e_2), mu, logvar, rel], mu
        else:
            return [self.trans(z, e_1, e_2), mu, logvar, rel]
    
    def sample(self, num_samples):
        if self.use_prior:
            mu = self.tanh(self.prior_layer(self.prior)).view(1, -1).expand(num_samples, -1)
        else:
            mu = torch.zeros(num_samples, self.latent_dim*self.num_rel)
        log_var = torch.zeros_like(mu)#.view(1, -1).expand(num_samples, -1)
        z = self.reparameterize(mu, log_var)

        trans_paras = self.tanh(self.mure_layer(z)) ######## non-linear is important

        Wu = trans_paras[:,:self.num_rel*self.mure_dim].view(trans_paras.size(0), self.num_rel, self.mure_dim)
        rvh = trans_paras[:,self.num_rel*self.mure_dim:].view(trans_paras.size(0), self.num_rel, self.mure_dim)
        # [num_rel, mure_dim]

        Wu = Wu.detach().cpu().numpy()
        rvh = rvh.detach().cpu().numpy()

        return Wu, rvh

    # mure
    def trans(self, z, e_1, e_2, out_reps=False):
        trans_paras = self.tanh(self.mure_layer(z)) ######## non-linear is important

        Wu = trans_paras[:,:self.num_rel*self.mure_dim].view(trans_paras.size(0), self.num_rel, self.mure_dim)
        rvh = trans_paras[:,self.num_rel*self.mure_dim:].view(trans_paras.size(0), self.num_rel, self.mure_dim)

        u = e_1
        v = self.manifold.expmap0(e_2)

        out_um = []
        out_vm = []

        logits = []
        for i in range(self.num_rel):
            Ru = Wu[:,i]
            add_v = rvh[:,i]

            u_m = u * Ru
            v_m = self.manifold.mobius_add(v, add_v)
            u_m = self.manifold.expmap0(u_m)
            out_um.append(u_m.detach().cpu().numpy())
            out_vm.append(v_m.detach().cpu().numpy())
            add_v = self.manifold.expmap0(add_v)
            #sqdist = torch.norm(u_m - v_m, 2, dim=-1)
            sqdist = self.manifold.dist(u_m, v_m)

            logits.append(sqdist.view(-1, 1))

        out_um = np.stack(out_um) # [num_rel, batch, um]
        out_vm = np.stack(out_vm) # [num_rel, batch, um]

        logits = torch.cat(logits, dim=1)

        if out_reps:
            return logits, out_um, out_vm
        else:
            return  logits
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        return eps * std.sqrt() + mu
    
    def sample_prior(self, batch_size):
        # sample from prior
        para_prior = self.tanh(self.prior_layer(self.prior))
        std = 1 # identity variance
        eps = torch.randn(batch_size, para_prior.size(-1)).to(self.device)
        return eps + para_prior.view(1, -1).expand(batch_size, -1).to(self.device)


    def loss_function(self,
                      *args,
                      **kwargs):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:model =
        :param kwargs:
        :return:
        """
        logits = args[0]
        mu = args[1]
        log_var = args[2]
        target = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        classification_loss = F.cross_entropy(torch.softmax(logits, dim=-1), target)

        if self.prior is not None:
            para_prior = self.tanh(self.prior_layer(self.prior))
            if self.regularization_type == 'mmd':
                kld_loss = self.mmd_penalty(self.reparameterize(mu, log_var), self.sample_prior(mu.size(0)))
            else:
                kld_loss = torch.mean(0.5 * torch.sum(-1 - log_var + (mu - para_prior)**2 + log_var.exp()))
        else:
            if self.regularization_type == 'mmd':
                kld_loss = self.mmd_penalty(self.reparameterize(mu, log_var), self.reparameterize(torch.zeros_like(mu), torch.zeros_like(log_var)))
            else:
                kld_loss = torch.mean(0.5 * torch.sum(-1 - log_var + mu ** 2 + log_var.exp(), dim = 1), dim = 0)

        # coefficient
        loss = classification_loss + self.beta * kld_weight  * kld_loss  # the kl loss is not too large

        return {'loss': loss, 'Classification_Loss': classification_loss, 'KLD': kld_loss}
    
    def mmd_penalty(self, sample_qz, sample_pz, kernel='IMQ'):
        n = self.batch_size
        nf = n * 1.0
        half_size = (n * n - n) / 2

        norms_pz = torch.sum(torch.square(sample_pz), dim=1, keepdim=True)
        dotprods_pz = torch.matmul(sample_pz, sample_pz.t())
        distances_pz = norms_pz + norms_pz.t() - 2. * dotprods_pz

        norms_qz = torch.sum(torch.square(sample_qz), dim=1, keepdim=True)
        dotprods_qz = torch.matmul(sample_qz, sample_qz.t())
        distances_qz = norms_qz + norms_qz.t() - 2. * dotprods_qz

        dotprods = torch.matmul(sample_qz, sample_pz.t())
        distances = norms_qz + norms_pz.t() - 2. * dotprods

        if kernel == 'RBF':
            sigma2_k = torch.topk(distances.view(-1), half_size)[-1]
            sigma2_k += torch.topk(distances_qz.view(-1), half_size)[-1]
            res1 = torch.exp(- distances_qz / 2. / sigma2_k)
            res1 += torch.exp(- distances_pz / 2. / sigma2_k)
            res1 = res1 * (1. - torch.eye(n, device=self.device))
            res1 = torch.sum(res1) / (nf * nf - nf)
            res2 = torch.exp(- distances / 2. / sigma2_k)
            res2 = torch.sum(res2) * 2. / (nf * nf)
            stat = res1 - res2
        elif kernel == 'IMQ':
            Cbase = 2. * self.latent_dim * 2. * 1. # sigma2_p # for normal sigma2_p = 1
            stat = 0.
            for scale in [.1, .2, .5, 1., 2., 5., 10.]:
                C = Cbase * scale
                res1 = C / (C + distances_qz)
                res1 += C / (C + distances_pz)
                res1 = res1 * (1. - torch.eye(n, device=self.device))
                res1 = torch.sum(res1) / (nf * nf - nf)
                res2 = C / (C + distances)
                res2 = torch.sum(res2) * 2. / (nf * nf)
                stat += res1 - res2
        return stat


class Bayesian_TransE(nn.Module):
    def __init__(self, device, latent_dim=50, mure_dim=100, prior_path='',
                num_rel=4, beta=1., dropout=0.5, hidden_dim=300, input_dim=1024, reg_type='kl'):
        super().__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.mure_dim = mure_dim
        self.num_rel = num_rel
        self.beta = beta
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.fc1 = nn.Linear(2*self.mure_dim, self.hidden_dim, bias=True)
        self.mu = nn.Linear(self.hidden_dim, self.latent_dim, bias=True)
        self.sigma = nn.Linear(self.hidden_dim, self.latent_dim, bias=True)
        self.mu_bn = nn.BatchNorm1d(self.latent_dim)
        self.var_bn = nn.BatchNorm1d(self.latent_dim)
        self.regularization_type = reg_type

        self.event_map_layer = nn.Linear(self.input_dim, self.mure_dim, bias=True)
        self.dropout = dropout
        if self.dropout > 0.:
            self.event_drop = nn.Dropout(self.dropout)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.mure_layer = nn.Linear(self.latent_dim, self.num_rel*self.mure_dim, bias=True)

        nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_uniform_(self.mure_layer.weight, nn.init.calculate_gain("tanh"))
        nn.init.xavier_uniform_(self.mu.weight, nn.init.calculate_gain("linear"))
        nn.init.xavier_uniform_(self.sigma.weight, nn.init.calculate_gain("linear"))
        nn.init.xavier_uniform_(self.event_map_layer.weight, nn.init.calculate_gain("linear"))
        self.batch_size = 1

        if len(prior_path) > 0:
            self.use_prior = True
            self.prior = torch.tensor(pickle.load(open(prior_path, 'rb'))).to(self.device)
            self.prior_layer = nn.Linear(self.prior.size(0), self.latent_dim)
            nn.init.xavier_uniform_(self.prior_layer.weight, nn.init.calculate_gain("tanh"))
        else:
            self.use_prior = False
            self.prior = None

    def mmd_penalty(self, sample_qz, sample_pz, kernel='IMQ'):
        n = self.batch_size
        nf = n * 1.0
        half_size = (n * n - n) / 2

        norms_pz = torch.sum(torch.square(sample_pz), dim=1, keepdim=True)
        dotprods_pz = torch.matmul(sample_pz, sample_pz.t())
        distances_pz = norms_pz + norms_pz.t() - 2. * dotprods_pz

        norms_qz = torch.sum(torch.square(sample_qz), dim=1, keepdim=True)
        dotprods_qz = torch.matmul(sample_qz, sample_qz.t())
        distances_qz = norms_qz + norms_qz.t() - 2. * dotprods_qz

        dotprods = torch.matmul(sample_qz, sample_pz.t())
        distances = norms_qz + norms_pz.t() - 2. * dotprods

        if kernel == 'RBF':
            sigma2_k = torch.topk(distances.view(-1), half_size)[-1]
            sigma2_k += torch.topk(distances_qz.view(-1), half_size)[-1]
            res1 = torch.exp(- distances_qz / 2. / sigma2_k)
            res1 += torch.exp(- distances_pz / 2. / sigma2_k)
            res1 = res1 * (1. - torch.eye(n, device=self.device))
            res1 = torch.sum(res1) / (nf * nf - nf)
            res2 = torch.exp(- distances / 2. / sigma2_k)
            res2 = torch.sum(res2) * 2. / (nf * nf)
            stat = res1 - res2
        elif kernel == 'IMQ':
            Cbase = 2. * self.latent_dim * 2. * 1. # sigma2_p # for normal sigma2_p = 1
            stat = 0.
            for scale in [.1, .2, .5, 1., 2., 5., 10.]:
                C = Cbase * scale
                res1 = C / (C + distances_qz)
                res1 += C / (C + distances_pz)
                res1 = res1 * (1. - torch.eye(n, device=self.device))
                res1 = torch.sum(res1) / (nf * nf - nf)
                res2 = C / (C + distances)
                res2 = torch.sum(res2) * 2. / (nf * nf)
                stat += res1 - res2
        return stat
    
    def sample_prior(self, batch_size):
        # sample from prior
        para_prior = self.tanh(self.prior_layer(self.prior))
        std = 1 # identity variance
        eps = torch.randn(batch_size, para_prior.size(-1)).to(self.device)
        return eps + para_prior.view(1, -1).expand(batch_size, -1).to(self.device)

    def forward(self, lm_hidden_state, epos_1, epos_2, rel, drop=True):
        self.batch_size = lm_hidden_state.size(0)

        e_1 = torch.gather(lm_hidden_state, dim=1,
                index=epos_1.unsqueeze(dim=1).expand(lm_hidden_state.size(0), lm_hidden_state.size(2)).unsqueeze(dim=1))
        e_2 = torch.gather(lm_hidden_state, dim=1,
                index=epos_2.unsqueeze(dim=1).expand(lm_hidden_state.size(0), lm_hidden_state.size(2)).unsqueeze(dim=1))

        e_1 = self.event_map_layer(e_1.squeeze(1))
        e_2 = self.event_map_layer(e_2.squeeze(1))

        event_reps = torch.cat((e_1, e_2), dim=1)

        event_reps = self.relu(self.fc1(event_reps))

        if self.dropout > 0. and drop:
            event_reps = self.event_drop(event_reps)
        
        mu = self.mu(event_reps)
        mu = self.mu_bn(mu)

        logvar = self.sigma(event_reps)
        logvar = self.var_bn(logvar)

        z = self.reparameterize(mu, logvar)

        return [self.trans(z, e_1, e_2), mu, logvar, rel]

    # TransE
    def trans(self, z, e_1, e_2):
        trans_paras = self.tanh(self.mure_layer(z))
        rvh = trans_paras.view(trans_paras.size(0), self.num_rel, self.mure_dim)

        u = e_1
        v = e_2

        logits = []
        for i in range(self.num_rel):
            add_v = rvh[:,i]

            u_m = u + add_v
            v_m = v #+ add_v

            sqdist = torch.norm(u_m - v_m, 2, dim=-1)

            logits.append(sqdist.view(-1, 1))

        return torch.cat(logits, dim=1)
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        return eps * std.sqrt() + mu

    def loss_function(self,
                      *args,
                      **kwargs):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:model =
        :param kwargs:
        :return:
        """
        logits = args[0]
        mu = args[1]
        log_var = args[2]
        target = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        classification_loss = F.cross_entropy(F.softmax(logits, dim=-1), target)

        if self.prior is not None:
            para_prior = self.tanh(self.prior_layer(self.prior))
            if self.regularization_type == 'mmd':
                kld_loss = self.mmd_penalty(self.reparameterize(mu, log_var), self.sample_prior(mu.size(0)))
            else:
                kld_loss = torch.mean(0.5 * torch.sum(-1 - log_var + (mu - para_prior)**2 + log_var.exp()))
        else:
            if self.regularization_type == 'mmd':
                kld_loss = self.mmd_penalty(self.reparameterize(mu, log_var), self.reparameterize(torch.zeros_like(mu), torch.zeros_like(log_var)))
            else:
                kld_loss = torch.mean(0.5 * torch.sum(-1 - log_var + mu ** 2 + log_var.exp(), dim = 1), dim = 0)

        # coefficient
        loss = classification_loss + self.beta * kld_weight  * kld_loss  # the kl loss is not too large

        return {'loss': loss, 'Classification_Loss': classification_loss, 'KLD': kld_loss}


class Bayesian_MuRE(nn.Module):
    def __init__(self, device, latent_dim=50, mure_dim=100, prior_path='', 
                num_rel=4, beta=1., dropout=0.5, hidden_dim=300, input_dim=1024, reg_type='kl'):
        super().__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mure_dim = mure_dim
        self.num_rel = num_rel
        self.beta = beta
        self.regularization_type = reg_type
        
        self.fc1 = nn.Linear(2*self.mure_dim, self.hidden_dim, bias=True)
        self.mu = nn.Linear(self.hidden_dim, self.latent_dim, bias=True)
        self.sigma = nn.Linear(self.hidden_dim, self.latent_dim, bias=True)
        self.mu_bn = nn.BatchNorm1d(self.latent_dim)
        self.var_bn = nn.BatchNorm1d(self.latent_dim)

        self.event_map_layer = nn.Linear(self.input_dim, self.mure_dim, bias=True)
        self.dropout = dropout
        if self.dropout > 0.:
            self.event_drop = nn.Dropout(self.dropout)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.mure_layer = nn.Linear(self.latent_dim, self.num_rel*self.mure_dim*2, bias=True)

        nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_uniform_(self.mure_layer.weight, nn.init.calculate_gain("tanh"))
        nn.init.xavier_uniform_(self.mu.weight, nn.init.calculate_gain("linear"))
        nn.init.xavier_uniform_(self.sigma.weight, nn.init.calculate_gain("linear"))
        nn.init.xavier_uniform_(self.event_map_layer.weight, nn.init.calculate_gain("linear"))

        self.batch_size = 1
        if len(prior_path) > 0:
            self.use_prior = True
            self.prior = torch.tensor(pickle.load(open(prior_path, 'rb'))).to(self.device)
            self.prior_layer = nn.Linear(self.prior.size(0), self.latent_dim)
            nn.init.xavier_uniform_(self.prior_layer.weight, nn.init.calculate_gain("tanh"))
        else:
            self.use_prior = False
            self.prior = None
        
    def forward(self, lm_hidden_state, epos_1, epos_2, rel, drop=True, get_z=False):
        # hidden_state [batch, length, 1024]
        self.batch_size = lm_hidden_state.size(0)

        e_1 = torch.gather(lm_hidden_state, dim=1,
                index=epos_1.unsqueeze(dim=1).expand(lm_hidden_state.size(0), lm_hidden_state.size(2)).unsqueeze(dim=1))
        e_2 = torch.gather(lm_hidden_state, dim=1,
                index=epos_2.unsqueeze(dim=1).expand(lm_hidden_state.size(0), lm_hidden_state.size(2)).unsqueeze(dim=1))

        e_1 = self.event_map_layer(e_1.squeeze(1))
        e_2 = self.event_map_layer(e_2.squeeze(1))

        event_reps = torch.cat((e_1, e_2), dim=1)

        event_reps = self.relu(self.fc1(event_reps))

        if self.dropout > 0. and drop:
            event_reps = self.event_drop(event_reps)
        
        mu = self.mu(event_reps)
        mu = self.mu_bn(mu)

        logvar = self.sigma(event_reps)
        logvar = self.var_bn(logvar)

        z = self.reparameterize(mu, logvar)

        if get_z:
            return [self.trans(z, e_1, e_2), mu, logvar, rel], mu
        else:
            return [self.trans(z, e_1, e_2), mu, logvar, rel]
    
    def sample(self, num_samples):
        if self.use_prior:
            mu = self.tanh(self.prior_layer(self.prior)).view(1, -1).expand(num_samples, -1)
        else:
            mu = torch.zeros(num_samples, self.latent_dim*self.num_rel)
        log_var = torch.zeros_like(mu)#.view(1, -1).expand(num_samples, -1)
        z = self.reparameterize(mu, log_var)

        trans_paras = self.tanh(self.mure_layer(z)) ######## non-linear is important

        Wu = trans_paras[:,:self.num_rel*self.mure_dim].view(trans_paras.size(0), self.num_rel, self.mure_dim)
        rvh = trans_paras[:,self.num_rel*self.mure_dim:].view(trans_paras.size(0), self.num_rel, self.mure_dim)
        # [num_rel, mure_dim]

        Wu = Wu.detach().cpu().numpy()
        rvh = rvh.detach().cpu().numpy()

        return Wu, rvh

    # mure
    def trans(self, z, e_1, e_2, out_reps=False):
        trans_paras = self.tanh(self.mure_layer(z)) ######## non-linear is important

        Wu = trans_paras[:,:self.num_rel*self.mure_dim].view(trans_paras.size(0), self.num_rel, self.mure_dim)
        rvh = trans_paras[:,self.num_rel*self.mure_dim:].view(trans_paras.size(0), self.num_rel, self.mure_dim)

        u = e_1
        v = e_2

        out_um = []
        out_vm = []

        logits = []
        for i in range(self.num_rel):
            Ru = Wu[:,i]
            add_v = rvh[:,i]

            u_m = u * Ru
            v_m = v + add_v

            out_um.append(u_m.detach().cpu().numpy())
            out_vm.append(v_m.detach().cpu().numpy())

            sqdist = - torch.norm(u_m - v_m, 2, dim=-1)
            #sqdist = sqdist ** 2 # necessary?

            logits.append(sqdist.view(-1, 1))

        out_um = np.stack(out_um) # [num_rel, batch, um]
        out_vm = np.stack(out_vm) # [num_rel, batch, um]

        logits = torch.cat(logits, dim=1)

        if out_reps:
            return logits, out_um, out_vm
        else:
            return  logits
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        return eps * std.sqrt() + mu
    
    def sample_prior(self, batch_size):
        # sample from prior
        para_prior = self.tanh(self.prior_layer(self.prior))
        std = 1 # identity variance
        eps = torch.randn(batch_size, para_prior.size(-1)).to(self.device)
        return eps + para_prior.view(1, -1).expand(batch_size, -1).to(self.device)


    def loss_function(self,
                      *args,
                      **kwargs):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:model =
        :param kwargs:
        :return:
        """
        logits = args[0]
        mu = args[1]
        log_var = args[2]
        target = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        classification_loss = F.cross_entropy(torch.softmax(logits, dim=-1), target)

        if self.prior is not None:
            para_prior = self.tanh(self.prior_layer(self.prior))
            if self.regularization_type == 'mmd':
                kld_loss = self.mmd_penalty(self.reparameterize(mu, log_var), self.sample_prior(mu.size(0)))
            else:
                kld_loss = torch.mean(0.5 * torch.sum(-1 - log_var + (mu - para_prior)**2 + log_var.exp()))
        else:
            if self.regularization_type == 'mmd':
                kld_loss = self.mmd_penalty(self.reparameterize(mu, log_var), self.reparameterize(torch.zeros_like(mu), torch.zeros_like(log_var)))
            else:
                kld_loss = torch.mean(0.5 * torch.sum(-1 - log_var + mu ** 2 + log_var.exp(), dim = 1), dim = 0)

        # coefficient
        loss = classification_loss + self.beta * kld_weight  * kld_loss  # the kl loss is not too large

        return {'loss': loss, 'Classification_Loss': classification_loss, 'KLD': kld_loss}
    
    def mmd_penalty(self, sample_qz, sample_pz, kernel='IMQ'):
        n = self.batch_size
        nf = n * 1.0
        half_size = math.ceil((n * n - n) / 2)

        norms_pz = torch.sum(torch.square(sample_pz), dim=1, keepdim=True)
        dotprods_pz = torch.matmul(sample_pz, sample_pz.t())
        distances_pz = norms_pz + norms_pz.t() - 2. * dotprods_pz

        norms_qz = torch.sum(torch.square(sample_qz), dim=1, keepdim=True)
        dotprods_qz = torch.matmul(sample_qz, sample_qz.t())
        distances_qz = norms_qz + norms_qz.t() - 2. * dotprods_qz

        dotprods = torch.matmul(sample_qz, sample_pz.t())
        distances = norms_qz + norms_pz.t() - 2. * dotprods

        if kernel == 'RBF':
            sigma2_k = torch.topk(distances.view(-1), half_size).values[half_size-1]
            sigma2_k += torch.topk(distances_qz.view(-1), half_size).values[half_size-1]
            res1 = torch.exp(- distances_qz / 2. / sigma2_k)
            res1 = res1 + torch.exp(- distances_pz / 2. / sigma2_k)
            res1 = res1 * (1. - torch.eye(n, device=self.device))
            res1 = torch.sum(res1) / (nf * nf - nf)
            res2 = torch.exp(- distances / 2. / sigma2_k)
            res2 = torch.sum(res2) * 2. / (nf * nf)
            stat = res1 - res2
        elif kernel == 'IMQ':
            Cbase = 2. * self.latent_dim * 2. * 1. # sigma2_p # for normal sigma2_p = 1
            stat = 0.
            for scale in [.1, .2, .5, 1., 2., 5., 10.]:
                C = Cbase * scale
                res1 = C / (C + distances_qz)
                res1 += C / (C + distances_pz)
                res1 = res1 * (1. - torch.eye(n, device=self.device))
                res1 = torch.sum(res1) / (nf * nf - nf)
                res2 = C / (C + distances)
                res2 = torch.sum(res2) * 2. / (nf * nf)
                stat += res1 - res2
        return stat
    



class Bayesian_MuRE_LSTM(nn.Module):
    def __init__(self, device, latent_dim=50, mure_dim=100, prior_path='', 
                num_rel=4, beta=1., dropout=0.5, hidden_dim=300, input_dim=1024, reg_type='kl'):
        super().__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mure_dim = mure_dim
        self.num_rel = num_rel
        self.beta = beta
        self.regularization_type = reg_type
        
        self.fc1 = nn.Linear(2*self.mure_dim, self.hidden_dim, bias=True)
        self.mu = nn.Linear(self.hidden_dim, self.latent_dim, bias=True)
        self.sigma = nn.Linear(self.hidden_dim, self.latent_dim, bias=True)
        self.mu_bn = nn.BatchNorm1d(self.latent_dim)
        self.var_bn = nn.BatchNorm1d(self.latent_dim)

        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=math.ceil(self.mure_dim/2),
                            batch_first=True, bidirectional=True)

        self.event_map_layer = nn.Linear(self.input_dim, self.mure_dim, bias=True)
        self.dropout = dropout
        if self.dropout > 0.:
            self.event_drop = nn.Dropout(self.dropout)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.mure_layer = nn.Linear(self.latent_dim, self.num_rel*self.mure_dim*2, bias=True)

        nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_uniform_(self.mure_layer.weight, nn.init.calculate_gain("tanh"))
        nn.init.xavier_uniform_(self.mu.weight, nn.init.calculate_gain("linear"))
        nn.init.xavier_uniform_(self.sigma.weight, nn.init.calculate_gain("linear"))
        nn.init.xavier_uniform_(self.event_map_layer.weight, nn.init.calculate_gain("linear"))

        self.batch_size = 1
        if len(prior_path) > 0:
            self.use_prior = True
            self.prior = torch.tensor(pickle.load(open(prior_path, 'rb'))).to(self.device)
            self.prior_layer = nn.Linear(self.prior.size(0), self.latent_dim)
            nn.init.xavier_uniform_(self.prior_layer.weight, nn.init.calculate_gain("tanh"))
        else:
            self.use_prior = False
            self.prior = None


    def forward(self, lm_hidden_state, epos_1, epos_2, rel, drop=True):
        # hidden_state [batch, length, 1024]

        lstm_states = self.lstm(lm_hidden_state)[0]

        e_1 = torch.gather(lm_hidden_state, dim=1,
                index=epos_1.unsqueeze(dim=1).expand(lstm_states.size(0), lstm_states.size(2)).unsqueeze(dim=1))
        e_2 = torch.gather(lm_hidden_state, dim=1,
                index=epos_2.unsqueeze(dim=1).expand(lstm_states.size(0), lstm_states.size(2)).unsqueeze(dim=1))

        e_1 = e_1.squeeze(1)
        e_2 = e_2.squeeze(1)

        event_reps = torch.cat((e_1, e_2), dim=1)

        event_reps = self.relu(self.fc1(event_reps))

        if self.dropout > 0. and drop:
            event_reps = self.event_drop(event_reps)
        
        mu = self.mu(event_reps)
        mu = self.mu_bn(mu)

        logvar = self.sigma(event_reps)
        logvar = self.var_bn(logvar)

        z = self.reparameterize(mu, logvar)

        return [self.trans(z, e_1, e_2), mu, logvar, rel]

    def sample(self, num_samples):
        if self.use_prior:
            mu = self.tanh(self.prior_layer(self.prior)).view(1, -1).expand(num_samples, -1)
        else:
            mu = torch.zeros(num_samples, self.latent_dim*self.num_rel)
        log_var = torch.zeros_like(mu)#.view(1, -1).expand(num_samples, -1)
        z = self.reparameterize(mu, log_var)

        trans_paras = self.tanh(self.mure_layer(z)) ######## non-linear is important

        Wu = trans_paras[:,:self.num_rel*self.mure_dim].view(trans_paras.size(0), self.num_rel, self.mure_dim)
        rvh = trans_paras[:,self.num_rel*self.mure_dim:].view(trans_paras.size(0), self.num_rel, self.mure_dim)
        # [num_rel, mure_dim]

        Wu = Wu.detach().cpu().numpy()
        rvh = rvh.detach().cpu().numpy()

        return Wu, rvh

    # mure
    def trans(self, z, e_1, e_2, out_reps=False):
        trans_paras = self.tanh(self.mure_layer(z)) ######## non-linear is important

        Wu = trans_paras[:,:self.num_rel*self.mure_dim].view(trans_paras.size(0), self.num_rel, self.mure_dim)
        rvh = trans_paras[:,self.num_rel*self.mure_dim:].view(trans_paras.size(0), self.num_rel, self.mure_dim)

        u = e_1
        v = e_2

        out_um = []
        out_vm = []

        logits = []
        for i in range(self.num_rel):
            Ru = Wu[:,i]
            add_v = rvh[:,i]

            u_m = u * Ru
            v_m = v + add_v

            out_um.append(u_m.detach().cpu().numpy())
            out_vm.append(v_m.detach().cpu().numpy())

            sqdist = - torch.norm(u_m - v_m, 2, dim=-1)
            #sqdist = sqdist ** 2 # necessary?

            logits.append(sqdist.view(-1, 1))

        out_um = np.stack(out_um) # [num_rel, batch, um]
        out_vm = np.stack(out_vm) # [num_rel, batch, um]

        logits = torch.cat(logits, dim=1)

        if out_reps:
            return logits, out_um, out_vm
        else:
            return  logits
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        return eps * std.sqrt() + mu
    
    def sample_prior(self, batch_size):
        # sample from prior
        para_prior = self.tanh(self.prior_layer(self.prior))
        std = 1 # identity variance
        eps = torch.randn(batch_size, para_prior.size(-1)).to(self.device)
        return eps + para_prior.view(1, -1).expand(batch_size, -1).to(self.device)


    def loss_function(self,
                      *args,
                      **kwargs):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:model =
        :param kwargs:
        :return:
        """
        logits = args[0]
        mu = args[1]
        log_var = args[2]
        target = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        classification_loss = F.cross_entropy(torch.softmax(logits, dim=-1), target)

        if self.prior is not None:
            para_prior = self.tanh(self.prior_layer(self.prior))
            if self.regularization_type == 'mmd':
                kld_loss = self.mmd_penalty(self.reparameterize(mu, log_var), self.sample_prior(mu.size(0)))
            else:
                kld_loss = torch.mean(0.5 * torch.sum(-1 - log_var + (mu - para_prior)**2 + log_var.exp()))
        else:
            if self.regularization_type == 'mmd':
                kld_loss = self.mmd_penalty(self.reparameterize(mu, log_var), self.reparameterize(torch.zeros_like(mu), torch.zeros_like(log_var)))
            else:
                kld_loss = torch.mean(0.5 * torch.sum(-1 - log_var + mu ** 2 + log_var.exp(), dim = 1), dim = 0)

        # coefficient
        loss = classification_loss + self.beta * kld_weight  * kld_loss  # the kl loss is not too large

        return {'loss': loss, 'Classification_Loss': classification_loss, 'KLD': kld_loss}
    
    def mmd_penalty(self, sample_qz, sample_pz, kernel='IMQ'):
        n = self.batch_size
        nf = n * 1.0
        half_size = (n * n - n) / 2

        norms_pz = torch.sum(torch.square(sample_pz), dim=1, keepdim=True)
        dotprods_pz = torch.matmul(sample_pz, sample_pz.t())
        distances_pz = norms_pz + norms_pz.t() - 2. * dotprods_pz

        norms_qz = torch.sum(torch.square(sample_qz), dim=1, keepdim=True)
        dotprods_qz = torch.matmul(sample_qz, sample_qz.t())
        distances_qz = norms_qz + norms_qz.t() - 2. * dotprods_qz

        dotprods = torch.matmul(sample_qz, sample_pz.t())
        distances = norms_qz + norms_pz.t() - 2. * dotprods

        if kernel == 'RBF':
            sigma2_k = torch.topk(distances.view(-1), half_size)[-1]
            sigma2_k += torch.topk(distances_qz.view(-1), half_size)[-1]
            res1 = torch.exp(- distances_qz / 2. / sigma2_k)
            res1 += torch.exp(- distances_pz / 2. / sigma2_k)
            res1 = res1 * (1. - torch.eye(n, device=self.device))
            res1 = torch.sum(res1) / (nf * nf - nf)
            res2 = torch.exp(- distances / 2. / sigma2_k)
            res2 = torch.sum(res2) * 2. / (nf * nf)
            stat = res1 - res2
        elif kernel == 'IMQ':
            Cbase = 2. * self.latent_dim * 2. * 1. # sigma2_p # for normal sigma2_p = 1
            stat = 0.
            for scale in [.1, .2, .5, 1., 2., 5., 10.]:
                C = Cbase * scale
                res1 = C / (C + distances_qz)
                res1 += C / (C + distances_pz)
                res1 = res1 * (1. - torch.eye(n, device=self.device))
                res1 = torch.sum(res1) / (nf * nf - nf)
                res2 = C / (C + distances)
                res2 = torch.sum(res2) * 2. / (nf * nf)
                stat += res1 - res2
        return stat


class Bayesian_AttH(nn.Module):
    def __init__(self, device, latent_dim=50, mure_dim=100, prior_path='', 
                num_rel=4, beta=1., dropout=0.5, hidden_dim=300, input_dim=1024, cur=1, reg_type='kl'):
        super().__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mure_dim = mure_dim
        self.num_rel = num_rel
        self.beta = beta
        self.cur = cur
        self.manifold = gt.Stereographic(-self.cur, True)
        self.regularization_type = reg_type

        self.fc1 = nn.Linear(2*self.mure_dim, self.hidden_dim, bias=True)
        self.mu = nn.Linear(self.hidden_dim, self.latent_dim, bias=True)
        self.sigma = nn.Linear(self.hidden_dim, self.latent_dim, bias=True)
        self.mu_bn = nn.BatchNorm1d(self.latent_dim)
        self.var_bn = nn.BatchNorm1d(self.latent_dim)

        self.event_map_layer = nn.Linear(self.input_dim, self.mure_dim, bias=True)
        self.dropout = dropout
        if self.dropout > 0.:
            self.event_drop = nn.Dropout(self.dropout)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.mure_layer = nn.Linear(self.latent_dim, self.num_rel*self.mure_dim*4, bias=True)

        nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_uniform_(self.mure_layer.weight, nn.init.calculate_gain("tanh"))
        nn.init.xavier_uniform_(self.mu.weight, nn.init.calculate_gain("linear"))
        nn.init.xavier_uniform_(self.sigma.weight, nn.init.calculate_gain("linear"))
        nn.init.xavier_uniform_(self.event_map_layer.weight, nn.init.calculate_gain("linear"))

        self.scale = torch.Tensor([1. / np.sqrt(self.mure_dim)]).to(self.device)
        self.softmax = nn.Softmax(dim=1)

        if len(prior_path) > 0:
            self.prior = pickle.load(open(prior_path, 'rb'))
            self.prior = torch.tensor(self.prior)
            self.prior = self.prior.flatten().to(self.device)
            self.prior_layer = nn.Linear(self.prior.size(0), self.latent_dim)
            nn.init.xavier_uniform_(self.prior_layer.weight, nn.init.calculate_gain("tanh"))
        else:
            self.prior = None

    def forward(self, lm_hidden_state, epos_1, epos_2, rel, drop=True):
        self.batch_size = lm_hidden_state.size(0)

        # hidden_state [batch, length, 1024]
        e_1 = torch.gather(lm_hidden_state, dim=1,
                index=epos_1.unsqueeze(dim=1).expand(lm_hidden_state.size(0), lm_hidden_state.size(2)).unsqueeze(dim=1))
        e_2 = torch.gather(lm_hidden_state, dim=1,
                index=epos_2.unsqueeze(dim=1).expand(lm_hidden_state.size(0), lm_hidden_state.size(2)).unsqueeze(dim=1))

        e_1 = self.event_map_layer(e_1.squeeze(1))
        e_2 = self.event_map_layer(e_2.squeeze(1))

        event_reps = torch.cat((e_1, e_2), dim=1)

        event_reps = self.relu(self.fc1(event_reps))

        if self.dropout > 0. and drop:
            event_reps = self.event_drop(event_reps)
        
        mu = self.mu(event_reps)
        mu = self.mu_bn(mu)

        logvar = self.sigma(event_reps)
        logvar = self.var_bn(logvar)

        z = self.reparameterize(mu, logvar)

        return [self.trans(z, e_1, e_2), mu, logvar, rel]

    # mure
    def trans(self, z, e_1, e_2):
        trans_paras = self.tanh(self.mure_layer(z)) ######## non-linear is important

        rot_q = trans_paras[:,:self.num_rel*self.mure_dim].view(trans_paras.size(0), self.num_rel, self.mure_dim)
        ref_q = trans_paras[:,self.num_rel*self.mure_dim:self.num_rel*self.mure_dim*2].view(trans_paras.size(0), self.num_rel, self.mure_dim)
        context_vec = trans_paras[:,self.num_rel*self.mure_dim*2:self.num_rel*self.mure_dim*3].view(trans_paras.size(0), self.num_rel, self.mure_dim)
        rel = trans_paras[:,self.num_rel*self.mure_dim*3:].view(trans_paras.size(0), self.num_rel, self.mure_dim)

        u = self.manifold.expmap0(e_1)
        v = self.manifold.expmap0(e_2)

        logits = []
        for i in range(self.num_rel):
            this_rot_q = rot_q[:,i]
            this_ref_q = ref_q[:,i]
            this_cv = context_vec[:,i].view((-1, 1, self.mure_dim))

            rt = givens_rotations(this_rot_q, u).view((-1, 1, self.mure_dim))
            rf = givens_reflection(this_ref_q, u).view((-1, 1, self.mure_dim))

            rt = self.manifold.logmap0(rt)
            rf = self.manifold.logmap0(rf)
            cands = torch.cat([rf, rt], dim=1)

            att_weights = torch.sum(this_cv * cands * self.scale, dim=-1, keepdim=True)
            att_weights = self.softmax(att_weights)

            att_q = torch.sum(att_weights * cands, dim=1)
            lhs = self.manifold.expmap0(att_q)

            this_rel = rel[:,i]
            res = self.manifold.mobius_add(lhs, this_rel)


            sqdist = self.manifold.dist2(res, v)
            #sqdist = sqdist ** 2 # necessary?

            logits.append(sqdist.view(-1, 1))

        return torch.cat(logits, dim=1)
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        return eps * std.sqrt() + mu

    def sample_prior(self, batch_size):
        # sample from prior
        para_prior = self.tanh(self.prior_layer(self.prior))
        std = 1 # identity variance
        eps = torch.randn(batch_size, para_prior.size(-1)).to(self.device)
        return eps + para_prior.view(1, -1).expand(batch_size, -1).to(self.device)


    def loss_function(self,
                      *args,
                      **kwargs):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:model =
        :param kwargs:
        :return:
        """
        logits = args[0]
        mu = args[1]
        log_var = args[2]
        target = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        classification_loss = F.cross_entropy(torch.softmax(logits, dim=-1), target)

        if self.prior is not None:
            para_prior = self.tanh(self.prior_layer(self.prior))
            if self.regularization_type == 'mmd':
                kld_loss = self.mmd_penalty(self.reparameterize(mu, log_var), self.sample_prior(mu.size(0)))
            else:
                kld_loss = torch.mean(0.5 * torch.sum(-1 - log_var + (mu - para_prior)**2 + log_var.exp()))
        else:
            if self.regularization_type == 'mmd':
                kld_loss = self.mmd_penalty(self.reparameterize(mu, log_var), self.reparameterize(torch.zeros_like(mu), torch.zeros_like(log_var)))
            else:
                kld_loss = torch.mean(0.5 * torch.sum(-1 - log_var + mu ** 2 + log_var.exp(), dim = 1), dim = 0)

        # coefficient
        loss = classification_loss + self.beta * kld_weight  * kld_loss  # the kl loss is not too large

        return {'loss': loss, 'Classification_Loss': classification_loss, 'KLD': kld_loss}
    
    def mmd_penalty(self, sample_qz, sample_pz, kernel='IMQ'):
        n = self.batch_size
        nf = n * 1.0
        half_size = (n * n - n) / 2

        norms_pz = torch.sum(torch.square(sample_pz), dim=1, keepdim=True)
        dotprods_pz = torch.matmul(sample_pz, sample_pz.t())
        distances_pz = norms_pz + norms_pz.t() - 2. * dotprods_pz

        norms_qz = torch.sum(torch.square(sample_qz), dim=1, keepdim=True)
        dotprods_qz = torch.matmul(sample_qz, sample_qz.t())
        distances_qz = norms_qz + norms_qz.t() - 2. * dotprods_qz

        dotprods = torch.matmul(sample_qz, sample_pz.t())
        distances = norms_qz + norms_pz.t() - 2. * dotprods

        if kernel == 'RBF':
            sigma2_k = torch.topk(distances.view(-1), half_size)[-1]
            sigma2_k += torch.topk(distances_qz.view(-1), half_size)[-1]
            res1 = torch.exp(- distances_qz / 2. / sigma2_k)
            res1 += torch.exp(- distances_pz / 2. / sigma2_k)
            res1 = res1 * (1. - torch.eye(n, device=self.device))
            res1 = torch.sum(res1) / (nf * nf - nf)
            res2 = torch.exp(- distances / 2. / sigma2_k)
            res2 = torch.sum(res2) * 2. / (nf * nf)
            stat = res1 - res2
        elif kernel == 'IMQ':
            Cbase = 2. * self.latent_dim * 2. * 1. # sigma2_p # for normal sigma2_p = 1
            stat = 0.
            for scale in [.1, .2, .5, 1., 2., 5., 10.]:
                C = Cbase * scale
                res1 = C / (C + distances_qz)
                res1 += C / (C + distances_pz)
                res1 = res1 * (1. - torch.eye(n, device=self.device))
                res1 = torch.sum(res1) / (nf * nf - nf)
                res2 = C / (C + distances)
                res2 = torch.sum(res2) * 2. / (nf * nf)
                stat += res1 - res2
        return stat

def givens_rotations(r, x):
    """Givens rotations.
    Args:
        r: torch.Tensor of shape (N x d), rotation parameters
        x: torch.Tensor of shape (N x d), points to rotate
    Returns:
        torch.Tensor os shape (N x d) representing rotation of x by r
    """
    givens = r.view((r.shape[0], -1, 2))
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
    x = x.view((x.shape[0], -1, 2))
    x_rot = givens[:, :, 0:1] * x + givens[:, :, 1:] * torch.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    return x_rot.view((r.shape[0], -1))


def givens_reflection(r, x):
    """Givens reflections.

    Args:
        r: torch.Tensor of shape (N x d), rotation parameters
        x: torch.Tensor of shape (N x d), points to reflect

    Returns:
        torch.Tensor os shape (N x d) representing reflection of x by r
    """
    givens = r.view((r.shape[0], -1, 2))
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
    x = x.view((x.shape[0], -1, 2))
    x_ref = givens[:, :, 0:1] * torch.cat((x[:, :, 0:1], -x[:, :, 1:]), dim=-1) + givens[:, :, 1:] * torch.cat(
        (x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    return x_ref.view((r.shape[0], -1))

