# -*- coding: utf-8 -*-
"""
    src.DeepMF
    ~~~~~~~~~~~

    @Copyright: (c) 2018-04 by Lingxi Chen (chanlingxi@gmail.com).
    @License: LICENSE_NAME, see LICENSE for more details.
"""

from __future__ import print_function
import torch
import torch.nn
from torch.nn import init
from torch.nn.parameter import Parameter
from torch._jit_internal import weak_module, weak_script_method
from sklearn.decomposition import FastICA
import math
import numpy as np
import os
import logging
from ray.kmean_torch import kmeans_core

torch.manual_seed(2019)


@weak_module
class SparseLinear(torch.nn.Module):

    _constants__ = ['bias']

    def __init__(self, in_features, out_features, bias=True):
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    @weak_script_method
    def forward(self, input):

        if input.dim() == 2 and self.bias is not None:
            # fused op is marginally faster
            ret = torch.sparse.addmm(self.bias, input, self.weight.t())
        else:
            output = torch.sparse.mm(input, self.weight.t())
            if self.bias is not None:
                output += self.bias
            ret = output
        return ret

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class DeepMF(object):

    def __init__(self, M, N, K=10, L=3,
                 learning_rate=1e-2,
                 min_s=0.2,
                 epoches=10000,
                 device='cpu',
                 neighbor_proximity='Lap',
                 problem='regression',
                 data_type='hic',
                 prefix='test',
                 seed=2019):

        self.M = M
        self.N = N
        self.K = K
        self.L = L
        self.learning_rate = learning_rate
        self.min_s = min_s
        self.epoches = epoches
        self.device = device
        self.neighbor_proximity = neighbor_proximity
        self.problem = problem

        self.prefix = prefix

        if self.problem == 'regression':
            self.loss_fun = torch.nn.MSELoss()
        else:
            self.loss_fun = torch.nn.BCELoss()

        self.data_type = data_type

        layers = [SparseLinear(M, K)]
        for i in range(L):
            layers.append(torch.nn.Linear(K, K))
            if L > 1:
                layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(K, N))
        if self.problem == 'classification':
            layers.append(torch.nn.Sigmoid())

        self.model = torch.nn.Sequential(*layers).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.learning_rate,
                                          weight_decay=0.0001)  # append L2 penalty

        self.log_fn = os.path.join(self.prefix+'.log')
        logging.basicConfig(filename=self.log_fn, level=logging.INFO)

    def get_similarity(self, data, row_or_col, n=15):
        self.data_isnan = torch.isnan(data)
        if torch.any(self.data_isnan):
            D = self.get_distance_with_nan(data, self.data_isnan, row_or_col)
        else:
            D = self.get_distance(data, row_or_col)
        S = 1 / (1 + D)
        N, _ = D.shape
        if not n:
            n = int(N / self.K)
        cutoff = D[range(N), torch.argsort(D, dim=1)[:, n-1]]
        Indicator = torch.zeros(N, N, dtype=torch.float, device=self.device)
        for i in range(N):
            for j in range(N):
                if D[i, j] <= cutoff[i] or D[i, j] <= cutoff[j]:
                    Indicator[i, j] = 1
        # sigma = D.std()/2
        # S = torch.exp(-D/(2*sigma*sigma))
        S = Indicator * S
        return S, D

    def get_distance(self, data, row_or_col):
        M, N = data.shape
        if row_or_col == 'row':
            G = torch.mm(data, data.t())
        else:
            G = torch.mm(data.t(), data)
        g = torch.diag(G)
        if row_or_col == 'row':
            x = g.repeat(M, 1)
        else:
            x = g.repeat(N, 1)
        D = x + x.t() - 2*G
        return D

    def get_distance_with_nan(self, o_data, data_isnan, row_or_col):
        data = o_data.clone().detach()
        data[torch.isnan(data)] = 0
        D = self.get_distance(data, row_or_col)
        M, N = data.shape
        if row_or_col == 'row':
            row_or_col_nan = torch.sum(data_isnan, dim=1).repeat(M, 1)
            penalty = torch.max(data) * 1.0 / N
        else:
            row_or_col_nan = torch.sum(data_isnan, dim=0).repeat(N, 1)
            penalty = torch.max(data) * 1.0 / M
        if penalty == 0:
            penalty = 1
        nan_penalty = penalty * (row_or_col_nan + row_or_col_nan.t())
        nan_penalty = nan_penalty.float()
        D = D + nan_penalty - torch.diag(torch.diag(nan_penalty))

        return D

    def mask_weak_linkage(self, S):
        m, n = S.shape
        s = S.view(m*n, -1)
        nan_count = torch.sum(self.data_isnan).float()
        a, b = self.data_isnan.shape
        nan_rate = nan_count*1.0 / (a*b)
        if nan_rate > 0.3:
            k = 2
        else:
            k = 3
        km = kmeans_core(k, s, all_cuda=True)
        km.run()
        idxs = km.idx.view(m, n)
        _, idx = torch.min(km.cent, 0)
        S[idxs == idx[0]] = 0

        return S

    def fit(self, data, n_batch, delta=0.00001, alpha=0.01):
        data = torch.tensor(data, dtype=torch.float, device=self.device)
        Su, Du = self.get_similarity(data, 'row')
        Sv, Dv = self.get_similarity(data, 'col')

        # Su = self.mask_weak_linkage(Su)
        # Sv = self.mask_weak_linkage(Sv)

        # self.initial_U_V(data)

        first_loss = 0
        current_loss = 2
        prev_loss = 1
        for i in range(self.epoches):
            if i > 0:
                pass
            if current_loss - prev_loss >= 0 and current_loss - prev_loss < delta:
                break
            prev_loss = current_loss
            for j, p in enumerate(self.data_loader(data, n_batch)):
                local_x, local_y = p

                if len(local_x) == 0:
                    continue

                # local_x = local_x.to_dense()

                y_pred = self.model(local_x)
                y_pred, local_y = self.clean_y(y_pred, local_y)
                loss = self.loss_fun(y_pred, local_y)
                first_loss = loss.item()

                u_loss, v_loss = self.U_V_loss(Su, Sv, Du, Dv)
                loss = (1 - alpha) * loss + alpha * (u_loss + v_loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                current_loss = loss.item()
                logging.info('{}\t{}\t{}'.format(i, j, current_loss))
            info = 'epoch={}\tbatch={}\t{}\t{}\t{}\t{}'.format(i, j,
                                                               current_loss,
                                                               (1 - alpha) * first_loss,
                                                               alpha*u_loss,
                                                               alpha*v_loss)
            print(info)
            logging.info(info)
        return y_pred

    def predict(self, data):
        data = torch.tensor(data, dtype=torch.float, device=self.device)
        M, N = data.shape
        n_batch = N
        Y = torch.tensor([], device=self.device)
        for i, p in enumerate(self.data_loader(data, n_batch, task='predict')):
            if i >= M:
                continue
            local_x, local_y = p
            y_pred = self._predict(local_x)
            Y = torch.cat((Y, y_pred))

        Y = Y.cpu().detach().numpy()
        return Y

    def _predict(self, X):
        y_pred = self.model(X)
        if self.problem == 'classification':
            y_pred[y_pred <= 0.5] = 0
            y_pred[y_pred > 0.5] = 1
        return y_pred

    def U_V_loss(self, Su, Sv, Du, Dv):
        U, V = self.load_U_V()

        if self.neighbor_proximity == 'Lap':
            u_loss = self._lap_loss(U, Su, 'row')
            v_loss = self._lap_loss(V, Sv, 'col')
        elif self.neighbor_proximity == 'MSE':
            u_loss = torch.sum(torch.pow(self.get_distance(U, 'row') - Du, 2))/(self.M*self.M)*0.01
            v_loss = torch.sum(torch.pow(self.get_distance(V, 'col') - Dv, 2))/(self.N*self.N)*0.01
        elif self.neighbor_proximity == 'KL':
            u_loss = self._kl_loss(1/(1+Du), 1/(1+self.get_distance(U, 'row')))
            v_loss = self._kl_loss(1/(1+Dv), 1/(1+self.get_distance(V, 'col')))
        else:
            u_loss = 0
            v_loss = 0

        return u_loss, v_loss

    def _kl_loss(self, P, Q):
        # return torch.sum(P*torch.log2(P/Q))

        return torch.log(torch.sum((P*P/Q)-P+Q))

    def _lap_loss(self, W, Sw, row_or_col):
        return torch.sum(self.get_distance(W, row_or_col)*Sw)

    def initial_U_V(self, data):
        model = FastICA(n_components=self.K)

        U = model.fit_transform(np.nan_to_num(data))
        U = torch.tensor(U, dtype=torch.float, device=self.device)

        V = model.fit_transform(np.nan_to_num(data).T)
        V = torch.tensor(V, dtype=torch.float, device=self.device)

        self.model[0].weight.data = U.t()
        self.model[-1].weight.data = V

    def load_U_V(self):
        U = self.model[0].weight.t()
        V = self.model[-1].weight.t()
        return U, V

    def save_U_V(self):
        U, V = self.load_U_V()
        U = U.data.cpu().detach().numpy()
        V = V.data.cpu().detach().numpy()
        np.savetxt(os.path.join(self.prefix + '.U'), U)
        np.savetxt(os.path.join(self.prefix + '.V'), V)
        return U, V

    def clean_y(self, y_pred, y):
        y_pred[torch.isnan(y)] = 0
        y[torch.isnan(y)] = 0
        return y_pred, y

    def data_loader(self, m, n_batch, task='fit'):
        M, N = m.shape
        idx = []
        for i in range(M):
            idx.append(i)

            if (i + 1) % n_batch == 0:
                Xi = torch.tensor([range(n_batch), idx], dtype=torch.long, device=self.device)
                Xv = torch.ones(n_batch, dtype=torch.float, device=self.device)
                X = torch.sparse.FloatTensor(Xi, Xv, torch.Size([n_batch, M]))
                Y = m[idx, :]
                yield X, Y
                idx = []

        if idx:
            Xi = torch.tensor([range(len(idx)), idx], dtype=torch.long, device=self.device)
            Xv = torch.ones(len(idx), dtype=torch.float, device=self.device)
            X = torch.sparse.FloatTensor(Xi, Xv, torch.Size([len(idx), M]))
            Y = m[idx, :]
            yield X, Y
