# -*- coding: utf-8 -*-
"""
    src.DMF
    ~~~~~~~~~~~

    @Copyright: (c) 2019-08 by Lingxi Chen (chanlingxi@gmail.com).
    @License: LICENSE_NAME, see LICENSE for more details.
"""

from __future__ import print_function
import torch
from torch import nn
import numpy as np
import os
import logging


class DMFNet(nn.Module):

    def __init__(self, M, N, K=10, L=3, decoder='cos'):
        super(DMFNet, self).__init__()
        self.M = M
        self.N = N
        self.K = K
        self.L = L

        self.U_layers = self.encoder(N, K, L)
        self.V_layers = self.encoder(M, K, L)

        self.MLP = self.encoder(2*K, 1, L)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.decoder = decoder

    def encoder(self, N, K, L):
        step = int((N - K)/L)
        prev_dim = N
        current_dim = N - step
        layers = []
        for i in range(L):
            layers.append(nn.Linear(prev_dim, current_dim))
            layers.append(nn.ReLU())
            prev_dim = current_dim
            current_dim -= step
        layers.append(nn.Linear(prev_dim, K))
        return nn.Sequential(*layers)

    def forward(self, u_x, v_x):
        u = self.U_layers(u_x)
        v = self.V_layers(v_x)
        if self.decoder == 'cos':
            y = torch.max(self.cos(u, v), torch.tensor(1e-8, device=u.device))
        else:
            x = torch.cat((u, v), 1)
            y = self.MLP(x).squeeze()

        return u, v, y


class DMF(object):

    def __init__(self, M, N, K=10, L=3,
                 learning_rate=1e-2,
                 epoches=10000,
                 device='cpu',
                 decoder='cos',
                 prefix='test'):
        self.M = M
        self.N = N
        self.K = K
        self.L = L
        self.learning_rate = learning_rate
        self.epoches = epoches
        self.device = device

        self.prefix = prefix
        self.decoder = decoder

        self.model = DMFNet(M, N, K, L, decoder=self.decoder).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.learning_rate,
                                          weight_decay=0.0001)  # append L2 penalty

        self.log_fn = os.path.join(self.prefix+'.log')
        logging.basicConfig(filename=self.log_fn, level=logging.INFO)

    def fit(self, data, n_batch, delta=0.0001, alpha=0.01):
        data = torch.tensor(data, dtype=torch.float, device=self.device)

        current_loss = 2
        prev_loss = 1
        for i in range(self.epoches):
            if i > 0:
                # break
                pass
            if current_loss - prev_loss >= 0 and current_loss - prev_loss < delta:
                break
            prev_loss = current_loss
            for j, p in enumerate(self.data_loader(data, n_batch)):
                local_u_x, local_v_x, local_y = p

                if len(local_u_x) == 0:
                    continue

                _, _, y_pred = self.model(local_u_x, local_v_x)

                y_pred, local_y = self.clean_y(y_pred, local_y)

                loss = self.loss_fun(y_pred, local_y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                current_loss = loss.item()
                logging.info('{}\t{}\t{}'.format(i, j, current_loss))

                info = 'epoch={}\tbatch={}\t{}\n'.format(i, j, current_loss)
                print(info)
            logging.info(info)

    def loss_fun(self, y_pred, y):
        if self.decoder == 'cos':
            norm_y = y / torch.max(y)
        if self.decoder == 'MLP':
            norm_y = torch.sigmoid(y)
            y_pred = torch.sigmoid(y_pred)
        losses = norm_y * torch.log(y_pred) + (1 - norm_y) * torch.log(1 - y_pred)
        loss = - torch.sum(losses)
        return loss

    def predict(self, data):
        data = torch.tensor(data, dtype=torch.float, device=self.device)
        M, N = data.shape
        n_batch = N
        U = torch.tensor([], device=self.device)
        V = torch.tensor([], device=self.device)
        Y = torch.tensor([], device=self.device)
        for i, p in enumerate(self.data_loader(data, n_batch, task='predict')):
            if i >= M:
                continue
            local_u_x, local_v_x, local_y = p
            u, v, y_pred = self.model(local_u_x, local_v_x)
            U = torch.cat((U, u[0].unsqueeze(0)))
            V = v
            Y = torch.cat((Y, y_pred.unsqueeze(0)))

        self.U = U
        self.V = torch.transpose(V, 0, 1)
        Y = Y.cpu().detach().numpy()
        return Y

    def save_U_V(self):
        U = self.U.data.cpu().detach().numpy()
        V = self.V.data.cpu().detach().numpy()
        np.savetxt(os.path.join(self.prefix + '.U'), U)
        np.savetxt(os.path.join(self.prefix + '.V'), V)
        return U, V

    def clean_y(self, y_pred, y):
        y_pred[torch.isnan(y)] = 0
        y[torch.isnan(y)] = 0
        return y_pred, y

    def data_loader(self, m, n_batch, task='fit'):
        m[torch.isnan(m)] = 0
        M, N = m.shape
        if n_batch > M*N:
            n_batch = M*N

        count = 0
        U_X = torch.zeros(n_batch, N, dtype=torch.float, device=self.device)
        V_X = torch.zeros(n_batch, M, dtype=torch.float, device=self.device)
        Y = torch.zeros(n_batch, dtype=torch.float, device=self.device)

        left = M*N
        for i in range(M):
            for j in range(N):
                U_X[count, :] = m[i, :]
                V_X[count, :] = m[:, j]
                Y[count] = m[i, j]
                count += 1
                left -= 1

                if count == n_batch:
                    yield U_X, V_X, Y
                    count = 0
                    if left < n_batch:
                        n_batch = left
                    U_X = torch.zeros(n_batch, N, dtype=torch.float, device=self.device)
                    V_X = torch.zeros(n_batch, M, dtype=torch.float, device=self.device)
                    Y = torch.zeros(n_batch, dtype=torch.float, device=self.device)

        if count > 0:
            yield U_X, V_X, Y
