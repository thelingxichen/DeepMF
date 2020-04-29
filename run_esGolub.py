import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import DMF
import simu_impute
sns.set()


def main():

    rate = float(sys.argv[1])
    nrun = sys.argv[2]
    device = 'cpu'

    data = np.genfromtxt('../esGolub/esGolub.csv', delimiter=',')[1:, 1:]
    if not os.path.exists('esGolub'):
        os.makedirs('esGolub')
    prefix = 'esGolub/esGolub_{}_{}'.format(rate, nrun)

    fn = 'esGolub/esGolub_{}.txt'.format(rate)
    if os.path.isfile(fn):
        data = np.genfromtxt(fn)
    else:
        data = simu_impute.random_missing(data, rate=rate)
        np.savetxt(fn, data)

    M, N = data.shape
    K = 3

    run(M, N, K, data, device, prefix)


def run(M, N, K, data, device, prefix):
    device = torch.device(device)
    L = 3
    n_batch = 10000

    epoches = 1000
    '''
    model = DeepMF.DeepMF(M, N, K=K, L=L,
                          learning_rate=1e-2,
                          epoches=epoches,
                          device=device, problem='regression', data_type='impute', prefix=prefix)
    '''
    model = DMF.DMF(M, N, K=K, L=L,
                    learning_rate=1e-3,
                    epoches=epoches,
                    decoder='MLP',
                    device=device, prefix=prefix)

    model.fit(data, n_batch, delta=0.0001, alpha=0.01)
    y_pred = model.predict(data)

    U, V = model.save_U_V()

    np.savetxt('{}.pred'.format(prefix), y_pred)

    plt.figure()
    plt.subplot(2, 2, 1)
    sns.heatmap(data, cmap="rainbow", xticklabels=False, yticklabels=False)
    plt.subplot(2, 2, 2)
    sns.heatmap(y_pred, cmap="rainbow", xticklabels=False, yticklabels=False)

    plt.subplot(2, 2, 3)
    sns.heatmap(U, cmap="rainbow", xticklabels=False, yticklabels=False)
    plt.subplot(2, 2, 4)
    sns.heatmap(V, cmap="rainbow", xticklabels=False, yticklabels=False)
    plt.savefig('{}.png'.format(prefix))

    plt.figure()


if __name__ == "__main__":
    main()
