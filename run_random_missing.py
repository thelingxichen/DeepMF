import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
# import DeepMF
import DMF
import simu_impute
import timeit
sns.set()


def main():

    M, N = int(sys.argv[1]), int(sys.argv[2])
    data_type = sys.argv[3]
    if not os.path.exists('random_missing/{}_{}'.format(M, N)):
        os.makedirs('random_missing/{}_{}'.format(M, N))

    if data_type == 'a1':
        o_prefix = 'random_missing/{}_{}/a1'.format(M, N)
        o_data = simu_impute.generate_a1(M, N)
    elif data_type == 'a2':
        o_prefix = 'random_missing/{}_{}/a2'.format(M, N)
        o_data = simu_impute.generate_a2(M, N)
    elif data_type == 'a3':
        o_prefix = 'random_missing/{}_{}/a3'.format(M, N)
        o_data = simu_impute.generate_a3(M, N)
    elif data_type == 'a4':
        o_prefix = 'random_missing/{}_{}/a4'.format(M, N)
        o_data = simu_impute.generate_a4(M, N)
    elif data_type == 'a4T':
        o_prefix = 'random_missing/{}_{}/a4T'.format(M, N)
        o_data = simu_impute.generate_a4(M, N)
        o_data = o_data.T
        M, N = o_data.shape
    else:
        print('wrong data type')
        return
    np.savetxt('{}.txt'.format(o_prefix), o_data)
    o_data = np.sin(o_data) * np.cos(o_data) * np.tan(o_data)

    # random_missing
    for rate in [0]:
        data = simu_impute.random_missing(o_data, rate=rate)
        prefix = '{}_{}'.format(o_prefix, rate)
        np.savetxt('{}.txt'.format(prefix), data)

        start = timeit.default_timer()
        run(M, N, o_data, data, prefix)
        stop = timeit.default_timer()
        print('Time: ', stop - start)


def run(M, N, o_data, data, prefix):
    device = torch.device("cuda")
    K = 3
    L = 2   # L = 2 for 1000x600
    n_batch = 10000

    epoches = 2000
    '''
    model = DeepMF.DeepMF(M, N, K=K, L=L,
                          learning_rate=1e-2,
                          epoches=epoches, min_s=0.1,
                          device=device, problem='regression', data_type='impute', prefix=prefix)
    '''
    model = DMF.DMF(M, N, K=K, L=L,
                    learning_rate=1e-3,
                    epoches=epoches,
                    decoder='MLP',
                    device=device, prefix=prefix)

    model.fit(data, n_batch, alpha=0.01)

    y_pred = model.predict(data)
    np.savetxt('{}.pred'.format(prefix), y_pred)
    U, V = model.save_U_V()

    plt.figure()
    plt.subplot(5, 1, 1)
    sns.heatmap(o_data, cmap="rainbow")
    plt.subplot(5, 1, 2)
    sns.heatmap(data, cmap="rainbow")
    plt.subplot(5, 1, 3)
    print(y_pred.shape)
    sns.heatmap(y_pred, cmap="rainbow")
    plt.subplot(5, 1, 4)
    sns.heatmap(U, cmap="rainbow")
    plt.subplot(5, 1, 5)
    sns.heatmap(V, cmap="rainbow")
    plt.savefig('{}.png'.format(prefix))


if __name__ == "__main__":
    main()
