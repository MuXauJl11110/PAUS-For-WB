import pickle

import click
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import ot
import scipy
from IPython import display
from numpy import linalg as LA
from scipy.sparse import csr_matrix

from methods.paus import PAUS
from oracles.oracle import OperatorOracle
from oracles.point import OTProblemOracle
from utils.dataset import get_gaussian
from utils.tm import get_1d_tm


def mirror_ver_distributed_star(data, bartrue, C, numItermax=100000, stopThr=1e-9, verbose=True):
    m = data.shape[0]  # number of measures
    n = data.shape[1]  # img_size**

    G = nx.star_graph(m - 1)
    barW = nx.laplacian_matrix(G)
    W = scipy.sparse.kron(barW, np.eye(n))

    d = C.flatten()  # vectorized cost matrix
    dnorm = LA.norm(d, np.inf)

    # D = np.repeat(d, m)
    # Maxcol2 = lambda x: np.max(np.sum(np.abs(x)**2,axis=-1))

    loss_fidel = []
    cpt_his = []

    loss_history = []
    time_history = []

    Wass = 0
    for i in range(data.shape[0]):
        Wass += ot.emd2(bartrue, data[i], C)  # exact linear program
    Wass = Wass / data.shape[0]

    eig = np.linalg.eigvals(barW.A)
    eigmin = min(i for i in eig if i > 0.00001)
    eigmax = max(eig)
    chi = eigmax / eigmin

    # define algorithm constants
    gamma = np.sqrt(8) * dnorm / (eigmax)
    eta = 1 / (8 * m * dnorm * np.sqrt(6 * n * np.log(n) * dnorm * chi))

    alpha = 4 * n * eta * (dnorm**2) * chi * m
    beta = 6 * eta * np.log(n) * dnorm * m
    kappa = 3 * eta * np.log(n) * m
    theta = 2 * n * eta * dnorm * chi * m
    #     beta =  6*dnorm*eta*np.log(n)
    #     gamma= 3*eta*np.log(n)

    # define inputs
    x = np.ones((m, n**2)) / (n**2)
    u = np.copy(x)

    p = np.ones((m, n)) / n

    s = np.copy(p)

    z = np.zeros(m * 2 * n)
    v = np.copy(z)

    y = np.zeros(m * n)
    lm = np.copy(y)

    shat = np.copy(p)

    # edge-incidence matrix

    barA = np.zeros((2 * n, n**2))
    i = 0
    j = 0
    for i in range(n):
        barA[i, j : j + n] = 1
        j += n
    t = 0
    for i in range(n, 2 * n):
        for j in range(t, n**2, n):
            barA[i, j] = 1
        t += 1

    Abar = csr_matrix(barA)

    A = scipy.sparse.kron(np.eye(m), barA)

    # Algorithm
    err = 1
    cpt = 0

    ind = np.hstack((np.ones(n), np.zeros(n)))
    ind = np.tile(ind, m)

    while err > stopThr and cpt < numItermax:
        cpt += 1
        # for k in range(0, numItermax):

        b = []
        for i in range(m):
            b = np.hstack((b, p[i], data[i]))

            sft = x[i] * np.exp(-kappa * (d + 2 * dnorm * (barA.transpose().dot(z.reshape(m, 2 * n)[i]))))
            u[i] = sft / np.sum(sft)  # variable for x

            sft2 = p[i] * np.exp(
                beta * z[ind > 0].reshape(m, n)[i] - 3 * eta * np.log(n) * gamma * m * ((W.dot(y)).reshape(m, n)[i])
            )
            s[i] = sft2 / np.sum(sft2)  # variable for p

        v = z + alpha * (A.dot(x.reshape(m * n**2)) - b)  # variable for y

        flag = np.abs(v) > 1
        v[flag] = np.sign(v[flag])

        lm = y + theta * gamma * W.dot(p.reshape(m * n))

        # pold = np.copy(shat)

        # solutions
        shat += s

        b = []
        for i in range(m):
            b = np.hstack((b, s[i], data[i]))

            sft = x[i] * np.exp(-kappa * (d + 2 * dnorm * (barA.transpose().dot(v.reshape(m, 2 * n)[i]))))
            x[i] = sft / np.sum(sft)  # variable for x

            sft2 = p[i] * np.exp(
                beta * v[ind > 0].reshape(m, n)[i] - 3 * eta * np.log(n) * gamma * m * ((W.dot(lm)).reshape(m, n)[i])
            )
            p[i] = sft2 / np.sum(sft2)  # variable for p

        z = z + alpha * (A.dot(u.reshape(m * n**2)) - b)  # variable for y

        flag = np.abs(z) > 1
        z[flag] = np.sign(z[flag])

        y = y + theta * gamma * W.dot(s.reshape(m * n))

        # solutions
        phat = shat / (cpt + 1)

        Wass_c = 0
        for i in range(data.shape[0]):
            Wass_c += ot.emd2(phat[i], data[i], C)
        Wass_c = Wass_c / data.shape[0]

        loss = Wass_c - Wass
        loss_history.append(loss)

        loss_fi = LA.norm(W.dot(phat.reshape(m * n)))
        loss_fidel.append(loss_fi)

        cpt_his.append(cpt)

        if verbose and cpt % 1000 == 0:
            # Visualize
            # display.clear_output(wait=True)
            # plt.figure(figsize=(6, 4), dpi=300)
            # plt.plot(np.linspace(-10, 10, 100), phat[0])
            # plt.show()

            print("Pass {} iterations".format(cpt), flush=True)
            # with open('WBimages/Distr_star/iter{}.npy'.format(cpt), 'wb') as f:
            #     np.save(f, phat[0])
            print("$\|Wp\|_2=$", loss_fi)
            print("function loss is", loss)
            print("emd2", Wass_c)
    #             with open('WBimages/Distr_time_star/iter{}.npy'.format(cpt), 'wb') as f:
    #                    np.save(f, cpt_his)
    #             with open('WBimages/Distr_loss_star/iter{}.npy'.format(cpt), 'wb') as f:
    #                    np.save(f, loss_fidel)
    #             with open('WBimages/Distr_loss_func_star/iter{}.npy'.format(cpt), 'wb') as f:
    # np.save(f, loss_history)
    #  Output

    return phat[0], cpt_his, loss_history, loss_fidel, chi


@click.command()
@click.option("--d", default=100, help="Support len.")
@click.option("--t", default=10, help="Number of gaussians.")
@click.option("--n", default=5, help="Number of gaussians on the server.")
def run_gaus(d: int, t: int, n: int):
    try:
        with open(f"reports/oracles/F_d={d}_T={t}_n={n}.pkl", "rb") as f:
            F = pickle.load(f)
        with open(f"reports/oracles/F1_d={d}_T={t}_n={n}.pkl", "rb") as f:
            F1 = pickle.load(f)
        with open(f"reports/oracles/bar_true_d={d}_T={t}_n={n}.pkl", "rb") as f:
            bartrue = pickle.load(f)
    except FileNotFoundError:
        data, bartrue = get_gaussian(d, t)
        C = get_1d_tm(d)
        oracles = [OTProblemOracle(data[i], C) for i in range(t)]
        F, F1 = OperatorOracle(oracles, d, t), OperatorOracle(oracles, d, n)
        with open(f"reports/oracles/F_d={d}_T={t}_n={n}.pkl", "wb") as f:
            pickle.dump(F, f)
        with open(f"reports/oracles/F1_d={d}_T={t}_n={n}.pkl", "wb") as f:
            pickle.dump(F1, f)
        with open(f"reports/oracles/bar_true_d={d}_T={t}_n={n}.pkl", "wb") as f:
            pickle.dump(bartrue, f)
    C = get_1d_tm(d)

    numItermax = 10000
    gaus = np.array([oracle.q for oracle in F._oracles])

    # Star Graph
    output = mirror_ver_distributed_star(gaus, bartrue, C, numItermax, verbose=True)

    with open(
        f"reports/methods/mirror_prox_output_d={d}_T={t}_n={n}_step_scale={step_scale}_max_iter={max_iter}.pkl",
        "wb",
    ) as f:
        pickle.dump(output, f)


if __name__ == "__main__":
    np.random.seed(30)
    run_gaus()
