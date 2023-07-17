# Module in charge of calculating different evaluation metrics
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from scipy.special import ndtr
import matplotlib.pyplot as plt
import seaborn as sns
import torch


def calculatePEHE(YActual, YEstimated):
    """
    Calculate expected Precision in Estimation of Heterogeneous Effect
    :param YActual: [Y1,Y0] actual
    :param YEstimated: [Yhat1, Yhat0], estimated potential outcomes
    :return:
    """
    PEHE_mean = np.mean(np.square((YActual[:, 1] - YActual[:, 0]) - (YEstimated[:, 1] - YEstimated[:, 0])))
    return PEHE_mean


def calculatePEHELog(YActual, YEstimated):
    """
    Calculate expected Precision in Estimation of Heterogeneous Effect
    :param YActual: [Y1,Y0] actual
    :param YEstimated: [Yhat1, Yhat0], estimated potential outcomes
    :return:
    """
    YEstimated = np.where(YEstimated > 0.5, 1, 0)
    PEHE_val = np.sqrt(
        np.mean(np.square((YActual[:, 1] - YActual[:, 0]) - (YEstimated[:, 1] - YEstimated[:, 0]))))
    return PEHE_val


def calculateATE(YActual, YEstimated):
    """
    Calculate absolute error in Average Treatment Effect
    :param YActual: [Y1,Y0] actual
    :param YEstimated: [Yhat1, Yhat0], estimated potential outcomes
    :return:
    """
    ATE = np.abs(np.mean(YActual[:, 1] - YActual[:, 0]) - np.mean(YEstimated[:, 1] - YEstimated[:, 0]))
    return ATE


def loglikelihood(y, y_hat, return_average=True):
    # region_size = 0.5
    # EPS = 1e-12

    lls = np.zeros(len(y))

    for i in range(len(y)):
        try:
            kde = gaussian_kde(dataset=y_hat[i, :])
            ll = kde.logpdf(y[i])
            # ll = kde.pdf(y[i])
        except:
            ll = np.nan

        lls[i] = ll

        # kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(y_hat[i, :].reshape(-1, 1))
        # ll = kde.score_samples(y[i].reshape(-1, 1))
        # lls.append(ll[0])

        # CCN style:
        # region = [y[i] - region_size, y[i] + region_size]
        # cdf = tuple(ndtr(np.ravel(item - kde.dataset) / kde.factor).mean()
        #             for item in region)
        # lls.append(np.log(cdf[1] - cdf[0] + EPS))

        # Visualize:
        # if i % 100 == 0:
        #     sns.histplot(y_hat[i, :], kde=True, stat='probability')
        #     # Scipy:
        #     kde = gaussian_kde(dataset=y_hat[i, :])
        #     ll_scipy = kde.logpdf(y[i])
        #     plt.plot(np.linspace(plt.gca().get_xlim()[0], plt.gca().get_xlim()[1], 50),
        #              kde.pdf(np.linspace(plt.gca().get_xlim()[0], plt.gca().get_xlim()[1], 50)),
        #              color='blue', label='Scipy')
        #     # Sklearn:
        #     kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(y_hat[i, :].reshape(-1, 1))
        #     ll_sklearn = kde.score_samples(y[i].reshape(-1, 1))
        #     plt.plot(np.linspace(plt.gca().get_xlim()[0], plt.gca().get_xlim()[1], 50), np.exp(
        #         kde.score_samples(np.linspace(plt.gca().get_xlim()[0], plt.gca().get_xlim()[1], 50).reshape(-1, 1))),
        #              color='green', label='Sklearn')
        #     plt.vlines(y[i], 0, plt.gca().get_ylim()[-1], colors='orange')
        #     plt.title('Scipy: ' + str(ll_scipy) + ' | Sklearn: ' + str(ll_sklearn))
        #     plt.legend()
        #     plt.show()

    if return_average:
        return np.mean(lls[~np.isnan(lls)])
    else:
        return lls

# Calculate Intersection-over-Union
def iou_ci(ite_l_test, ite_r_test, ite_l_pred, ite_r_pred):

    intersection = np.min((ite_r_test, ite_r_pred), axis=0) - np.max((ite_l_test, ite_l_pred), axis=0)
    intersection = np.maximum(intersection, 0)
    union = np.max((ite_r_test, ite_r_pred), axis=0) - np.min((ite_l_test, ite_l_pred), axis=0)

    iou = np.mean(intersection/union)

    assert iou >= 0, AssertionError

    return iou


# Adapted from https://github.com/clinicalml/cfrnet/blob/d38f333bff2474030529d84b84daa48d8b4a298b/cfr/evaluation.py#L108
# def pdist2(X, Y):
#     """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
#     # C = - 2 * X.dot(Y.T)
#     C = - 2 * X.matmul(Y.T)
#     nx = np.sum(np.square(X), 1, keepdims=True)
#     ny = np.sum(np.square(Y), 1, keepdims=True)
#     D = (C + ny.T) + nx
#
#     return np.sqrt(D + 1e-8)

def cf_nn(x, t):
    It = np.array(np.where(t==1))[0,:]
    Ic = np.array(np.where(t==0))[0,:]

    x_c = x[Ic,:]
    x_t = x[It,:]

    # D = pdist2(x_c, x_t)
    D = torch.cdist(x_c, x_t, 2)

    nn_t = Ic[np.argmin(D,0)]
    nn_c = It[np.argmin(D,1)]

    return nn_t, nn_c


def pehe_nn(yf_p, ycf_p, y, x, t, nn_t=None, nn_c=None):
    if nn_t is None or nn_c is None:
        nn_t, nn_c = cf_nn(x, t)

    It = np.array(np.where(t == 1))[0, :]
    Ic = np.array(np.where(t == 0))[0, :]

    ycf_t = 1.0 * y[nn_t]
    eff_nn_t = ycf_t - 1.0*y[It]
    eff_pred_t = ycf_p[It] - yf_p[It]

    eff_pred = eff_pred_t
    eff_nn = eff_nn_t

    '''
    ycf_c = 1.0*y[nn_c]
    eff_nn_c = ycf_c - 1.0*y[Ic]
    eff_pred_c = ycf_p[Ic] - yf_p[Ic]
    eff_pred = np.vstack((eff_pred_t, eff_pred_c))
    eff_nn = np.vstack((eff_nn_t, eff_nn_c))
    '''

    pehe_nn = torch.sqrt(torch.mean(torch.square(eff_pred - eff_nn)))

    return pehe_nn
