import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def plot_errors(outcomes_test, outcomes_pred):
    fig, axs = plt.subplots(1, 4, figsize=(12, 4), dpi=300)
    axs[0].set_title('ITE: Errors')
    axs[0].hist((outcomes_test[:, 0] - outcomes_test[:, 1]) - (outcomes_pred[:, 0] - outcomes_pred[:, 1]),
                zorder=0, rwidth=0.5, color='blue')
    axs[0].vlines(x=0, ymin=0, ymax=axs[0].get_ylim()[1], zorder=1, color='orange', linestyles='--')

    axs[1].set_title('PO: Errors')
    axs[1].hist((outcomes_test - outcomes_pred).flatten(), zorder=0, rwidth=0.5, color='blue')
    axs[1].vlines(x=0, ymin=0, ymax=axs[1].get_ylim()[1], zorder=1, color='orange', linestyles='--')

    axs[2].set_title('ITE: True vs Pred')
    axs[2].plot(outcomes_test[:, 0] - outcomes_test[:, 1], outcomes_pred[:, 0] - outcomes_pred[:, 1], linestyle='none',
                marker='.', alpha=0.5, color='blue')
    lims = [
        np.min([axs[2].get_xlim(), axs[2].get_ylim()]),  # min of both axes
        np.max([axs[2].get_xlim(), axs[2].get_ylim()]),  # max of both axes
    ]
    axs[2].plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    axs[2].set_xlabel('True')
    axs[2].set_ylabel('Predicted')

    axs[3].set_title('PO: True vs Pred')
    axs[3].plot(outcomes_test[:, 0], outcomes_pred[:, 0], linestyle='none', marker='.', alpha=0.5, color='red')
    axs[3].plot(outcomes_test[:, 1], outcomes_pred[:, 1], linestyle='none', marker='.', alpha=0.5, color='green')

    lims = [
        np.min([axs[3].get_xlim(), axs[3].get_ylim()]),  # min of both axes
        np.max([axs[3].get_xlim(), axs[3].get_ylim()]),  # max of both axes
    ]
    axs[3].plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    axs[3].set_xlabel('True')
    axs[3].set_ylabel('Predicted')
    plt.tight_layout()
    plt.show()


def plot_samples(yf_test, ycf_test, t_test, y0_pred, y1_pred, y0_posteriors, y1_posteriors):
    n_instances = 8
    fig, axs = plt.subplots(int(n_instances / 4), int(n_instances / 2), figsize=(12, 8), dpi=300)
    plt.gcf().suptitle('PO: Samples', fontsize=14)
    row = -1
    col = 0
    for i in range(n_instances):
        col += 1
        if i % int(n_instances / 2) == 0:
            col = 0
            row += 1
        ax = axs[row, col]

        id = np.random.randint(len(t_test))
        # z0 = torch.distributions.Normal(loc=mu0[id, :], scale=sig0[id, :]).sample([1000])
        # z1 = torch.distributions.Normal(loc=mu1[id, :], scale=sig1[id, :]).sample([1000])
        # sns.kdeplot(model.decode(z0.float()).flatten(), color='orange', ax=ax)
        # sns.kdeplot(model.decode(z1.float()).flatten(), color='blue', ax=ax)
        sns.kdeplot(y0_posteriors[id, :], color='orange', ax=ax, fill=True, alpha=0.1, label='Control')
        sns.kdeplot(y1_posteriors[id, :], color='blue', ax=ax, fill=True, alpha=0.1, label='Treated')
        # sns.kdeplot(y0_posteriors[id, :] - y1_posteriors[id, :], color='green', ax=ax, fill=True, alpha=0.2,
        #             label='ITE')

        max_y = ax.get_ylim()[1]
        ax.errorbar(x=y0_pred[id], y=1.1 * max_y,
                    xerr=[[np.percentile(y0_posteriors[id, :], 5)] - y0_pred[id],
                          [y0_pred[id] - np.percentile(y0_posteriors[id, :], 95)]],
                    color='orange', marker='x', capsize=5)
        ax.errorbar(x=y1_pred[id], y=1.2 * max_y,
                    xerr=[[np.percentile(y1_posteriors[id, :], 5)] - y1_pred[id],
                          [y1_pred[id] - np.percentile(y1_posteriors[id, :], 95)]],
                    color='blue', marker='x', capsize=5)
        # ax.errorbar(x=y0_pred[id] - y1_pred[id], y=1*max_y,
        #             xerr=[[np.percentile(y0_posteriors[id, :] - y1_posteriors[id, :], 5)] - (y0_pred[id] - y1_pred[id]),
        #                   [y0_pred[id] - y1_pred[id] - np.percentile(y0_posteriors[id, :] - y1_posteriors[id, :], 95)]],
        #             color='green', marker='x', capsize=5)
        # Add ground truth
        if t_test[id] == 0:  # Yf for mu0 - orange, Ycf for mu1 - blue
            ax.vlines(x=yf_test[id, :], ymin=0, ymax=max_y, zorder=1, color='orange', linestyles='--')
            ax.vlines(x=ycf_test[id, :], ymin=0, ymax=max_y, zorder=1, color='blue', linestyles='--')
        else:  # Yf for mu1 - blue, Ycf for mu0 - orange
            ax.vlines(x=ycf_test[id, :], ymin=0, ymax=max_y, zorder=1, color='orange', linestyles='--')
            ax.vlines(x=yf_test[id, :], ymin=0, ymax=max_y, zorder=1, color='blue', linestyles='--')
    # plt.legend()
    plt.tight_layout()
    plt.show()


def plot_samples_ite(ite_test, ite_samples_pred):
    n_instances = 8
    fig, axs = plt.subplots(int(n_instances / 4), int(n_instances / 2), figsize=(12, 8), dpi=300)
    plt.gcf().suptitle('ITE: Samples', fontsize=14)
    row = -1
    col = 0
    for i in range(n_instances):
        col += 1
        if i % int(n_instances / 2) == 0:
            col = 0
            row += 1
        ax = axs[row, col]

        id = np.random.randint(len(ite_test))
        # z0 = torch.distributions.ite_test(loc=mu0[id, :], scale=sig0[id, :]).sample([1000])
        # z1 = torch.distributions.Normal(loc=mu1[id, :], scale=sig1[id, :]).sample([1000])
        # sns.kdeplot(model.decode(z0.float()).flatten(), color='orange', ax=ax)
        # sns.kdeplot(model.decode(z1.float()).flatten(), color='blue', ax=ax)
        sns.kdeplot(ite_samples_pred[id, :], color='purple', ax=ax, fill=True, alpha=0.2, label='ITE')
        # sns.kdeplot(y0_posteriors[id, :] - y1_posteriors[id, :], color='green', ax=ax, fill=True, alpha=0.2,
        #             label='ITE')

        max_y = ax.get_ylim()[1]
        ax.errorbar(x=ite_samples_pred[id, :].mean(), y=1.1 * max_y,
                    xerr=[[np.percentile(ite_samples_pred[id, :], 5)] - ite_samples_pred[id, :].mean(),
                          [ite_samples_pred[id, :].mean() - np.percentile(ite_samples_pred[id, :], 95)]],
                    color='purple', marker='x', capsize=5)
        # ax.errorbar(x=y0_pred[id] - y1_pred[id], y=1*max_y,
        #             xerr=[[np.percentile(y0_posteriors[id, :] - y1_posteriors[id, :], 5)] - (y0_pred[id] - y1_pred[id]),
        #                   [y0_pred[id] - y1_pred[id] - np.percentile(y0_posteriors[id, :] - y1_posteriors[id, :], 95)]],
        #             color='green', marker='x', capsize=5)
        # Add ground truth
        ax.vlines(x=ite_test[id], ymin=0, ymax=max_y, zorder=1, color='purple', linestyles='--')
    # plt.legend()
    plt.tight_layout()
    plt.show()


def plot_cis(ite_test, ite_l_test, ite_r_test, ite_pred, ite_l_pred, ite_r_pred):
    for i in range(8):
        id = i      # np.random.randint(0, len(ite_test))  # No randomness for same instances across models
        plt.errorbar(x=ite_test[id], y=i + 0.2, xerr=[np.abs(ite_test[id] - ite_l_test[id]), np.abs(ite_r_test[id] - ite_test[id])], capsize=5, capthick=3,
                     marker='o', color='black')
        plt.errorbar(x=ite_pred[id], y=i, xerr=np.array([[np.abs(ite_pred[id] - ite_l_pred[id])], [np.abs(ite_r_pred[id] - ite_pred[id])]]), capsize=5, capthick=3,
                     marker='o', color='green', linestyle='--')
    plt.gca().set(ylabel=None)
    plt.gca().set_yticklabels([])
    plt.gca().set_yticks([])
    plt.tight_layout()
    plt.show()


def plot_cis_edu(itv0, itv1, M):
    fig, ax = plt.subplots()

    fig.set_size_inches(4.2, 4.2)
    plt.rcParams['axes.labelsize'] = 14

    plt.rcParams['axes.titlesize'] = 22

    plt.rcParams['xtick.labelsize'] = 10

    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 11

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    widthexp = (stats.expon.ppf(0.95, scale=0.5) - stats.expon.ppf(0.05, scale=0.5)) * (
                2 - M)  # width for exponential distribution

    bp1 = ax.boxplot(np.array(itv0)[M == 0], positions=[stats.norm.ppf(0.95) * 2], notch=True, widths=0.2,
                     patch_artist=True, boxprops=dict(facecolor="C0"))

    bp2 = ax.boxplot(np.array(itv0)[M == 1], notch=True, positions=[stats.norm.ppf(0.95)], widths=0.2,
                     patch_artist=True, boxprops=dict(facecolor="C1"))

    bp3 = ax.boxplot(np.array(itv1)[M == 0], notch=True, positions=[widthexp.max()], widths=0.2,
                     patch_artist=True, boxprops=dict(facecolor="C2"))

    bp4 = ax.boxplot(np.array(itv1)[M == 1], notch=True, positions=[widthexp.min()], widths=0.2,
                     patch_artist=True, boxprops=dict(facecolor="C3"))

    plt.plot([0, 5], [0, 5])

    plt.xlim([0.7, 4.4])
    plt.ylim([0.7, 4.4])

    ax.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0], bp4["boxes"][0]],
              ['T=0, M=0', 'T=0, M=1', 'T=1, M=0', 'T=1, M=1'], loc='upper left', frameon=False)

    plt.xticks(np.linspace(1, 4, 4), np.linspace(1, 4, 4))
    plt.yticks(np.linspace(1, 4, 4), np.linspace(1, 4, 4))

    plt.xlabel('True Interval Width')
    plt.ylabel('Estimated Interval Width')

    plt.show()

def plot_cis_edu(itv0, itv1, S):
    fig, ax = plt.subplots()

    fig.set_size_inches(4.2, 4.2)
    plt.rcParams['axes.labelsize'] = 14

    plt.rcParams['axes.titlesize'] = 22

    plt.rcParams['xtick.labelsize'] = 10

    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 11

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    widthexp = (stats.expon.ppf(0.95, scale=0.5) - stats.expon.ppf(0.05, scale=0.5)) * (
                2 - S)  # width for exponential distribution

    bp1 = ax.boxplot(np.array(itv0)[S == 0], positions=[stats.norm.ppf(0.95) * 2], notch=True, widths=0.2,
                     patch_artist=True, boxprops=dict(facecolor="C0"))

    bp2 = ax.boxplot(np.array(itv0)[S == 1], notch=True, positions=[stats.norm.ppf(0.95)], widths=0.2,
                     patch_artist=True, boxprops=dict(facecolor="C1"))

    bp3 = ax.boxplot(np.array(itv1)[S == 0], notch=True, positions=[widthexp.max()], widths=0.2,
                     patch_artist=True, boxprops=dict(facecolor="C2"))

    bp4 = ax.boxplot(np.array(itv1)[S == 1], notch=True, positions=[widthexp.min()], widths=0.2,
                     patch_artist=True, boxprops=dict(facecolor="C3"))

    plt.plot([0, 5], [0, 5])

    plt.xlim([0.7, 4.4])
    plt.ylim([0.7, 4.4])

    ax.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0], bp4["boxes"][0]],
              ['T=0,S=0', 'T=0,S=1', 'T=1,S=0', 'T=1,S=1'], loc='upper left', frameon=False)

    plt.xticks(np.linspace(1, 4, 4), np.linspace(1, 4, 4))
    plt.yticks(np.linspace(1, 4, 4), np.linspace(1, 4, 4))

    plt.xlabel('True Interval Width')
    plt.ylabel('Estimated Interval Width')

    plt.show()
