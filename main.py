import argparse
import numpy as np
import torch
import pytorch_lightning as pl
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem, truncnorm, norm
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger
from data_loading import DataModule, IHDP, LBIDD, News, Synthetic, SyntheticGMM, SyntheticCI, Twins, EDU
from noflite import Noflite
from cmgp import CMGP
from benchmarks.cevae import cevae
from benchmarks.ganite.ganite import ganite, ganite_prob
from benchmarks.fccn import FCCN
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from metrics import calculatePEHE, calculateATE, loglikelihood, iou_ci, pehe_nn
from visualize import plot_errors, plot_samples, plot_samples_ite, plot_cis, plot_cis_edu



if __name__ == '__main__':
    # Hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=5e-4)                    # IHDP: 5e-4; EDU: 5e-4; News: 5e-4
    parser.add_argument('--lambda_l1', type=float, default=1e-3)             # L1; IHDP: 1e-3; EDU: 0; News: 5e-3
    parser.add_argument('--lambda_l2', type=float, default=5e-3)             # L2; IHDP: 1e-4; EDU: 1e-3; News: 1e-5
    parser.add_argument('--batch_size', type=int, default=128)               # IHDP: 128; EDU: 512; News: 64/128
    # Encoder parameters
    parser.add_argument('--hidden_neurons_encoder', type=int, default=32)    # Balancer; IHDP: 25; EDU: 25
    parser.add_argument('--hidden_layers_balancer', type=int, default=3)     # Layers balancer (min = 1)
    parser.add_argument('--hidden_layers_encoder', type=int, default=0)      # Layers encoder (after balancer; min = 0)
    parser.add_argument('--hidden_layers_prior', type=int, default=2)        # Layers cond (after encoder; min = 1)
    # Flow parameters
    parser.add_argument('--n_flows', type=int, default=1)                    # IHDP: 0; EDU: 4
    parser.add_argument('--flow_type', type=str, default='SigmoidX')  # Sigmoid (None, X, XT, T), GF, RQNSF-AR, Residual
    parser.add_argument('--hidden_neurons_trans', type=int, default=4)       # Flow transformer (Sigmoid/Resid/RQNSF-AR)
    parser.add_argument('--dense', type=bool, default=False, action=argparse.BooleanOptionalAction)  # Sigmoid transform
    parser.add_argument('--hidden_neurons_cond', type=int, default=16)       # Flow conditioner (Sigmoid)
    parser.add_argument('--hidden_layers_cond', type=int, default=2)         # Flow conditioner (Sigmoid; min=1)
    parser.add_argument('--datapoint_num', type=int, default=8)              # For Gaussianization flow
    parser.add_argument('--resid_layers', type=int, default=1)               # For Residual/RQNSF-AR
    # Training parameters
    parser.add_argument('--noise_reg_x', type=float, default=1e-0)              # IHDP: 0; News: 1e-0
    parser.add_argument('--noise_reg_y', type=float, default=5e-1)              # IHDP: 5e-1; News: 5e-1
    parser.add_argument('--lambda_mmd', type=float, default=0.1)             # IHDP: 1; EDU: 0.1
    parser.add_argument('--max_steps', type=int, default=10000)               # IHDP: 5k; EDU: 2.5k; News: 10k
    parser.add_argument('--n_samples', type=int, default=500)
    parser.add_argument('--trunc_prob', type=float, default=0.01)
    parser.add_argument('--metalearner', type=str, default='T')              # S or T
    # Benchmarks
    parser.add_argument('--NOFLITE', type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--CMGP', type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--CEVAE', type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--GANITE', type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--FCCN', type=bool, default=False, action=argparse.BooleanOptionalAction)
    # Experimental
    parser.add_argument('--wandb', type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--sweep', type=bool, default=False, action=argparse.BooleanOptionalAction)  # For tuning
    parser.add_argument('--visualize', type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--dataset', type=str, default='IHDP')  # Synthetic, IHDP, EDU, News, LBIDD
    parser.add_argument('--iterations', type=int, default=1)
    parser_args = parser.parse_args()

    # Load in params
    params = {
        'lr': parser_args.lr,
        'lambda_l1': parser_args.lambda_l1,
        'lambda_l2': parser_args.lambda_l2,
        'batch_size': parser_args.batch_size,
        'noise_reg_x': parser_args.noise_reg_x,
        'noise_reg_y': parser_args.noise_reg_y,
        'hidden_neurons_encoder': parser_args.hidden_neurons_encoder,
        'hidden_layers_balancer': parser_args.hidden_layers_balancer,
        'hidden_layers_encoder': parser_args.hidden_layers_encoder,
        'hidden_layers_prior': parser_args.hidden_layers_prior,
        'hidden_neurons_trans': parser_args.hidden_neurons_trans,
        'hidden_neurons_cond': parser_args.hidden_neurons_cond,
        'hidden_layers_cond': parser_args.hidden_layers_cond,
        'dense': parser_args.dense,
        'n_flows': parser_args.n_flows,
        'datapoint_num': parser_args.datapoint_num,
        'resid_layers': parser_args.resid_layers,
        'max_steps': parser_args.max_steps,
        'flow_type': parser_args.flow_type,
        'metalearner': parser_args.metalearner,
        'lambda_mmd': parser_args.lambda_mmd,
        'n_samples': parser_args.n_samples,
        'trunc_prob': parser_args.trunc_prob,
        'dataset': parser_args.dataset,  # IHDP, Twins, News, Synthetic, SyntheticGMM, SyntheticCI
        'bin_outcome': True if parser_args.dataset == 'Twins' else False,
        'iterations': parser_args.iterations,
        'visualize': parser_args.visualize,
        'sweep': parser_args.sweep,
        'wandb': parser_args.wandb,
        'NOFLITE': parser_args.NOFLITE,
        'CMGP': parser_args.CMGP,
        # 'BART': parser_args.BART,
        'CEVAE': parser_args.CEVAE,
        'GANITE': parser_args.GANITE,
        'FCCN': parser_args.FCCN,
    }

    print('\nParameters:    ', params)

    # Seed everything to ensure full reproducibility (if not doing a sweep)
    if not params['sweep']:
        seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load in the data:
    if params['dataset'] == 'IHDP':
        dataset = IHDP      # 0 - 99 iterations
        iterator = range(params['iterations'])
    elif params['dataset'] == 'EDU':
        dataset = EDU       # 0 - 9
        iterator = range(params['iterations'])
    elif params['dataset'] == 'LBIDD':
        dataset = LBIDD
        iterator = range(params['iterations'])
    elif params['dataset'] == 'News':
        dataset = News      # 1 - 50 | Max 50
        iterator = range(1, 1 + params['iterations'])
    elif params['dataset'] == 'Synthetic':
        dataset = Synthetic
        iterator = range(params['iterations'])
    elif params['dataset'] == 'Twins':
        dataset = Twins
        iterator = range(params['iterations'])
    elif params['dataset'] == 'SyntheticGMM':
        dataset = SyntheticGMM
        iterator = range(1)
    elif params['dataset'] == 'SyntheticCI':
        dataset = SyntheticCI
        iterator = range(params['iterations'])

    # Initialize result arrays:
    if params['CMGP']:
        pehe_cmgp = np.zeros(iterator.__len__())
        ll_cmgp = np.zeros(iterator.__len__())
        if params['dataset'] == 'SyntheticCI' or params['dataset'] == 'EDU' or params['dataset'] == 'IHDP':
            iou_cmgp = np.zeros(iterator.__len__())
            accuracy_cmgp = np.zeros(iterator.__len__())
        coverage_cmgp = np.zeros(iterator.__len__())
    if params['CEVAE']:
        pehe_cevae = np.zeros(iterator.__len__())
        ll_cevae = np.zeros(iterator.__len__())
        if params['dataset'] == 'SyntheticCI' or params['dataset'] == 'EDU' or params['dataset'] == 'IHDP':
            iou_cevae = np.zeros(iterator.__len__())
            accuracy_cevae = np.zeros(iterator.__len__())
        coverage_cevae = np.zeros(iterator.__len__())
    if params['GANITE']:
        pehe_ganite = np.zeros(iterator.__len__())
        ll_ganite = np.zeros(iterator.__len__())
        if params['dataset'] == 'SyntheticCI' or params['dataset'] == 'EDU' or params['dataset'] == 'IHDP':
            iou_ganite = np.zeros(iterator.__len__())
            accuracy_ganite = np.zeros(iterator.__len__())
        coverage_ganite = np.zeros(iterator.__len__())
    if params['FCCN']:
        pehe_fccn = np.zeros(iterator.__len__())
        ll_fccn = np.zeros(iterator.__len__())
        if params['dataset'] == 'SyntheticCI' or params['dataset'] == 'EDU' or params['dataset'] == 'IHDP':
            iou_fccn = np.zeros(iterator.__len__())
            accuracy_fccn = np.zeros(iterator.__len__())
        coverage_fccn = np.zeros(iterator.__len__())
    if params['NOFLITE']:
        pehe_noflite = np.zeros(iterator.__len__())
        ll_noflite = np.zeros(iterator.__len__())
        if params['dataset'] == 'SyntheticCI' or params['dataset'] == 'EDU' or params['dataset'] == 'IHDP':
            iou_noflite = np.zeros(iterator.__len__())
            accuracy_noflite = np.zeros(iterator.__len__())
        coverage_noflite = np.zeros(iterator.__len__())

    for iter in iterator:
        print('\nIteration ' + str(iter))
        if params['dataset'] == 'News':
            iter = iter - 1

        # Load data:
        if params['dataset'] == 'IHDP' or params['dataset'] == 'EDU' or params['dataset'] == 'News':
            dm = DataModule(dataset, iteration=iter, batch_size=params['batch_size'])
        else:
            dm = DataModule(dataset, iteration=None, batch_size=params['batch_size'])

        params['input_size'] = dm.dataset.input_size

        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        train_val_loader = dm.train_val_dataloader()
        test_loader = dm.test_dataloader()

        # Use entire training + validation set in the end:
        x_train, yf_train, t_train = dm.dataset.get_train_val_data()
        x_val, yf_val, t_val = dm.dataset.get_val_data()
        x_test, yf_test, ycf_test, t_test, ite_sample_test = dm.dataset.get_test_data()

        # Define the potential outcomes
        outcomes_test = np.zeros_like(np.hstack((yf_test, ycf_test)))
        outcomes_test[t_test == 0, 0] = yf_test[t_test == 0, 0]
        outcomes_test[t_test == 0, 1] = ycf_test[t_test == 0, 0]
        outcomes_test[t_test == 1, 1] = yf_test[t_test == 1, 0]
        outcomes_test[t_test == 1, 0] = ycf_test[t_test == 1, 0]
        ite_mu_test = outcomes_test[:, 1] - outcomes_test[:, 0]

        # Get optimal treatment (for given utility) to calculate treatment accuracy for IHDP and News:
        if params['dataset'] == 'EDU' or params['dataset'] == 'IHDP':
            opt_treatment = dm.opt_treatment

        # Benchmarks:
        # CMGP:
        if params['CMGP']:
            print('\n---- CMGP ---------------------')

            # For News, take first 50 principal components:
            if params['dataset'] == 'News':
                pca = PCA(n_components=100)
                x_train_cmgp = pca.fit_transform(x_train)
                x_test_cmgp = pca.transform(x_test)
            else:
                x_train_cmgp = x_train
                x_test_cmgp = x_test

            cmgp = CMGP(x_train_cmgp, t_train, yf_train[:, 0], max_gp_iterations=100)

            # Sample
            X_0 = np.array(np.hstack(
                [x_test_cmgp, np.zeros_like(x_test[:, 1].reshape((len(x_test_cmgp[:, 1]), 1)))]))
            X_1 = np.array(np.hstack(
                [x_test_cmgp, np.ones_like(x_test[:, 1].reshape((len(x_test_cmgp[:, 1]), 1)))]))
            X_0_shape = X_0.shape
            X_1_shape = X_1.shape
            noise_dict_0 = {"output_index": X_0[:, X_0_shape[1] - 1].reshape((X_0_shape[0], 1)).astype(int)}
            noise_dict_1 = {"output_index": X_1[:, X_1_shape[1] - 1].reshape((X_1_shape[0], 1)).astype(int)}

            mu0 = np.array(list(cmgp.model.predict(X_0, Y_metadata=noise_dict_0)[0]))
            mu1 = np.array(list(cmgp.model.predict(X_1, Y_metadata=noise_dict_1)[0]))
            var0 = np.array(list(cmgp.model.predict(X_0, Y_metadata=noise_dict_0)[1]))
            var1 = np.array(list(cmgp.model.predict(X_1, Y_metadata=noise_dict_1)[1]))

            y0_posteriors = np.zeros((len(t_test), params['n_samples']))
            y1_posteriors = np.zeros((len(t_test), params['n_samples']))
            for i in range(len(yf_test)):
                y0_posteriors[i, :] = np.random.normal(mu0[i, 0], np.sqrt(var0[i, 0]), size=params['n_samples'])
                y1_posteriors[i, :] = np.random.normal(mu1[i, 0], np.sqrt(var1[i, 0]), size=params['n_samples'])
            ite_samples_pred = y1_posteriors - y0_posteriors
            y0_pred = mu0[:, 0]
            y1_pred = mu1[:, 0]

            ite_pred = y1_pred - y0_pred

            # Evaluation:
            PEHE = np.mean(np.square(ite_pred - ite_mu_test))
            ATE = np.mean(np.square(ite_pred.mean() - ite_mu_test.mean()))
            print('\nPEHE =', np.round(PEHE, 4), '\t\tsqrt(PEHE) =', np.round(np.sqrt(PEHE), 4), '\t\tATE =',
                  np.round(ATE, 4))
            pehe_cmgp[iter] = np.sqrt(PEHE)
            # Log likelihood:
            # loglik = loglikelihood(ite_mu_test, ite_samples_pred)
            loglik = loglikelihood(ite_sample_test, ite_samples_pred)
            print('LL\t =', np.round(loglik, 4))
            ll_cmgp[iter] = loglik

            ite_l_pred = np.percentile(a=ite_samples_pred, q=5, axis=1)
            ite_r_pred = np.percentile(a=ite_samples_pred, q=95, axis=1)
            if params['dataset'] == 'SyntheticCI' or params['dataset'] == 'EDU' or params['dataset'] == 'IHDP':
                iou = iou_ci(dm.ite_l_test.numpy()[:, 0], dm.ite_r_test.numpy()[:, 0], ite_l_pred, ite_r_pred)

                iou_cmgp[iter] = iou
                print('IoU = ', np.round(iou, 4))

                # Assign treatments based on highest expected utility
                if params['dataset'] == 'IHDP':
                    expected_utility = ((ite_samples_pred - 4) ** 3).mean(axis=1)
                elif params['dataset'] == 'EDU':
                    expected_utility = ((ite_samples_pred - 1) ** 3).mean(axis=1)
                pred_opt_treatment = np.zeros_like(expected_utility)
                pred_opt_treatment[expected_utility > 0] = 1
                treatment_accuracy = (pred_opt_treatment == opt_treatment).mean()
                accuracy_cmgp[iter] = treatment_accuracy
                print('Treatment accuracy = ', np.round(treatment_accuracy, 4))

            coverage = np.mean((ite_l_pred <= ite_sample_test) * (ite_sample_test <= ite_r_pred))
            coverage_cmgp[iter] = coverage
            print('Coverage = ', np.round(coverage, 4))

            if params['visualize']:
                # plot_errors(outcomes_test, np.vstack((y0_pred, y1_pred)).T)
                # plot_samples(yf_test, ycf_test, t_test, y0_pred, y1_pred, y0_posteriors, y1_posteriors)
                plot_samples_ite(ite_mu_test, np.array(ite_samples_pred))

                loglik = loglikelihood(ite_mu_test, ite_samples_pred, return_average=False)
                sns.kdeplot(loglik, fill=True, alpha=0.5, cut=0, color='pink')
                plt.show()

                if params['dataset'] == 'SyntheticCI' or params['dataset'] == 'EDU' or params['dataset'] == 'IHDP':
                    plot_cis(ite_mu_test, dm.ite_l_test.numpy(), dm.ite_r_test.numpy(), ite_pred, ite_l_pred, ite_r_pred)

                    plt.plot(dm.ite_l_test, ite_l_pred, '.', color='green', alpha=0.4)
                    plt.plot(dm.ite_r_test, ite_r_pred, '.', color='orange', alpha=0.4)
                    plt.plot(dm.ite_l_test.mean(), ite_l_pred.mean(), 's', color='green', alpha=1)
                    plt.plot(dm.ite_r_test.mean(), ite_r_pred.mean(), 's', color='orange', alpha=1)
                    lims = [
                        np.min([plt.gca().get_xlim(), plt.gca().get_ylim()]),  # min of both axes
                        np.max([plt.gca().get_xlim(), plt.gca().get_ylim()]),  # max of both axes
                    ]
                    plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
                    plt.show()

                elif params['dataset'] == 'EDU':
                    ite_l_pred = np.quantile(a=y0_posteriors, q=5, axis=1)
                    ite_r_pred = np.quantile(a=y0_posteriors, q=95, axis=1)
                    itv0 = ite_r_pred - ite_l_pred

                    ite_l_pred = np.quantile(a=y1_posteriors, q=5, axis=1)
                    ite_r_pred = np.quantile(a=y1_posteriors, q=95, axis=1)
                    itv1 = ite_r_pred - ite_l_pred

                    plot_cis_edu(itv0, itv1, x_test[:, 23])

        # # BART
        # if params['BART']:
        #     print('\n---- BART ---------------------')

            # model = pm.Model()
            # with pm.Model() as model:
            #     # x_train_cmgp, t_train,
            #     mu = pmb.BART("mu", X=x_train, Y=yf_train[:, 0], m=1)
            #     # μ = pm.Deterministic("μ", pm.math.exp(μ_))
            #     normal = pm.Normal("normal", mu=mu)
            #     # y_pred = pm.Poisson("y_pred", mu=μ, observed=y_data)
            #     # idata_coal = pm.sample(random_seed=RANDOM_SEED)
            #     y_pred = pm.sample(

        # CEVAE:
        if params['CEVAE']:
            print('\n---- CEVAE --------------------')
            if params['dataset'] == 'IHDP':
                ite_samples_pred = cevae(torch.Tensor(x_train), torch.Tensor(t_train[:, None]), torch.Tensor(yf_train),
                                         torch.Tensor(x_test), torch.Tensor(t_test[:, None]),
                                         dim_bin=19, dim_cont=6,
                                         lr=1e-4, decay=1e-4, batch_size=100, iters=7000, n_h=64,
                                         n_samples=params['n_samples'])
            elif params['dataset'] == 'EDU':
                ite_samples_pred = cevae(torch.Tensor(x_train), torch.Tensor(t_train[:, None]), torch.Tensor(yf_train),
                                         torch.Tensor(x_test), torch.Tensor(t_test[:, None]),
                                         dim_bin=14, dim_cont=18,
                                         lr=1e-4, decay=1e-4, batch_size=256, iters=18000, n_h=64,
                                         n_samples=params['n_samples'])
            elif params['dataset'] == 'News':
                ite_samples_pred = cevae(torch.Tensor(x_train), torch.Tensor(t_train[:, None]), torch.Tensor(yf_train),
                                         torch.Tensor(x_test), torch.Tensor(t_test[:, None]),
                                         dim_bin=0, dim_cont=x_train.shape[1],
                                         lr=1e-4, decay=5e-3, batch_size=100, iters=7000, n_h=64,
                                         n_samples=params['n_samples'])
            else:
                ite_samples_pred = cevae(torch.Tensor(x_train), torch.Tensor(t_train[:, None]), torch.Tensor(yf_train),
                                         torch.Tensor(x_test), torch.Tensor(t_test[:, None]),
                                         dim_bin=0, dim_cont=x_train.shape[1],
                                         lr=1e-4, decay=1e-4, batch_size=100, iters=7000, n_h=64,
                                         n_samples=params['n_samples'])

            ite_pred = ite_samples_pred.mean(axis=0)

            # Evaluation:
            PEHE = np.mean(np.square(ite_pred - ite_mu_test))
            ATE = np.mean(np.square(ite_pred.mean() - ite_mu_test.mean()))
            print('\nPEHE =', np.round(PEHE, 4), '\t\tsqrt(PEHE) =', np.round(np.sqrt(PEHE), 4), '\t\tATE =', np.round(ATE, 4))
            pehe_cevae[iter] = np.sqrt(PEHE)
            # Log likelihood:
            # loglik = loglikelihood(ite_mu_test, ite_samples_pred)
            loglik = loglikelihood(ite_sample_test, ite_samples_pred)
            print('LL\t =', np.round(loglik, 4))
            ll_cevae[iter] = loglik

            ite_l_pred = np.percentile(a=ite_samples_pred, q=5, axis=0)
            ite_r_pred = np.percentile(a=ite_samples_pred, q=95, axis=0)
            if params['dataset'] == 'SyntheticCI' or params['dataset'] == 'EDU' or params['dataset'] == 'IHDP':
                iou = iou_ci(dm.ite_l_test.numpy()[:, 0], dm.ite_r_test.numpy()[:, 0], ite_l_pred, ite_r_pred)

                iou_cevae[iter] = iou
                print('IoU = ', np.round(iou, 4))

                # Assign treatments based on highest expected utility
                if params['dataset'] == 'IHDP':
                    expected_utility = ((ite_samples_pred - 4)**3).mean(axis=0)
                elif params['dataset'] == 'EDU':
                    expected_utility = ((ite_samples_pred - 1)**3).mean(axis=0)
                pred_opt_treatment = np.zeros_like(expected_utility)
                pred_opt_treatment[expected_utility > 0] = 1
                treatment_accuracy = (pred_opt_treatment == opt_treatment).mean()
                accuracy_cevae[iter] = treatment_accuracy
                print('Treatment accuracy = ', np.round(treatment_accuracy, 4))

            coverage = np.mean((ite_l_pred <= ite_sample_test) * (ite_sample_test <= ite_r_pred))
            coverage_cevae[iter] = coverage
            print('Coverage = ', np.round(coverage, 4))

            if params['visualize']:
                # plot_errors(outcomes_test, np.array(ite_pred.repeat((2, 1)).T))
                plot_samples_ite(ite_mu_test, np.array(ite_samples_pred))

                loglik = loglikelihood(ite_mu_test, ite_samples_pred, return_average=False)
                sns.kdeplot(loglik, fill=True, alpha=0.5, cut=0, color='pink')
                plt.show()

                if params['dataset'] == 'SyntheticCI' or params['dataset'] == 'EDU' or params['dataset'] == 'IHDP':
                    plot_cis(ite_mu_test, dm.ite_l_test.numpy(), dm.ite_r_test.numpy(), ite_pred, ite_l_pred, ite_r_pred)

                    plt.plot(dm.ite_l_test, ite_l_pred, '.', color='green', alpha=0.4)
                    plt.plot(dm.ite_r_test, ite_r_pred, '.', color='orange', alpha=0.4)
                    lims = [
                        np.min([plt.gca().get_xlim(), plt.gca().get_ylim()]),  # min of both axes
                        np.max([plt.gca().get_xlim(), plt.gca().get_ylim()]),  # max of both axes
                    ]
                    plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
                    plt.show()

        # # Train GANITE:
        if params['GANITE']:
            # Todo: hyperparameters
            print('\n---- GANITE --------------------')
            if params['dataset'] == 'IHDP':
                ganite_params = {'h_dim': 30,                          # hidden dimensions
                                 'batch_size': 64,                     # the number of samples in each batch
                                 'iterations': 10000,                  # the number of iterations for training
                                 'alpha': 2.,
                                 'beta': 5.,                           # hyper-parameter to adjust the loss importance
                                 'n_samples': params['n_samples'],
                                 'input_size': params['input_size'],
                                 }
            elif params['dataset'] == 'EDU':
                ganite_params = {'h_dim': 64,                           # hidden dimensions
                                 'batch_size': 128,                     # the number of samples in each batch
                                 'iterations': 15000,                   # the number of iterations for training
                                 'alpha': 2.,
                                 'beta': 1e-3,                           # hyper-parameter to adjust the loss importance
                                 'n_samples': params['n_samples'],
                                 'input_size': params['input_size'],
                                 }
            elif params['dataset'] == 'News':
                ganite_params = {'h_dim': 64,                           # hidden dimensions
                                 'batch_size': 128,                     # the number of samples in each batch
                                 'iterations': 15000,                   # the number of iterations for training
                                 'alpha': 0.,
                                 'beta': 0.,                           # hyper-parameter to adjust the loss importance
                                 'n_samples': params['n_samples'],
                                 'input_size': params['input_size'],
                                 }
            else:
                ganite_params = {'h_dim': 8,  # hidden dimensions
                                 'batch_size': 64,  # the number of samples in each batch
                                 'iterations': 10000,  # the number of iterations for training
                                 'alpha': 2,
                                 'beta': 5,  # hyper-parameter to adjust the loss importance
                                 'n_samples': params['n_samples'],
                                 'input_size': params['input_size'],
                                 }

            ite_samples_pred = ganite_prob(xtrain=x_train,
                                           trttrain=t_train[:, None],
                                           ytrain=yf_train,
                                           xtest=x_test,
                                           parameters=ganite_params)
            ite_pred = ite_samples_pred.mean(axis=1)

            # Evaluation:
            PEHE = np.mean(np.square(ite_pred - ite_mu_test))
            ATE = np.mean(np.square(ite_pred.mean() - ite_mu_test.mean()))
            print('\nPEHE =', np.round(PEHE, 4), '\t\tsqrt(PEHE) =', np.round(np.sqrt(PEHE), 4), '\t\tATE =', np.round(ATE, 4))
            pehe_ganite[iter] = np.sqrt(PEHE)
            # Log likelihood:
            # loglik = loglikelihood(ite_mu_test, ite_samples_pred)
            loglik = loglikelihood(ite_sample_test, ite_samples_pred)
            print('LL\t =', np.round(loglik, 4))
            ll_ganite[iter] = loglik

            ite_l_pred = np.percentile(a=ite_samples_pred, q=5, axis=1)
            ite_r_pred = np.percentile(a=ite_samples_pred, q=95, axis=1)
            if params['dataset'] == 'SyntheticCI' or params['dataset'] == 'EDU' or params['dataset'] == 'IHDP':
                iou = iou_ci(dm.ite_l_test.numpy()[:, 0], dm.ite_r_test.numpy()[:, 0], ite_l_pred, ite_r_pred)

                iou_ganite[iter] = iou
                print('IoU = ', np.round(iou, 4))

                # Assign treatments based on highest expected utility
                if params['dataset'] == 'IHDP':
                    expected_utility = ((ite_samples_pred - 4) ** 3).mean(axis=1)
                elif params['dataset'] == 'EDU':
                    expected_utility = ((ite_samples_pred - 1) ** 3).mean(axis=1)
                pred_opt_treatment = np.zeros_like(expected_utility)
                pred_opt_treatment[expected_utility > 0] = 1
                treatment_accuracy = (pred_opt_treatment == opt_treatment).mean()
                accuracy_ganite[iter] = treatment_accuracy
                print('Treatment accuracy = ', np.round(treatment_accuracy, 4))

            coverage = np.mean((ite_l_pred <= ite_sample_test) * (ite_sample_test <= ite_r_pred))
            coverage_ganite[iter] = coverage
            print('Coverage = ', np.round(coverage, 4))

            if params['visualize']:
                plot_samples_ite(ite_mu_test, np.array(ite_samples_pred))

                loglik = loglikelihood(ite_mu_test, ite_samples_pred, return_average=False)
                sns.kdeplot(loglik, fill=True, alpha=0.5, cut=0, color='pink')
                plt.show()

                if params['dataset'] == 'SyntheticCI' or params['dataset'] == 'EDU' or params['dataset'] == 'IHDP':
                    plot_cis(ite_mu_test, dm.ite_l_test.numpy(), dm.ite_r_test.numpy(), ite_pred, ite_l_pred, ite_r_pred)

                    plt.plot(dm.ite_l_test, ite_l_pred, '.', color='green', alpha=0.4)
                    plt.plot(dm.ite_r_test, ite_r_pred, '.', color='orange', alpha=0.4)
                    lims = [
                        np.min([plt.gca().get_xlim(), plt.gca().get_ylim()]),  # min of both axes
                        np.max([plt.gca().get_xlim(), plt.gca().get_ylim()]),  # max of both axes
                    ]
                    plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
                    plt.show()

        # CCN:
        if params['FCCN']:
            print('\n---- FCCN ----------------------')
            if params['dataset'] == 'IHDP':
                fccn = FCCN(params['input_size'], alpha=5e-4, beta=1e-5)
                g_losses, _, _, _ = fccn.train(x_train, yf_train, t_train, iters=20000)
            elif params['dataset'] == 'EDU':
                fccn = FCCN(params['input_size'], alpha=5e-4, beta=1e-4, EDU=True)    # 1e-5; 5e-3
                g_losses, _, _, _ = fccn.train(x_train, yf_train, t_train, iters=50000)
            elif params['dataset'] == 'News':
                fccn = FCCN(params['input_size'], alpha=1e-5, beta=5e-3, EDU=False)  # 1e-5; 5e-3
                g_losses, _, _, _ = fccn.train(x_train, yf_train, t_train, iters=20000)
            else:
                fccn = FCCN(params['input_size'])
                g_losses, _, _, _ = fccn.train(x_train, yf_train, t_train, iters=50000)
                plt.plot(g_losses)
                plt.show()

            ite_samples_pred = fccn.predict(x_test, params['n_samples'])

            # Predict by sampling n_samples
            ite_pred = ite_samples_pred.mean(axis=1)

            # Evaluation:
            PEHE = np.mean(np.square(ite_pred - ite_mu_test))
            ATE = np.mean(np.square(ite_pred.mean() - ite_mu_test.mean()))
            print('\nPEHE =', np.round(PEHE, 4), '\t\tsqrt(PEHE) =', np.round(np.sqrt(PEHE), 4), '\t\tATE =', np.round(ATE, 4))
            pehe_fccn[iter] = np.sqrt(PEHE)
            # Log likelihood:
            # loglik = loglikelihood(ite_mu_test, ite_samples_pred)
            loglik = loglikelihood(ite_sample_test, ite_samples_pred)
            print('LL\t =', np.round(loglik, 4))
            ll_fccn[iter] = loglik

            ite_l_pred = np.percentile(a=ite_samples_pred, q=5, axis=1)
            ite_r_pred = np.percentile(a=ite_samples_pred, q=95, axis=1)
            if params['dataset'] == 'SyntheticCI' or params['dataset'] == 'EDU' or params['dataset'] == 'IHDP':
                iou = iou_ci(dm.ite_l_test.numpy()[:, 0], dm.ite_r_test.numpy()[:, 0], ite_l_pred, ite_r_pred)

                iou_fccn[iter] = iou
                print('IoU = ', np.round(iou, 4))

                # Assign treatments based on highest expected utility
                if params['dataset'] == 'IHDP':
                    expected_utility = ((ite_samples_pred - 4) ** 3).mean(axis=1)
                elif params['dataset'] == 'EDU':
                    expected_utility = ((ite_samples_pred - 1) ** 3).mean(axis=1)
                pred_opt_treatment = np.zeros_like(expected_utility)
                pred_opt_treatment[expected_utility > 0] = 1
                treatment_accuracy = (pred_opt_treatment == opt_treatment).mean()
                accuracy_fccn[iter] = treatment_accuracy
                print('Treatment accuracy = ', np.round(treatment_accuracy, 4))

            coverage = np.mean((ite_l_pred <= ite_sample_test) * (ite_sample_test <= ite_r_pred))
            coverage_fccn[iter] = coverage
            print('Coverage = ', np.round(coverage, 4))

            if params['visualize']:
                # plot_errors(outcomes_test, np.array(ite_pred.repeat((2, 1)).T))
                # plot_samples_ite(ite_mu_test, np.array(ite_samples_pred))

                loglik = loglikelihood(ite_mu_test, ite_samples_pred, return_average=False)
                sns.kdeplot(loglik, fill=True, alpha=0.5, cut=0, color='pink')
                plt.show()

                if params['dataset'] == 'SyntheticCI' or params['dataset'] == 'EDU' or params['dataset'] == 'IHDP':
                    plot_cis(ite_mu_test, dm.ite_l_test.numpy(), dm.ite_r_test.numpy(), ite_pred, ite_l_pred, ite_r_pred)

                    plt.plot(dm.ite_l_test, ite_l_pred, '.', color='green', alpha=0.4)
                    plt.plot(dm.ite_r_test, ite_r_pred, '.', color='orange', alpha=0.4)
                    lims = [
                        np.min([plt.gca().get_xlim(), plt.gca().get_ylim()]),  # min of both axes
                        np.max([plt.gca().get_xlim(), plt.gca().get_ylim()]),  # max of both axes
                    ]
                    plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
                    plt.show()
                # elif params['dataset'] == 'EDU':
                #     ite_l_pred = np.quantile(a=y0_posteriors, q=5, axis=1)
                #     ite_r_pred = np.quantile(a=y0_posteriors, q=95, axis=1)
                #     itv0 = ite_r_pred - ite_l_pred
                #
                #     ite_l_pred = np.quantile(a=y1_posteriors, q=5, axis=1)
                #     ite_r_pred = np.quantile(a=y1_posteriors, q=95, axis=1)
                #     itv1 = ite_r_pred - ite_l_pred
                #
                #     plot_cis_edu(itv0, itv1, x_test[:, 23])

        # NOFLITE:
        if params['NOFLITE']:
            print('\n---- NOFLITE --------------------')
            if params['wandb']:
                # if not(params['sweep']):  # No init needed during sweep
                if wandb.run is None:
                    wandb.init(config=params, project="NOFLITE", entity="toon")
            # Create model:
            # (Initialize GF)
            params['cur_datapoints'] = torch.Tensor(dm.dataset.yf_train)[
                torch.randint(low=0, high=len(dm.dataset.yf_train), size=(params['datapoint_num'], 1)), 0]

            model = Noflite(params=params)

            # Train:
            if params['wandb']:
                logger = WandbLogger(project='NOFLITE')
            else:
                logger = False

            trainer = pl.Trainer(accelerator=device, max_steps=params['max_steps'], logger=logger,
                                 enable_checkpointing=False, check_val_every_n_epoch=10,
                                 # callbacks=[EarlyStopping(monitor="val_loss", mode="min", min_delta=0., patience=20,
                                 #                          verbose=True), ],
                                 gradient_clip_val=10,
                                 )

            if params['sweep']:  # For tuning/validation
                # trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
                trainer.fit(model, train_dataloaders=train_loader)
            else:
                trainer.fit(model, train_dataloaders=train_val_loader)  # For final testing

            # Testing
            model.eval()
            if params['sweep']:
                # # For tuning/validation
                trainer.validate(model, dataloaders=val_loader)
                # # Get factual outcome:
                mu, log_std, x_bal = model.get_conditional_prior(torch.from_numpy(x_val).float(),
                                                                 torch.from_numpy(t_val).float())
                y_pred = model.decode(mu, x_bal, torch.from_numpy(t_val).float())
                # Get counterfactual outcome:
                mu_cf, log_std_cf, x_bal_cf = model.get_conditional_prior(torch.from_numpy(x_val).float(),
                                                                          torch.from_numpy(1 - t_val).float())
                y_pred_cf = model.decode(mu_cf, x_bal_cf, torch.from_numpy(1 - t_val).float())

                # Get PEHE proxy based on 1-nearest neighbour (based on expected value only)
                pehe_proxy = pehe_nn(yf_p=y_pred[:, 0],
                                     ycf_p=y_pred_cf[:, 0],
                                     y=torch.from_numpy(yf_val[:, 0]).float(),
                                     x=torch.from_numpy(x_val).float(),
                                     t=torch.from_numpy(t_val).float())
                if params['wandb']:
                    wandb.log({'PEHE_nn': pehe_proxy})

                # Get weighted metrics:
                lr = LogisticRegression().fit(x_train, t_train)
                probas_val = lr.predict_proba(x_val)
                props_val = probas_val[:, 0]
                props_val[t_val == 1] = probas_val[:, 1][t_val == 1]
                # MSE
                mses_val_weighted = torch.mean((y_pred - yf_val) ** 2 / props_val[:, None])
                if params['wandb']:
                    wandb.log({'MSE_val_weighted': mses_val_weighted})
                # LL observed outcomes:
                mu, log_std, x_bal = model.get_conditional_prior(torch.from_numpy(x_val).float(),
                                                                 torch.from_numpy(t_val).float())
                sig = torch.exp(log_std)

                # Calculate Y_estimated by using inverse flows and µ(x,t)
                n_samples = params['n_samples']
                y_posteriors = np.zeros((len(t_test), n_samples))
                y_pred = np.zeros_like(t_test, dtype=float)

                # Probability to cut off for sampling
                trunc_prob = params['trunc_prob']
                if trunc_prob < 0.5:
                    trunc_prob = 1 - trunc_prob
                # Sample from truncated normal for more stable results
                trunc_perc = norm.ppf(trunc_prob)

                for i in range(len(x_val)):
                    print(end="\r|%-100s|" % ("=" * int(100 * i / (len(x_test) - 1))))
                    y_prior_samples = torch.Tensor(truncnorm.rvs(a=-trunc_perc, b=trunc_perc, size=n_samples)) \
                                        * sig[i, :] + mu[i, :]  # Transform to samples from conditional prior
                    y_posterior = model.decode(z=y_prior_samples[:, None],
                                                x=x_bal[i, :][None, :].repeat((n_samples, 1)),
                                                t=torch.zeros(n_samples))
                    y_posteriors[i, :] = y_posterior.detach().numpy()[:, 0]
                    y_pred[i] = y_posterior.mean()

                loglik_weighted = np.mean(loglikelihood(yf_val, y_posteriors, return_average=False) / props_val[:, None])
                if params['wandb']:
                    wandb.log({'NLL_test_weighted': -loglik_weighted})

            else:  # if not params['sweep']:
                # For final testing
                trainer.test(model, dataloaders=test_loader)
                # Take samples:
                # First, set max iter for sampling sufficiently high (only for Sigmoid flows)
                model.max_iter = 2000
                # Evaluation -- Get conditional prior's parameter (per instance)
                mu0, logsig0, x_bal0 = model.get_conditional_prior(x=torch.from_numpy(x_test).float(),
                                                                   t=torch.zeros(len(x_test)))
                mu1, logsig1, x_bal1 = model.get_conditional_prior(x=torch.from_numpy(x_test).float(),
                                                                   t=torch.ones(len(x_test)))
                sig0 = torch.exp(logsig0)
                sig1 = torch.exp(logsig1)
                t_test = torch.from_numpy(t_test)
                xt0 = torch.cat((x_bal0, torch.zeros((len(x_test), 1))), -1)
                xt1 = torch.cat((x_bal1, torch.ones((len(x_test), 1))), -1)

                # Calculate Y_estimated by using inverse flows and µ(x,t)
                n_samples = params['n_samples']
                y0_posteriors = np.zeros((len(t_test), n_samples))
                y1_posteriors = np.zeros((len(t_test), n_samples))
                y0_pred = np.zeros_like(t_test, dtype=float)
                y1_pred = np.zeros_like(t_test, dtype=float)

                # Probability to cut off for sampling
                trunc_prob = params['trunc_prob']
                if trunc_prob < 0.5:
                    trunc_prob = 1 - trunc_prob
                # Sample from truncated normal for more stable results
                trunc_perc = norm.ppf(trunc_prob)

                for i in range(len(x_test)):
                    print(end="\r|%-100s|" % ("=" * int(100 * i / (len(x_test) - 1))))
                    # t=0
                    y0_prior_samples = torch.Tensor(truncnorm.rvs(a=-trunc_perc, b=trunc_perc, size=n_samples)) \
                                        * sig0[i, :] + mu0[i, :]  # Transform to samples from conditional prior
                    y0_posterior = model.decode(z=y0_prior_samples[:, None],
                                                x=x_bal0[i, :][None, :].repeat((n_samples, 1)),
                                                t=torch.zeros(n_samples))
                    y0_posteriors[i, :] = y0_posterior.detach().numpy()[:, 0]
                    y0_pred[i] = y0_posterior.mean()
                    # t=1
                    y1_prior_samples = torch.Tensor(truncnorm.rvs(a=-trunc_perc, b=trunc_perc, size=n_samples))\
                                        * sig1[i, :] + mu1[i, :]
                    y1_posterior = model.decode(z=y1_prior_samples[:, None],
                                                x=x_bal1[i, :][None, :].repeat((n_samples, 1)),
                                                t=torch.ones(n_samples))
                    y1_posteriors[i, :] = y1_posterior.detach().numpy()[:, 0]
                    y1_pred[i] = y1_posterior.mean()
                # Combine
                outcomes_pred = np.stack((y0_pred, y1_pred), axis=-1)
                # calculate PEHE and ATE
                PEHE = calculatePEHE(outcomes_test, outcomes_pred)
                ATE = calculateATE(outcomes_test, outcomes_pred)
                print('\nPEHE =', np.round(PEHE, 4), '\t\tsqrt(PEHE) =', np.round(np.sqrt(PEHE), 4), '\t\tATE =', np.round(ATE, 4))
                pehe_noflite[iter] = np.sqrt(PEHE)
                # MSE for potential outcomes:
                mse_po_noflite = np.mean(np.square(outcomes_test - outcomes_pred))
                print('MSE PO =', np.round(mse_po_noflite, 4))
                if params['wandb']:
                    wandb.log({'PO_MSE': mse_po_noflite,
                               'PEHE': PEHE})
                # LL:
                # samples_pred = np.stack((y0_posteriors, y1_posteriors), axis=-1)
                ite_samples_pred = y1_posteriors - y0_posteriors
                ite_pred = ite_samples_pred.mean(axis=1)
                # loglik = loglikelihood(ite_mu_test, ite_samples_pred)
                loglik = loglikelihood(ite_sample_test, ite_samples_pred)
                print('LL\t =', np.round(loglik, 4))
                ll_noflite[iter] = loglik
                if params['wandb']:
                    wandb.log({'NLL_test': loglik})
                print('LL PO -- T=0\t =', np.round(loglikelihood(outcomes_test[:, 0], y0_posteriors), 4))
                print('LL PO -- T=1\t =', np.round(loglikelihood(outcomes_test[:, 1], y1_posteriors), 4))

                # Todo: check if better results with additional truncation
                # ite_l_pred = np.percentile(a=ite_samples_pred, q=5, axis=1)
                # ite_r_pred = np.percentile(a=ite_samples_pred, q=95, axis=1)
                # Adjusted 5/95 percentiles because we work with truncated normal
                # ite_l_pred = np.quantile(a=ite_samples_pred, q=(1 - 90/99.5)/2, axis=1)
                # ite_r_pred = np.quantile(a=ite_samples_pred, q=1 - (1 - 90/99.5)/2, axis=1)
                ite_l_pred = np.quantile(a=ite_samples_pred, q=0.05, axis=1)
                ite_r_pred = np.quantile(a=ite_samples_pred, q=0.95, axis=1)
                if params['dataset'] == 'SyntheticCI' or params['dataset'] == 'EDU' or params['dataset'] == 'IHDP':
                    iou = iou_ci(dm.ite_l_test.numpy()[:, 0], dm.ite_r_test.numpy()[:, 0], ite_l_pred, ite_r_pred)

                    iou_noflite[iter] = iou
                    print('IoU = ', np.round(iou, 4))
                    if params['wandb']:
                        wandb.log({'IoU': iou})

                    # Assign treatments based on highest expected utility
                    if params['dataset'] == 'IHDP':
                        expected_utility = ((ite_samples_pred - 4) ** 3).mean(axis=1)
                    elif params['dataset'] == 'EDU':
                        expected_utility = ((ite_samples_pred - 1) ** 3).mean(axis=1)
                    pred_opt_treatment = np.zeros_like(expected_utility)
                    pred_opt_treatment[expected_utility > 0] = 1
                    treatment_accuracy = (pred_opt_treatment == opt_treatment).mean()
                    accuracy_noflite[iter] = treatment_accuracy
                    print('Treatment accuracy = ', np.round(treatment_accuracy, 4))
                    if params['wandb']:
                        wandb.log({'Treatment accuracy': treatment_accuracy})

                coverage = np.mean((ite_l_pred <= ite_sample_test) * (ite_sample_test <= ite_r_pred))
                coverage_noflite[iter] = coverage
                print('Coverage = ', np.round(coverage, 4))
                if params['wandb']:
                    wandb.log({'Coverage': coverage})

                # Visualizations
                if params['visualize']:
                    plot_errors(outcomes_test, outcomes_pred)

                    plot_samples(yf_test, ycf_test, t_test, y0_pred, y1_pred, y0_posteriors, y1_posteriors)

                    plot_samples_ite(ite_mu_test, ite_samples_pred)

                    logliks = loglikelihood(ite_mu_test, ite_samples_pred, return_average=False)
                    sns.kdeplot(logliks, fill=True, alpha=0.5, cut=0, color='pink')
                    plt.show()

                    if params['n_flows'] > 0 and params['metalearner'] == 'S':
                        fig, axs = plt.subplots(1, params['n_flows'] + 1, figsize=(params['n_flows'] * 2, 3), dpi=300)
                        plt.gcf().suptitle('Flow visualization (random instance)', fontsize=14)

                        id = np.random.randint(len(t_test))

                        z0 = torch.normal(mean=mu0[id, :].float().repeat(n_samples), std=sig0[id, :].float().repeat(n_samples))
                        z1 = torch.normal(mean=mu1[id, :].float().repeat(n_samples), std=sig1[id, :].float().repeat(n_samples))
                        ax = axs[0]
                        ax.set_title('Prior')
                        sns.kdeplot(z0.detach().numpy(), color='orange', ax=ax, alpha=0.5)
                        sns.kdeplot(z1.detach().numpy(), color='blue', ax=ax, alpha=0.5)
                        ax.set_ylabel('')

                        z0 = z0[:, None]
                        z1 = z1[:, None]

                        for i in range(params['n_flows']):
                            ax = axs[i + 1]
                            ax.set_title('Flow ' + str(i+1))

                            if params['flow_type'] == 'SigmoidXT' or params['flow_type'] == 'SigmoidT':
                                flow_n = params['n_flows'] - i - 1      # Start counting from last flow
                                # flow_n = i
                                flow = model.flows.marginal_flow.layers[flow_n]
                                # t = 0
                                if params['flow_type'] == 'SigmoidXT':
                                    marginal_params = model.flows.marginal_conditioner(xt0[id, :])
                                elif params['flow_type'] == 'SigmoidT':
                                    marginal_params = model.flows.marginal_conditioner(torch.zeros(1))
                                temp_params = marginal_params[flow_n * model.flows.marginal_flow.params_length: (flow_n + 1) * model.flows.marginal_flow.params_length]
                                # Get inverse
                                max_value = 100
                                max_iter = 5000
                                left = -max_value * torch.ones_like(z0)
                                right = max_value * torch.ones_like(z0)
                                max_error = torch.inf
                                while max_error > 1e-4:
                                    mid = (left + right) / 2
                                    error = flow.forward_no_logdet(temp_params, mid) - z0
                                    left[error <= 0] = mid[error <= 0]      # Update left if error > 0
                                    right[error >= 0] = mid[error >= 0]     # Update right if error < 0

                                    max_error = error.abs().max().item()
                                z0 = mid
                                sns.kdeplot(z0.detach().float().flatten(), color='orange', ax=ax, alpha=0.5)
                                # t = 1
                                if params['flow_type'] == 'SigmoidXT':
                                    marginal_params = model.flows.marginal_conditioner(xt1[id, :])
                                elif params['flow_type'] == 'SigmoidT':
                                    marginal_params = model.flows.marginal_conditioner(torch.ones(1))
                                temp_params = marginal_params[flow_n * model.flows.marginal_flow.params_length: (flow_n + 1) * model.flows.marginal_flow.params_length]
                                # z1 = flow.forward_no_logdet(temp_params, z1)
                                left = -max_value * torch.ones_like(z1)
                                right = max_value * torch.ones_like(z1)
                                max_error = torch.inf
                                while max_error > 1e-4:
                                    mid = (left + right) / 2
                                    error = flow.forward_no_logdet(temp_params, mid) - z1
                                    left[error <= 0] = mid[error <= 0]
                                    right[error >= 0] = mid[error >= 0]

                                    max_error = error.abs().max().item()
                                z1 = mid
                                sns.kdeplot(z1.detach().float().flatten(), color='blue', ax=ax, alpha=0.5)
                            elif params['flow_type'] == 'GF':
                                process_size = params['datapoint_num']
                                flow = list(model.flows)[-(i + 1)]

                                for iter in range(len(z0) // process_size):
                                    z0[iter * process_size: (iter + 1) * process_size, :] = flow.sampling(
                                        z0[iter * process_size: (iter + 1) * process_size, :])
                                sns.kdeplot(z0.detach().float().flatten(), color='orange', ax=ax, alpha=0.5)

                                # datapoints_array = []
                                # cur_datapoints = params['cur_datapoints']
                                process_size = params['datapoint_num']
                                # datapoints_array.append(cur_datapoints)
                                for iter in range(len(z1) // process_size):
                                    z1[iter * process_size: (iter + 1) * process_size, :] = flow.sampling(
                                        z1[iter * process_size: (iter + 1) * process_size, :])
                                sns.kdeplot(z1.detach().float().flatten(), color='blue', ax=ax, alpha=0.5)
                            else:
                                for flow in reversed(model.flows[i:i+1]):
                                    z0, _ = flow.inverse(z0)
                                sns.kdeplot(z0.detach().float().flatten(), color='orange', ax=ax, alpha=0.5)
                                for flow in reversed(model.flows[i:i+1]):
                                    z1, _ = flow.inverse(z1)
                                sns.kdeplot(z1.detach().float().flatten(), color='blue', ax=ax, alpha=0.5)
                            ax.set_ylabel('')
                        plt.show()

                    if params['dataset'] == 'SyntheticCI' or params['dataset'] == 'EDU' or params['dataset'] == 'IHDP':
                        plot_cis(ite_mu_test, dm.ite_l_test.numpy(), dm.ite_r_test.numpy(), ite_pred, ite_l_pred,
                                 ite_r_pred)

                        plt.figure(dpi=200)
                        plt.plot(dm.ite_l_test, ite_l_pred, '.', color='green', alpha=0.4)
                        plt.plot(dm.ite_r_test, ite_r_pred, '.', color='orange', alpha=0.4)
                        plt.plot(dm.ite_l_test.mean(), ite_l_pred.mean(), 's', color='green', alpha=1)
                        plt.plot(dm.ite_r_test.mean(), ite_r_pred.mean(), 's', color='orange', alpha=1)
                        lims = [
                            np.min([plt.gca().get_xlim(), plt.gca().get_ylim()]),  # min of both axes
                            np.max([plt.gca().get_xlim(), plt.gca().get_ylim()]),  # max of both axes
                        ]
                        plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
                        plt.show()

                    elif params['dataset'] == 'EDU':
                        ite_l_pred = np.quantile(a=y0_posteriors, q=0.05, axis=1)
                        ite_r_pred = np.quantile(a=y0_posteriors, q=0.95, axis=1)
                        itv0 = ite_r_pred - ite_l_pred

                        ite_l_pred = np.quantile(a=y1_posteriors, q=0.05, axis=1)
                        ite_r_pred = np.quantile(a=y1_posteriors, q=0.95, axis=1)
                        itv1 = ite_r_pred - ite_l_pred

                        plot_cis_edu(itv0, itv1, x_test[:, 23])

    if not(params['sweep']):
        print('\n\n--------------RESULTS--------------')
        print('Dataset: ', str(params['dataset']) + '\n')

        if params['CMGP']:
            print('CMGP')
            print('\tsqrt(PEHE): ', np.round(pehe_cmgp.mean(), 4), np.round(sem(pehe_cmgp), 4))
            print('\tLL: ', np.round(ll_cmgp.mean(), 4), np.round(sem(ll_cmgp), 4))
            if params['dataset'] == 'SyntheticCI' or params['dataset'] == 'EDU' or params['dataset'] == 'IHDP':
                print('\tIoU: ', np.round(iou_cmgp.mean(), 4), np.round(sem(iou_cmgp), 4))
                print('\tAccuracy: ', np.round(accuracy_cmgp.mean(), 4), np.round(sem(accuracy_cmgp), 4))
            print('\tCoverage: ', np.round(coverage_cmgp.mean(), 4), np.round(sem(coverage_cmgp), 4))
        if params['CEVAE']:
            print('CEVAE')
            print('\tsqrt(PEHE): ', np.round(pehe_cevae.mean(), 4), np.round(sem(pehe_cevae), 4))
            print('\tLL: ', np.round(ll_cevae.mean(), 4), np.round(sem(ll_cevae), 4))
            if params['dataset'] == 'SyntheticCI' or params['dataset'] == 'EDU' or params['dataset'] == 'IHDP':
                print('\tIoU: ', np.round(iou_cevae.mean(), 4), np.round(sem(iou_cevae), 4))
                print('\tAccuracy: ', np.round(accuracy_cevae.mean(), 4), np.round(sem(accuracy_cevae), 4))
            print('\tCoverage: ', np.round(coverage_cevae.mean(), 4), np.round(sem(coverage_cevae), 4))
        if params['GANITE']:
            print('GANITE')
            print('\tsqrt(PEHE): ', np.round(pehe_ganite.mean(), 4), np.round(sem(pehe_ganite), 4))
            print('\tLL: ', np.round(ll_ganite.mean(), 4), np.round(sem(ll_ganite), 4))
            if params['dataset'] == 'SyntheticCI' or params['dataset'] == 'EDU' or params['dataset'] == 'IHDP':
                print('\tIoU: ', np.round(iou_ganite.mean(), 4), np.round(sem(iou_ganite), 4))
                print('\tAccuracy: ', np.round(accuracy_ganite.mean(), 4), np.round(sem(accuracy_ganite), 4))
            print('\tCoverage: ', np.round(coverage_ganite.mean(), 4), np.round(sem(coverage_ganite), 4))
        if params['FCCN']:
            print('FCCN')
            print('\tsqrt(PEHE): ', np.round(pehe_fccn.mean(), 4), np.round(sem(pehe_fccn), 4))
            print('\tLL: ', np.round(ll_fccn.mean(), 4), np.round(sem(ll_fccn), 4))
            if params['dataset'] == 'SyntheticCI' or params['dataset'] == 'EDU' or params['dataset'] == 'IHDP':
                print('\tIoU: ', np.round(iou_fccn.mean(), 4), np.round(sem(iou_fccn), 4))
            print('\tCoverage: ', np.round(coverage_fccn.mean(), 4), np.round(sem(coverage_fccn), 4))
            print('\tAccuracy: ', np.round(accuracy_fccn.mean(), 4), np.round(sem(accuracy_fccn), 4))
        if params['NOFLITE']:
            print('NOFLITE')
            print('\tsqrt(PEHE): ', np.round(pehe_noflite.mean(), 4), np.round(sem(pehe_noflite), 4))
            print('\tLL: ', np.round(ll_noflite.mean(), 4), np.round(sem(ll_noflite), 4))
            if params['dataset'] == 'SyntheticCI' or params['dataset'] == 'EDU' or params['dataset'] == 'IHDP':
                print('\tIoU: ', np.round(iou_noflite.mean(), 4), np.round(sem(iou_noflite), 4))
                print('\tAccuracy: ', np.round(accuracy_noflite.mean(), 4), np.round(sem(accuracy_noflite), 4))
            print('\tCoverage: ', np.round(coverage_noflite.mean(), 4), np.round(sem(coverage_noflite), 4))
            if params['wandb']:
                wandb.log({'PEHE Average': pehe_noflite.mean()})
                wandb.log({'PEHE SE': sem(pehe_noflite)})
                wandb.log({'LL Average': ll_noflite.mean()})
                wandb.log({'LL SE': sem(ll_noflite)})
                if params['dataset'] == 'SyntheticCI' or params['dataset'] == 'EDU' or params['dataset'] == 'IHDP':
                    wandb.log({'IoU Average': iou_noflite.mean()})
                    wandb.log({'IoU SE': sem(iou_noflite)})
                    wandb.log({'Accuracy Average': accuracy_noflite.mean()})
                    wandb.log({'Accuracy SE': sem(accuracy_noflite)})
                wandb.log({'Coverage Average': coverage_noflite.mean()})
                wandb.log({'Coverage SE': sem(coverage_noflite)})
