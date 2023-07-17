import random
import numpy as np
import scipy
from scipy import sparse, stats
from scipy.special import expit
import pandas as pd
import os
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset, iteration=None, batch_size=64):
        super().__init__()
        if iteration != None:
            self.dataset = dataset(iteration=iteration)
        else:
            self.dataset = dataset()

        self.batch_size = batch_size

        x_train, yf_train, t_train = self.dataset.get_train_data()
        x_val, yf_val, t_val = self.dataset.get_val_data()
        x_train_val, yf_train_val, t_train_val = self.dataset.get_train_val_data()
        x_test, yf_test, yc_test, t_test = self.dataset.get_test_data()

        if dataset.__name__ == 'SyntheticCI' or dataset.__name__ == 'EDU' or dataset.__name__ == 'IHDP':
            ite_l_test, ite_r_test = self.dataset.get_ci_data()
            self.ite_l_test = torch.from_numpy(ite_l_test).float()
            self.ite_r_test = torch.from_numpy(ite_r_test).float()

        # # Standardize
        # # Only data without CIs (to keep left and right limits)
        # standardize = False
        # # if not(dataset.__name__ in ['Twins', 'SyntheticCI', 'IHDP', 'EDU']):
        # #     standardize = True     # Standardizing Y is not compatible with 90% CI intervals
        # if standardize:
        #     # Normalize and make tensor
        #     # Covariates x
        #     scaler = StandardScaler().fit(x_train)
        #     x_train = scaler.transform(x_train)
        #     x_val = scaler.transform(x_val)
        #     x_test = scaler.transform(x_test)
        #
        #     # Outcomes for t=0
        #     scaler = StandardScaler().fit(yf_train[t_train == 0])
        #     yf_train[t_train == 0] = scaler.transform(yf_train[t_train == 0])
        #     yf_val[t_val == 0] = scaler.transform(yf_val[t_val == 0])
        #     yf_test[t_test == 0] = scaler.transform(yf_test[t_test == 0])
        #     yc_test[t_test == 1] = scaler.transform(yc_test[t_test == 1])
        #
        #     # Outcomes for t=1
        #     scaler = StandardScaler().fit(yf_train[t_train == 1])
        #     yf_train[t_train == 1] = scaler.transform(yf_train[t_train == 1])
        #     yf_val[t_val == 1] = scaler.transform(yf_val[t_val == 1])
        #     yf_test[t_test == 1] = scaler.transform(yf_test[t_test == 1])
        #     yc_test[t_test == 0] = scaler.transform(yc_test[t_test == 0])

        self.x_train_tensor = torch.from_numpy(x_train).float()
        self.yf_train_tensor = torch.from_numpy(yf_train).float()
        self.t_train_tensor = torch.from_numpy(t_train).float()

        self.x_val_tensor = torch.from_numpy(x_val).float()
        self.yf_val_tensor = torch.from_numpy(yf_val).float()
        self.t_val_tensor = torch.from_numpy(t_val).float()

        self.x_train_val_tensor = torch.from_numpy(x_train_val).float()
        self.yf_train_val_tensor = torch.from_numpy(yf_train_val).float()
        self.t_train_val_tensor = torch.from_numpy(t_train_val).float()

        self.x_test_tensor = torch.from_numpy(x_test).float()
        self.yf_test_tensor = torch.from_numpy(yf_test).float()
        self.yc_test_tensor = torch.from_numpy(yc_test).float()
        self.t_test_tensor = torch.from_numpy(t_test).float()

        self.training_dataset = TensorDataset(self.x_train_tensor, self.yf_train_tensor, self.t_train_tensor)
        self.validation_dataset = TensorDataset(self.x_val_tensor, self.yf_val_tensor, self.t_val_tensor)
        self.train_val_dataset = TensorDataset(self.x_train_val_tensor, self.yf_train_val_tensor,
                                               self.t_train_val_tensor)
        self.testing_dataset = TensorDataset(self.x_test_tensor, self.yf_test_tensor, self.yc_test_tensor,
                                             self.t_test_tensor)

    def train_dataloader(self):
        train_loader = DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.validation_dataset, batch_size=self.batch_size)
        return val_loader

    def train_val_dataloader(self):
        train_val_loader = DataLoader(self.train_val_dataset, batch_size=self.batch_size)
        return train_val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.testing_dataset, batch_size=16)
        # test_loader = DataLoader(self.testing_dataset, batch_size=self.batch_size)
        return test_loader


class IHDP(Dataset):
    def __init__(self, iteration=0):
        """
        Class in charge of reading data & ordering data into correct structures
        """
        super().__init__()
        self.input_size = 25

        # Reading data from CSV file
        data_train = np.load('datasets/IHDP100/ihdp_npci_1-100.train.npz')
        data_test = np.load('datasets/IHDP100/ihdp_npci_1-100.test.npz')

        # Putting data in correct structure
        self.x_train_val = data_train['x'][:, :, iteration]
        self.x_train_val[:, 13] = self.x_train_val[:, 13] - 1       # Correct this ]dummy
        self.t_train_val = data_train['t'][:, iteration]
        self.yf_train_val = data_train['yf'][:, iteration][:, np.newaxis]
        # yc_train = data_train['ycf'][:, iteration][:, np.newaxis]

        # Test data
        self.x_test = data_test['x'][:, :, iteration]
        self.x_test[:, 13] = self.x_test[:, 13] - 1
        self.t_test = data_test['t'][:, iteration]
        # self.yf_test = data_test['yf'][:, iteration][:, np.newaxis]
        # self.yc_test = data_test['ycf'][:, iteration][:, np.newaxis]
        self.mu0 = data_test['mu0'][:, iteration][:, np.newaxis]
        self.mu1 = data_test['mu1'][:, iteration][:, np.newaxis]
        self.yf_test = np.zeros_like(self.mu0)
        self.yc_test = np.zeros_like(self.mu0)
        self.yf_test[self.t_test == 0] = self.mu0[self.t_test == 0]
        self.yf_test[self.t_test == 1] = self.mu1[self.t_test == 1]
        self.yc_test[self.t_test == 1] = self.mu0[self.t_test == 1]
        self.yc_test[self.t_test == 0] = self.mu1[self.t_test == 0]

        # 90% CI:
        self.ite_l_test = (self.mu1 - self.mu0) - 1.959963984540 * 1
        self.ite_r_test = (self.mu1 - self.mu0) + 1.959963984540 * 1

        self.x_train, self.x_val, self.yf_train, self.yf_val, self.t_train, self.t_val = \
            train_test_split(self.x_train_val, self.yf_train_val, self.t_train_val, test_size=0.10, random_state=0)

    def get_train_data(self):
        """
        Returns train data
        """
        return self.x_train, self.yf_train, self.t_train

    def get_train_val_data(self):
        """
        Returns train data
        """
        return self.x_train_val, self.yf_train_val, self.t_train_val

    def get_val_data(self):
        """
        Returns validation data
        """
        return self.x_val, self.yf_val, self.t_val

    def get_test_data(self):
        """
        Returns test data
        """
        return self.x_test, self.yf_test, self.yc_test, self.t_test

    def get_ci_data(self):
        return self.ite_l_test, self.ite_r_test


class EDU(Dataset):
    def __init__(self, iteration=0):
        """
        Class in charge of reading data & ordering data into correct structures
        """
        super().__init__()
        self.input_size = 32

        # Reading data from CSV file
        data_train = np.load('datasets/EDU/edu_train.npy')[:, :, iteration]
        data_test = np.load('datasets/EDU/edu_test.npy')[:, :, iteration]

        # Putting data in correct structure
        self.x_train_val = data_train[:, 7:]
        self.t_train_val = data_train[:, 3]
        self.yf_train_val = data_train[:, 0][:, None]

        # Test data
        self.x_test = data_test[:, 7:]
        self.t_test = np.zeros(len(self.x_test))    # Placeholder
        # self.t_test[0] = 1                          # One 1 for StandardScaler
        self.mu0 = data_test[:, 4][:, None]         # Use mu0/mu1 for evaluation
        self.mu1 = data_test[:, 5][:, None]
        self.yf_test = np.zeros_like(self.mu0)
        self.yc_test = np.zeros_like(self.mu0)
        self.yf_test[self.t_test == 0] = self.mu0[self.t_test == 0]
        self.yf_test[self.t_test == 1] = self.mu1[self.t_test == 1]
        self.yc_test[self.t_test == 1] = self.mu0[self.t_test == 1]
        self.yc_test[self.t_test == 0] = self.mu1[self.t_test == 0]

        # Add CI left and right for test set
        M = self.x_test[:, 23]
        # widthexp = (stats.expon.ppf(0.95, scale=0.5) - stats.expon.ppf(0.05, scale=0.5)) * (
        #         2 - S)  # width for exponential distribution

        # Expected ITE + Noise Mu 1 - Noise Mu 0
        ite_samples = (self.mu1 - self.mu0).repeat(100000, axis=1) \
                      + (2 - M)[:, None].repeat(100000, axis=1) * np.random.exponential(0.5, (len(M), 100000)) \
                      - (2 - M)[:, None].repeat(100000, axis=1) * np.random.normal(0, 0.5, (len(M), 100000))
        self.ite_l_test = np.percentile(a=ite_samples, q=5, axis=1)[:, None]
        self.ite_r_test = np.percentile(a=ite_samples, q=95, axis=1)[:, None]

        # No counterfactual Y during training/validation
        self.x_train, self.x_val, self.yf_train, self.yf_val, self.t_train, self.t_val = \
            train_test_split(self.x_train_val, self.yf_train_val, self.t_train_val, test_size=0.10, random_state=0)

    def get_train_data(self):
        """
        Returns train data
        """
        return self.x_train, self.yf_train, self.t_train

    def get_train_val_data(self):
        """
        Returns train data
        """
        return self.x_train_val, self.yf_train_val, self.t_train_val

    def get_val_data(self):
        """
        Returns validation data
        """
        return self.x_val, self.yf_val, self.t_val

    def get_test_data(self):
        """
        Returns test data
        """
        return self.x_test, self.yf_test, self.yc_test, self.t_test

    def get_ci_data(self):
        return self.ite_l_test, self.ite_r_test


class Twins(Dataset):
    def __init__(self, iteration=0):
        """
        Class in charge of reading data & ordering data into correct structures
        """
        super().__init__()
        self.input_size = 30

        # Process following GANITE
        # See: https://github.com/jsyoon0823/GANITE/blob/master/data_loading.py
        ori_data = np.loadtxt('datasets/Twins/Twin_data.csv.gz', delimiter=",", skiprows=1)

        # Define features
        x = ori_data[:, :30]
        no, dim = x.shape

        # Define potential outcomes
        potential_y = ori_data[:, 30:]
        # Die within 1 year = 1, otherwise = 0
        potential_y = np.array(potential_y < 9999, dtype=float)

        ## Assign treatment
        coef = np.random.uniform(-0.01, 0.01, size=[dim, 1])
        prob_temp = expit(np.matmul(x, coef) + np.random.normal(0, 0.01, size=[no, 1]))

        prob_t = prob_temp / (2 * np.mean(prob_temp))
        prob_t[prob_t > 1] = 1

        t = np.random.binomial(1, prob_t, [no, 1])
        t = t.reshape([no, ])

        ## Define observable outcomes
        y = np.zeros([no, 1])
        y = np.transpose(t) * potential_y[:, 1] + np.transpose(1 - t) * potential_y[:, 0]
        y = np.reshape(np.transpose(y), [no, ])

        # Define counterfactuals:
        yc = np.zeros([no, 1])
        yc = np.transpose(t) * potential_y[:, 0] + np.transpose(1 - t) * potential_y[:, 1]
        yc = np.reshape(np.transpose(yc), [no, ])

        # Putting data in correct structure
        self.X = x
        self.T = t
        self.Yf = y[:, np.newaxis]
        self.Yc = yc[:, np.newaxis]

        # random.seed(a=42)
        # Train, validation and test splits
        # No counterfactual Y during training/validation
        x_train, self.x_test, yf_train, self.yf_test, _, self.yc_test, t_train, self.t_test = train_test_split(self.X,
                                                                                               self.Yf, self.Yc,
                                                                                               self.T, test_size=0.2,
                                                                                               random_state=0)

        # self.x_train, self.x_val, self.yf_train, self.yf_val, self.t_train, self.t_val = train_test_split(x_train,
        #                                                                                                   yf_train,
        #                                                                                                   t_train,
        #                                                                                                   test_size=0.1,
        #                                                                                                   random_state=0)

    def get_train_data(self):
        """
        Returns train data
        """
        return self.x_train, self.yf_train, self.t_train

    def get_val_data(self):
        """
        Returns validation data
        """
        return self.x_val, self.yf_val, self.t_val

    def get_test_data(self):
        """
        Returns test data
        """
        return self.x_test, self.yf_test, self.yc_test, self.t_test


class News(Dataset):
    def __init__(self, iteration=0):
        """
        Class in charge of reading data & ordering data into correct structures
        """
        super().__init__()
        self.input_size = 3477

        # Reading data from CSV file
        # Code from https://github.com/clinicalml/cfrnet/blob/9daea5d8ba7cb89f413065c5ce7f0136f0e84c9b/cfr/util.py#L67
        fname = 'datasets/News/NEWS_csv/csv/topic_doc_mean_n5000_k3477_seed_' + str(iteration + 1) + '.csv'
        data = np.loadtxt(open(fname + '.y', "rb"), delimiter=",")   # t, y_f, y_cf, mu0, mu1
        def load_sparse(fname):
            """ Load sparse data set """
            E = np.loadtxt(open(fname, "rb"), delimiter=",")
            H = E[0, :]
            n = int(H[0])
            d = int(H[1])
            E = E[1:, :]
            S = sparse.coo_matrix((E[:, 2], (E[:, 0] - 1, E[:, 1] - 1)), shape=(n, d))
            S = S.todense()

            return S

        self.X = load_sparse(fname + '.x')

        # pca = PCA(n_components=50)
        # self.X = pca.fit_transform(self.X)
        # self.input_size = pca.n_components

        # Putting data in correct structure
        self.T = data[:, 0]
        self.Yf = data[:, 1][:, np.newaxis]
        self.Yc = data[:, 2][:, np.newaxis]

        # random.seed(a=42)
        # Train, validation and test splits
        # No counterfactual Y during training/validation
        self.x_train_val, self.x_test, self.yf_train_val, self.yf_test, _, self.yc_test, self.t_train_val, self.t_test =\
            train_test_split(self.X, self.Yf, self.Yc, self.T, test_size=0.2, random_state=0)

        self.x_train, self.x_val, self.yf_train, self.yf_val, self.t_train, self.t_val = \
            train_test_split(self.x_train_val, self.yf_train_val, self.t_train_val, test_size=0.1, random_state=0)


    def get_train_data(self):
        """
        Returns train data
        """
        return self.x_train, self.yf_train, self.t_train

    def get_train_val_data(self):
        """
        Returns train data
        """
        return self.x_train_val, self.yf_train_val, self.t_train_val

    def get_val_data(self):
        """
        Returns validation data
        """
        return self.x_val, self.yf_val, self.t_val

    def get_test_data(self):
        """
        Returns test data
        """
        return self.x_test, self.yf_test, self.yc_test, self.t_test


class LBIDD:
    def __init__(self, iteration=0):
        """
        Class in charge of reading data & ordering data into correct structures
        """
        iteration_ids = ['083eb59efd2d488fb715164a7efeb5c7', '0b3676e2530b4427965e0c612bf41b08',
                         '170e8ab28ae845cfa1db3b3bae23e4cd', '1b08598fcad948c18d671c6cfd912d80',
                         '33cce0e4773e49fd8299eb66206cd30f', '3d9b216fa14f452eb323e851439b3a5c',
                         '476784608fcf4ccda016085f196fd652', '5567d3f92a93432c8041d977bb77e9d3',
                         '67b89e30a9ac400bb457b399a0ec3a3f', '7736d35dc6f24d53902a218cb9e9ac0a',
                         '7ddeadf1bc2e47679c570b3399799b47', '9f54591869a3418f83465edc02bc27a2',
                         'a13b3076894741c986af688df52d7d4a', 'b902cc756a5b45ebb48c1f4270328ee9',
                         'fd874f0db7cf47d2aea3e4f29b82b564']

        # Reading data from CSV file
        X_data = pd.read_csv('datasets/LBIDD/x.csv', delimiter=',', index_col='sample_id')
        iteration_id = iteration_ids[iteration]
        Yf_data = pd.read_csv('datasets/LBIDD/censoring/' + iteration_id + '.csv', delimiter=',', index_col='sample_id')
        censored = Yf_data['y'].isna().copy()
        Yf_data = Yf_data.drop(labels='y', axis=1)
        Ycf_data = pd.read_csv('datasets/LBIDD/censoring/' + iteration_id + '_cf.csv', delimiter=',', index_col='sample_id')

        data = pd.merge(X_data, Yf_data, on='sample_id')
        data = pd.merge(data, Ycf_data, on='sample_id')

        self.X = data.drop(labels=['z', 'y0', 'y1'], axis=1)
        self.Yf = data.y0.copy()
        self.Yf[data.z == 1] = data.y1[data.z == 1]
        self.Yc = data.y1.copy()
        self.Yc[data.z == 1] = data.y0[data.z == 1]
        self.T = data.z.copy()

        self.input_size = self.X.shape[1]

        # Train, validation and test splits
        self.x_train_val, self.x_test, self.yf_train_val, self.yf_test, \
        _, self.yc_test, self.t_train_val, self.t_test = train_test_split(self.X, self.Yf, self.Yc, self.T,
                                                                          test_size=0.2, random_state=0)

        # Drop censored instances from train/val:
        self.x_train_val = self.x_train_val[censored == False]
        self.yf_train_val = self.yf_train_val[censored == False]
        self.t_train_val = self.t_train_val[censored == False]

        # Convert to numpy:
        self.x_train_val = self.x_train_val.to_numpy()
        self.x_test = self.x_test.to_numpy()
        self.yf_train_val = self.yf_train_val.to_numpy()[:, np.newaxis]
        self.yf_test = self.yf_test.to_numpy()[:, np.newaxis]
        self.yc_test = self.yc_test.to_numpy()[:, np.newaxis]
        self.t_train_val = self.t_train_val.to_numpy()
        self.t_test = self.t_test.to_numpy()

        self.x_train, self.x_val, self.yf_train, self.yf_val, self.t_train, self.t_val = train_test_split(self.x_train_val,
                                                                                                          self.yf_train_val,
                                                                                                          self.t_train_val,
                                                                                                          test_size=0.1,
                                                                                                          random_state=0)

    def get_train_data(self):
        """
        Returns train data
        """
        return self.x_train, self.yf_train, self.t_train

    def get_train_val_data(self):
        """
        Returns train data
        """
        return self.x_train_val, self.yf_train_val, self.t_train_val

    def get_val_data(self):
        """
        Returns validation data
        """
        return self.x_val, self.yf_val, self.t_val

    def get_test_data(self):
        """
        Returns test data
        """
        return self.x_test, self.yf_test, self.yc_test, self.t_test


class Synthetic(Dataset):
    def __init__(self, size=2000, input_size=20):
        """
        Class in charge of reading data & ordering data into correct structures
        """
        super().__init__()
        self.input_size = input_size

        # Putting data in correct structure
        self.X = np.random.multivariate_normal(np.zeros(input_size), np.diag(np.ones(input_size)), size)
        self.T = np.random.binomial(1, 0.5, (size))     # Todo: (size, 1) everywhere
        # Y0 = 2**(np.sum(np.random.uniform(-0.05, 0.05, 25) * self.X, axis=1)[:, None] + np.random.uniform(0, 0.5) * -1) \
        #    + np.random.normal(0, 0.2, (size, 1))
        # Y1 = 2**(np.sum(np.random.uniform(-0.05, 0.05, 25) * self.X, axis=1)[:, None] + np.random.uniform(0.5, 1) * 1) \
        #    + np.random.normal(0, 0.1, (size, 1))
        means = np.sum(np.random.uniform(-0.05, 0.05, input_size) * self.X, axis=1)[:, None]**2
        sigmas = 2 ** (np.sum(np.random.uniform(-0.05, 0.05, input_size) * self.X, axis=1))[:, None]
        Y0 = np.random.lognormal(mean=means, sigma=sigmas, size=(size, 1))

        means = np.sum(np.random.uniform(-0.05, 0.05, input_size) * self.X, axis=1)[:, None]
        sigmas = 2 ** (np.sum(np.random.uniform(-0.05, 0.05, input_size) * self.X, axis=1))[:, None]
        Y1 = - np.random.lognormal(mean=means, sigma=sigmas, size=(size, 1))

        # Y0 = Y0 * p + Y0_ * (1 - p)
        # Y1 = Y1 * p + Y1_ * (1 - p)
        # sns.kdeplot(Y0.flatten())
        # sns.kdeplot(Y1.flatten())
        # plt.show()

        self.Yf = np.zeros_like(Y0)
        self.Yf[self.T == 0] = Y0[self.T == 0]
        self.Yf[self.T == 1] = Y1[self.T == 1]
        self.Yc = np.zeros_like(Y0)
        self.Yc[self.T == 0] = Y1[self.T == 0]
        self.Yc[self.T == 1] = Y0[self.T == 1]

        # random.seed(a=42)
        # Train, validation and test splits
        # No counterfactual Y during training/validation
        x_train, self.x_test, yf_train, self.yf_test, _, self.yc_test, t_train, self.t_test = train_test_split(self.X,
                                                                                               self.Yf, self.Yc,
                                                                                               self.T, test_size=0.2,
                                                                                               random_state=0)

        self.x_train, self.x_val, self.yf_train, self.yf_val, self.t_train, self.t_val = train_test_split(x_train,
                                                                                                          yf_train,
                                                                                                          t_train,
                                                                                                          test_size=0.1,
                                                                                                          random_state=0)

    def get_train_data(self):
        """
        Returns train data
        """
        return self.x_train, self.yf_train, self.t_train

    def get_val_data(self):
        """
        Returns validation data
        """
        return self.x_val, self.yf_val, self.t_val

    def get_test_data(self):
        """
        Returns test data
        """
        return self.x_test, self.yf_test, self.yc_test, self.t_test


class SyntheticGMM(Dataset):
    def __init__(self, size=1000, input_size=10):
        """
        Class in charge of reading data & ordering data into correct structures
        """
        super().__init__()
        self.input_size = input_size

        # Putting data in correct structure
        self.X = np.random.multivariate_normal(np.zeros(input_size), np.diag(np.ones(input_size)), size)
        self.T = np.random.binomial(1, 0.5, (size))

        # mu0 = np.sum(np.random.uniform(-0.1, 0.1, 25) * self.X, axis=1)
        # mu1 = np.sum(np.random.uniform(-0.1, 0.1, 25) * self.X, axis=1) - 5
        # mu2 = np.sum(np.random.uniform(-0.1, 0.1, 25) * self.X, axis=1) + 5
        mu0 = 0
        mu1 = 2
        mu2 = 4

        # sig0 = 2**(np.sum(np.random.uniform(-0.05, 0.05, 25) * self.X, axis=1) + np.random.uniform(0, 0.5) * -1)
        # sig1 = 2**(np.sum(np.random.uniform(-0.1, 0.1, 25) * self.X, axis=1) + np.random.uniform(0, 0.5) * -1)
        # sig2 = 2**(np.sum(np.random.uniform(-0.15, 0.15, 25) * self.X, axis=1) + np.random.uniform(0, 0.5) * -1)
        sig0 = 1
        sig1 = 0.5
        sig2 = 0.25

        # Sample + mix them per class
        k0 = 0.
        k1 = 0.5
        k2 = 0.5
        k = np.random.choice(np.arange(0, 3), p=[k0, k1, k2], size=(size,))
        Y0 = np.zeros((size, 1))
        Y0[k == 0] = np.random.normal(mu0, sig0, (size, 1))[k == 0]# + 2
        Y0[k == 1] = np.random.normal(mu1, sig1, (size, 1))[k == 1]# + 4
        Y0[k == 2] = np.random.normal(mu2, sig2, (size, 1))[k == 2]# + 6
        k0 = 0.
        k1 = 0.5
        k2 = 0.5
        Y1 = np.zeros((size, 1))
        k = np.random.choice(np.arange(0, 3), p=[k0, k1, k2], size=(size,))
        Y1[k == 0] = np.random.normal(mu0, sig0, (size, 1))[k == 0]# - 5
        Y1[k == 1] = np.random.normal(mu1, sig1, (size, 1))[k == 1]# - 2
        Y1[k == 2] = np.random.normal(mu2, sig2, (size, 1))[k == 2]# - 2
        # Y0 = Y0[:, np.newaxis]
        # Y1 = Y1[:, np.newaxis]

        # sns.kdeplot(Y0.flatten())
        # sns.kdeplot(Y1.flatten())
        # plt.show()

        self.Yf = np.zeros_like(Y0)
        self.Yf[self.T == 0] = Y0[self.T == 0]
        self.Yf[self.T == 1] = Y1[self.T == 1]
        self.Yc = np.zeros_like(Y0)
        self.Yc[self.T == 0] = Y1[self.T == 0]
        self.Yc[self.T == 1] = Y0[self.T == 1]

        # random.seed(a=42)
        # Train, validation and test splits
        # No counterfactual Y during training/validation
        x_train, self.x_test, yf_train, self.yf_test, _, self.yc_test, t_train, self.t_test = train_test_split(self.X,
                                                                                               self.Yf, self.Yc,
                                                                                               self.T, test_size=0.2,
                                                                                               random_state=0)

        self.x_train, self.x_val, self.yf_train, self.yf_val, self.t_train, self.t_val = train_test_split(x_train,
                                                                                                          yf_train,
                                                                                                          t_train,
                                                                                                          test_size=0.1,
                                                                                                          random_state=0)

    def get_train_data(self):
        """
        Returns train data
        """
        return self.x_train, self.yf_train, self.t_train

    def get_val_data(self):
        """
        Returns validation data
        """
        return self.x_val, self.yf_val, self.t_val

    def get_test_data(self):
        """
        Returns test data
        """
        return self.x_test, self.yf_test, self.yc_test, self.t_test


class SyntheticCI(Dataset):
    def __init__(self, size=5000, input_size=50):
        super().__init__()
        self.input_size = input_size
        standardize = True

        train_fraction = 0.8
        test_fraction = 1. - train_fraction

        # Putting data in correct structure
        x_cov = np.random.uniform(-0.1, 0.1, (input_size, input_size))      # np.diag(np.ones(input_size))
        self.X = np.random.multivariate_normal(np.zeros(input_size), 0.5*(x_cov + x_cov.T), size)
        # self.X = np.random.binomial(1, 0.5, (size, input_size))
        prob = 1 / (1 + 1 / np.exp(np.sum(np.random.uniform(-1, 1, input_size) * self.X, axis=1)))
        self.T = np.random.binomial(1, prob, (size))

        # Split train-test:
        self.x_train_val = self.X[0:int(size * train_fraction)]
        self.t_train_val = self.T[0:int(size * train_fraction)]

        self.x_test = self.X[:int(size * test_fraction)]
        self.t_test = self.T[:int(size * test_fraction)]

        # Sample outcomes:
        self.yf_train_val = np.zeros((int(size * train_fraction), 1))

        outcomes_test = np.zeros((int(size * test_fraction), 2))
        self.ite_test = np.zeros((int(size * test_fraction), 1))
        self.ite_l_test = np.zeros((int(size * test_fraction), 1))
        self.ite_r_test = np.zeros((int(size * test_fraction), 1))

        coef0 = np.random.uniform(-0.5, 0.5, input_size)
        coef1 = np.random.uniform(-0.5, 0.5, input_size)
        coef_sigma = np.random.uniform(-0.5, 0.5, input_size)
        coef_scale = np.random.uniform(-0.5, 0.5, input_size)

        # Train-val - 80%
        for i in range(int(size * train_fraction)):
            if self.t_train_val[i] == 0:
                sigma = np.abs(np.sum(coef_sigma * self.x_train_val[i, :]))      # 0.5
                self.yf_train_val[i, 0] = np.sum(coef0 * self.x_train_val[i, :]) ** 2 + np.random.laplace(
                    loc=0, scale=sigma, size=1)  # np.random.uniform(low=0-sigma, high=0+sigma, size=1)
            else:
                exp_scale = 2**np.sum(coef_scale * self.x_train_val[i, :])    # 2
                self.yf_train_val[i, 0] = np.sum(coef1 * self.x_train_val[i, :]) ** 2 + np.random.exponential(
                    scale=exp_scale, size=1)

        # Standardize:
        if standardize:
            scaler0 = StandardScaler().fit(self.yf_train_val[self.t_train_val == 0])
            self.yf_train_val[self.t_train_val == 0] = scaler0.transform(self.yf_train_val[self.t_train_val == 0])
            scaler1 = StandardScaler().fit(self.yf_train_val[self.t_train_val == 1])
            self.yf_train_val[self.t_train_val == 1] = scaler1.transform(self.yf_train_val[self.t_train_val == 1])

        # Test - 20%
        for i in range(int(size * test_fraction)):
            sigma = np.abs(np.sum(coef_sigma * self.x_test[i, :]))       # 0.5
            exp_scale = 2**np.sum(coef_scale * self.x_test[i, :])   # 2

            # Noiseless outcomes:
            mu0 = np.sum(coef0 * self.x_test[i, :]) ** 2
            mu1 = np.sum(coef1 * self.x_test[i, :]) ** 2
            if standardize:
                outcomes_test[i, 0] = scaler0.transform((mu0 + 0).reshape(1, -1))
                outcomes_test[i, 1] = scaler1.transform((mu1 + exp_scale).reshape(1, -1))     # Exponential
            else:
                outcomes_test[i, 0] = mu0 + 0
                outcomes_test[i, 1] = mu1 + exp_scale
            self.ite_test[i, 0] = outcomes_test[i, 1] - outcomes_test[i, 0]

            # Simulate confidence intervals:
            samples0 = mu0.repeat(100000) + np.random.laplace(0, sigma, size=100000)    # np.random.uniform(low=0-sigma, high=0+sigma, size=100000)
            samples1 = mu1.repeat(100000) + np.random.exponential(exp_scale, size=100000)
            if standardize:
                samples0 = scaler0.transform(samples0[:, None])[:, 0]
                samples1 = scaler1.transform(samples1[:, None])[:, 0]
            ite_samples = samples1 - samples0
            self.ite_l_test[i, 0] = np.percentile(a=ite_samples, q=5)
            self.ite_r_test[i, 0] = np.percentile(a=ite_samples, q=95)

            if i % int(int(size * test_fraction) / 5) == 0:
                colors = ['blue', 'red', 'orange', 'green', 'black', 'purple']
                sns.kdeplot(ite_samples, color=colors[int(i / int(size * test_fraction / 5))], alpha=0.5)
                # plt.vlines(outcomes_test[i, 0] - outcomes_test[i, 1], 0, 0.95 * np.max(plt.gca().get_ylim()),
                #            color=colors[int(i / int(size * test_fraction / 5))], alpha=0.5)
                plt.errorbar(x=ite_samples.mean(), y=1.005 * np.max(plt.gca().get_ylim()),
                             xerr=[[ite_samples.mean() - self.ite_l_test[i, 0]],
                                   [self.ite_r_test[i, 0] - ite_samples.mean()]],
                             capsize=5, capthick=3, marker='o', color=colors[int(i / int(size * test_fraction / 5))],
                             alpha=0.5)
        plt.show()

        sns.kdeplot(samples0.flatten())
        sns.kdeplot(samples1.flatten())
        sns.kdeplot(ite_samples.flatten())
        plt.show()

        Y0 = outcomes_test[:, 0][:, np.newaxis]
        Y1 = outcomes_test[:, 1][:, np.newaxis]

        # Split observed/counterfactual
        self.yf_test = np.zeros_like(Y0)
        self.yf_test[self.t_test == 0] = Y0[self.t_test == 0]
        self.yf_test[self.t_test == 1] = Y1[self.t_test == 1]
        self.yc_test = np.zeros_like(Y0)
        self.yc_test[self.t_test == 0] = Y1[self.t_test == 0]
        self.yc_test[self.t_test == 1] = Y0[self.t_test == 1]

        # Train and validation splits
        self.x_train, self.x_val, self.yf_train, self.yf_val, self.t_train, self.t_val = \
            train_test_split(self.x_train_val, self.yf_train_val, self.t_train_val, test_size=0.1, random_state=0)

    def get_train_data(self):
        """
        Returns train data
        """
        return self.x_train, self.yf_train, self.t_train

    def get_train_val_data(self):
        """
        Returns train data
        """
        return self.x_train_val, self.yf_train_val, self.t_train_val

    def get_val_data(self):
        """
        Returns validation data
        """
        return self.x_val, self.yf_val, self.t_val

    def get_test_data(self):
        """
        Returns test data
        """
        return self.x_test, self.yf_test, self.yc_test, self.t_test

    def get_ci_data(self):
        return self.ite_l_test, self.ite_r_test
