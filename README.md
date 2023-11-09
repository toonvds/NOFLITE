# NOFLITE: Learning to Predict Individual Treatment Effect Distributions

This repository provides the code for the paper _"NOFLITE: Learning to Predict Individual Treatment Effect Distributions"_ (TMLR 2023). 

The structure of the code is as follows:
```
NOFLITE/
|_ benchmarks/      # Folder containing the benchmark methodologies (CMGP is provided through its own package; BART was ran in R)
  |_ ganite/        
  |_ cevae.py        
  |_ fccn.py        
|_ datasets/        # Folder that should contain the datasets; the paper's appendix provides information where to obtain this data
LICENSE             # MIT License
data_loading.py     # Code to load the data
gaussianization.py  # Code for Gaussianization flow
lossfunctions.py    # Functions for the MMD loss
main.py             # Main executable to run experiment
metrics.py          # Evaluation metrics
noflite.py          # Code for NOFLITE
sigmoid_flow.py     # Code for Sigmoidal Flow
visualize.py        # Code for creating the visualizations

```

## Installation
The ```requirements.txt``` provides the necessary packages.
All code was written for ```python 3.9```.
Weights and Biases ([W&B](https://wandb.com)) is required to log the experiments. 

## Usage
The experiments can be run through the ```main.py```file using the following arguments:
```
python main.py
[--lr [learning rate]]
[--lambda_l1 [l1 regularization]]
[--lambda_l2 [l2 regularization]]
[--batch_size [batch size]]
[--noise_reg_x [x noise regularization]]
[--noise_reg_y [y noise regularization]]
[--hidden_neurons_encoder [hidden neurons in NOFLITE's encoder]]
[--hidden_layers_balancer [hidden layers in NOFLITE's balancer]]
[--hidden_layers_encoder [hidden layers in NOFLITE's encoder]]
[--hidden_layers_prior [hidden layers in NOFLITE's prior]]
[--hidden_neurons_trans [hidden neurons in the flow's transformer]]
[--hidden_neurons_cond [hidden neurons in the flow's conditioner]]
[--hidden_layers_cond [hidden layers in the flow's conditioner]]
[--dense [dense version of the flow]]
[--n_flows [number of flow transformations]]
[--datapoint_num [kernel size for Gaussianization flow]]
[--resid_layers [residual layers for Residual/RQNSF-AR flow]]
[--max_steps [training steps]]
[--flow_type [type of flow transformation]]
[--metalearner [type of metalearner]]
[--lambda_mmd [strength of the MMD regularization]]
[--n_samples [number of samples]]
[--trunc_prob [truncation probability]]
[--dataset [dataset to use for the experiment]]
[--bin_outcome [whether to use a binary outcome]]
[--iterations [number of experiment iterations]]
[--visualize [whether to visualize the model's output]]
[--sweep [whether to perform a hyperparameter sweep]]
[--wandb [whether to use wandb]]
[--NOFLITE [whether to include NOFLITE in the experiment]]
[--CMGP [whether to include CMGP in the experiment]]
[--CEVAE [whether to include CEVAE in the experiment]]
[--GANITE [whether to include GANITE in the experiment]]
[--FCCN [whether to include FCCN in the experiment]]

```
Example usage to train NOFLITE on IHDP:

```python main.py --dataset=IHDP --iterations=100 --n_samples=500 --metalearner=T --flow_type=SigmoidX --batch_size=128 --hidden_neurons_encoder=8 --hidden_layers_balancer=2 --hidden_layers_encoder=3 --hidden_layers_prior=2 --lr=5e-4 --max_steps=5000 --n_flows=0 --lambda_mmd=1 --lambda_l1=1e-3 --lambda_l2=5e-4 --noise_reg_x=0.0 --noise_reg_y=5e-1 --trunc_prob=0.0 --no-sweep --no-wandb --no-visualize --NOFLITE --no-CMGP --no-CEVAE --no-GANITE --no-FCCN```

## Acknowledgements
We build upon code provided by the authors of CCN ([Zhou et al., 2023](https://openreview.net/forum?id=q1Fey9feu7)). 

Zhou, Tianhui, William E. Carson IV, and David Carlson. "Estimating Potential Outcome Distributions with Collaborating Causal Networks." _Transactions on Machine Learning Research_ (2022).

## Citing
Please cite our paper and/or code as follows:
```tex

@InProceedings{noflite2023,
  title = 	 {{NOFLITE: Learning to Predict Individual Treatment Effect Distributions}},
  author =       {Vanderschueren, Toon and Berrevoets, Jeroen and Verbeke, Wouter},
  booktitle = 	 {Transactions on Machine Learning Research},
  year = 	 {2023},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  publisher =    {PMLR},
  url =      {https://openreview.net/forum?id=q1Fey9feu7},
}
```
