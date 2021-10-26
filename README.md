# Generalized Jensen-Shannon Divergence Loss for Learning with Noisy Labels

The official code for the NeurIPS 2021 paper [Generalized Jensen-Shannon Divergence Loss for Learning with Noisy Labels](https://arxiv.org/abs/2105.04522) 

## Environment Setup
Create conda environment, activate environment, and install additional pip packages 
```bash
conda env create -f gjs_env.yml -n gjs
conda activate gjs
python -m pip install -r requirements.txt
```
## Running Experiments
Please check scripts/ folder for yaml files corresponding to different experiments.


For example, to run JS on 40% symmetric noise on the full CIFAR-10 training set, run the following
```bash
python train.py -c scripts/C10/sym/js-40.yaml \
                --data_dir /path/to/dataset/
```
or GJS on 20% asymmetric noise on CIFAR-100
```bash
python train.py -c scripts/C100/asym/gjs-20.yaml \
                --data_dir /path/to/dataset/
```
or GJS on WebVision
```bash
python train.py -c scripts/WebVision/gjs.yaml \
                --data_dir /path/to/dataset/
```


