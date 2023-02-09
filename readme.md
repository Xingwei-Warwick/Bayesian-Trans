Repository of EACL 2023 paper
=========================
Required packages:
-----
COMET-ATOMIC 2020 BART

Download the COMET code from [here](https://storage.googleapis.com/ai2-mosaic-public/projects/mosaic-kgs/comet-atomic_2020_BART.zip)

dgl>=0.9.1

pytorch>=1.8.0

numpy>=1.24.0

tqdm>=4.64.1

scikit-learn>=1.2.0

transformers>=4.25.1

spacy>=3.5.0

Training of prior
-----------------
This part of code is built upon the example code of dgl libary

    python atomic_prior.py --n-bases 10 --n-hidden 100 --evaluate-every 800 --gpu 0 --graph-batch-size 80000 --expname 0330_100dim

Training of Bayesian-Trans
----

    python main.py --cleanmode --model mure --prior rgcn_4000d_atomic_prior.pkl --kl-scaling 0.05 --cuda 1 --regularization_type mmd --batch 64 --mure-dim 50 --latent-dim 200 --expname mure_matres_sd22221 --sd 22221 --dropout 0.0

Model checkpoints can be downloaded at [this link](https://drive.google.com/drive/folders/1_zddSW2Rbk4fXdp-vjNT4EMpJnm6R1d4?usp=sharing)
Use the checkpoint to inference
----

    python main.py --cleanmode --model mure --kl-scaling 0.010000 --cuda 1 --prior prior/rgcn_4000d_atomic_prior.pkl --dataset matres --batch 64 --mure-dim 50 --regularization_type mmd --latent-dim 200 --expname mure_matres_sd7221 --sd 7221 --skiptraining
