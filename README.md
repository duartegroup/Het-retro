# Heterocycle Retrosynthesis

This repository complements our publication "Transfer learning for Heterocycle Synthesis Prediction": https://chemrxiv.org/engage/chemrxiv/article-details/6617d56321291e5d1d9ef449 

## Requirements

The specific version used in this project were:
Python: 3.6.9
Torch Version: 1.2.0
TorchText Version: 0.4.0
ONMT Version: 1.0.0
RDKit: 2019.03.2

## Conda Environemt Setup

```bash
conda create -n het-retro python=3.6
conda activate het-retro
conda install -c rdkit rdkit=2019.03.2 -y
conda install -c pytorch pytorch=1.2.0 -y
git clone https://github.com/ewawieczorek/Het-retro.git
cd Het-retro
pip install -e .
```

## Quickstart

The training and evaluation was performed using [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).
The full documentation of the OpenNMT library can be found [here](http://opennmt.net/OpenNMT-py/). 


### Step 1: Preprocess the data

Start by preparing the Ring and USPTO datasets as described in their respective directories.

#### Single data sets

This preprocessing approach is suitable for pre-training and fine-tuning:

```bash
DATADIR=data/uspto_dataset
onmt_preprocess -train_src $DATADIR/product-train.txt -train_tgt $DATADIR/reactant-train.txt -valid_src $DATADIR/product-valid.txt -valid_tgt $DATADIR/reactant-valid.txt -save_data $DATADIR/uspto -src_seq_length 3000 -tgt_seq_length 3000 -src_vocab_size 3000 -tgt_vocab_size 3000 -share_vocab
```

```bash
DATADIR=data/ring_dataset
onmt_preprocess -train_src $DATADIR/product-train.txt -train_tgt $DATADIR/reactant-train.txt -valid_src $DATADIR/product-valid.txt -valid_tgt $DATADIR/reactant-valid.txt -save_data $DATADIR/sequential -src_seq_length 3000 -tgt_seq_length 3000 -src_vocab_size 3000 -tgt_vocab_size 3000 -share_vocab
```

#### Multi-task data sets

This preprocessing approach is suitable for multi-task learning and mixed fine-tuning:

```bash
DATASET=data/uspto_dataset
DATASET_TRANSFER=data/ring_dataset

onmt_preprocess -train_src ${DATASET}/product-train.txt ${DATASET_TRANSFER}/product-train.txt -train_tgt ${DATASET}/reactant-train.txt ${DATASET_TRANSFER}/reactant-train.txt -train_ids uspto ring  -valid_src ${DATASET_TRANSFER}/product-valid.txt -valid_tgt ${DATASET_TRANSFER}/reactant-valid.txt -save_data ${DATASET_TRANSFER}/multi_task -src_seq_length 3000 -tgt_seq_length 3000 -src_vocab_size 3000 -tgt_vocab_size 3000 -share_vocab

```


The files have been previously tokenized using the tokenization function for the reaction SMILES adapted from https://github.com/pschwllr/MolecularTransformer.


The data consists of parallel precursors (`reactant`) and products (`product`) data containing one reaction per line with tokens separated by a space:

* `reactant-train.txt`
* `product-train.txt`
* `reactant-val.txt`
* `product-val.txt`


After running the preprocessing, the following files are generated:

* `uspto.train.pt`: serialized PyTorch file containing training data
* `uspto.valid.pt`: serialized PyTorch file containing validation data
* `uspto.vocab.pt`: serialized PyTorch file containing vocabulary data


Internally the system never touches the words themselves, but uses these indices.

### Step 2: Train the model

The transformer models were trained using the following hyperparameters:

#### Pretraining

```bash
DATADIR=data/uspto_dataset
onmt_train -data $DATADIR/uspto  \
        -save_model  baseline_model \
        -seed $SEED -gpu_ranks 0  \
        -train_steps 250000 -param_init 0 \
        -param_init_glorot -max_generator_batches 32 \
        -batch_size 6144 -batch_type tokens \
         -normalization tokens -max_grad_norm 0  -accum_count 4 \
        -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam  \
        -warmup_steps 8000 -learning_rate 2 -label_smoothing 0.0 \
        -layers 4 -rnn_size  384 -word_vec_size 384 \
        -encoder_type transformer -decoder_type transformer \
        -dropout 0.1 -position_encoding -share_embeddings  \
        -global_attention general -global_attention_function softmax \
        -self_attn_type scaled-dot -heads 8 -transformer_ff 2048
```

#### Multi-task transfer learning

```bash
DATADIR=data/ring_dataset
WEIGHT1=9
WEIGHT2=1

onmt_train -data $DATADIR/multi_task  \
        -save_model  multi_task_model \
        -data_ids uspto ring --data_weights $WEIGHT1 $WEIGHT2 \
        -seed $SEED -gpu_ranks 0  \
        -train_steps 250000 -param_init 0 \
        -param_init_glorot -max_generator_batches 32 \
        -batch_size 6144 -batch_type tokens \
         -normalization tokens -max_grad_norm 0  -accum_count 4 \
        -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam  \
        -warmup_steps 8000 -learning_rate 2 -label_smoothing 0.0 \
        -layers 4 -rnn_size  384 -word_vec_size 384 \
        -encoder_type transformer -decoder_type transformer \
        -dropout 0.1 -position_encoding -share_embeddings  \
        -global_attention general -global_attention_function softmax \
        -self_attn_type scaled-dot -heads 8 -transformer_ff 2048
```


#### Fine-tuning

```bash
DATADIR=data/ring_dataset
TRAIN_STEPS=6000

onmt_train -data $DATADIR/sequential  \
        -train_from models/baseline_model.pt \
        -save_model  fine_tuned_model \
        -seed $SEED -gpu_ranks 0  \
        -train_steps 250000+$TRAIN_STEPS -param_init 0 \
        -param_init_glorot -max_generator_batches 32 \
        -batch_size 6144 -batch_type tokens \
         -normalization tokens -max_grad_norm 0  -accum_count 4 \
        -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam  \
        -warmup_steps 8000 -learning_rate 2 -label_smoothing 0.0 \
        -layers 4 -rnn_size  384 -word_vec_size 384 \
        -encoder_type transformer -decoder_type transformer \
        -dropout 0.1 -position_encoding -share_embeddings  \
        -global_attention general -global_attention_function softmax \
        -self_attn_type scaled-dot -heads 8 -transformer_ff 2048
```
#### Mixed fine-tuning

```bash
DATADIR=data/ring_dataset
TRAIN_STEPS=6000

onmt_train -data $DATADIR/multi-task  \
        -train_from models/baseline_model.pt \
        -save_model  mixed_fine_tuned_model \
        -seed $SEED -gpu_ranks 0  \
        -train_steps 250000+$TRAIN_STEPS -param_init 0 \
        -param_init_glorot -max_generator_batches 32 \
        -batch_size 6144 -batch_type tokens \
         -normalization tokens -max_grad_norm 0  -accum_count 4 \
        -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam  \
        -warmup_steps 8000 -learning_rate 2 -label_smoothing 0.0 \
        -layers 4 -rnn_size  384 -word_vec_size 384 \
        -encoder_type transformer -decoder_type transformer \
        -dropout 0.1 -position_encoding -share_embeddings  \
        -global_attention general -global_attention_function softmax \
        -self_attn_type scaled-dot -heads 8 -transformer_ff 2048
```

### Step 3: Chemical reaction prediction

To test the model on new reactions run:

```bash
DATADIR=data/ring_dataset
onmt_translate -model models/mixed_fine_tuned_model.pt -src $DATADIR/product-test.txt -output predictions.txt  -n_best 1 -beam_size 5 -max_length 300 -batch_size 64 
```
To perfrom ensemble decoding:

```bash
DATADIR=data/ring_dataset
onmt_translate -model models/baseline_model.pt models/fine_tuned_model.pt -src $DATADIR/product-test.txt -output ensemble_predictions.txt  -n_best 1 -beam_size 5 -max_length 300 -batch_size 64
 
```

## Models

The models need to be downloaded from https://doi.org/10.6084/m9.figshare.25723818 and placed into a models folder.
The models provided are:
* pretrained (baseline) retrosynthesis prediction model
* forward reaction prediction multi-task model (used for round-trip accuracy calculation)
* retrosynthesis prediction multi-task, fine-tuned and mixed fine-tuned models

## Citation

```
@misc{wieczorek_transfer_2024,
	title = {Transfer learning for {Heterocycle} {Synthesis} {Prediction}},
	url = {https://chemrxiv.org/engage/chemrxiv/article-details/6617d56321291e5d1d9ef449},
	doi = {10.26434/chemrxiv-2024-ngqqg},
	publisher = {ChemRxiv},
	author = {Wieczorek, Ewa and Sin, Joshua W. and Holland, Matthew T. O. and Wilbraham, Liam and Perez, Victor S. and Bradley, Anthony and Miketa, Dominik and Brennan, Paul E. and Duarte, Fernanda},
	month = may,
	year = {2024}
}


```


This work is based on OpentNMT-py, if you reuse this code please also cite the underlying code framework.

[OpenNMT: Neural Machine Translation Toolkit](https://arxiv.org/pdf/1805.11462)

[OpenNMT technical report](https://doi.org/10.18653/v1/P17-4012)

```
@inproceedings{opennmt,
  author    = {Guillaume Klein and
               Yoon Kim and
               Yuntian Deng and
               Jean Senellart and
               Alexander M. Rush},
  title     = {Open{NMT}: Open-Source Toolkit for Neural Machine Translation},
  booktitle = {Proc. ACL},
  year      = {2017},
  url       = {https://doi.org/10.18653/v1/P17-4012},
  doi       = {10.18653/v1/P17-4012}
}
```
