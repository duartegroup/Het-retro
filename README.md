# Carbohydrate Transformer

This is the code and data of the "Carbohydrate Transformer: Predicting  Regio- and Stereoselective Reactions using Transfer Learning"

## Requirements

The specific version used in this project were:
Python: 3.6.9
Torch Version: 1.2.0
TorchText Version: 0.4.0
ONMT Version: 1.0.0
RDKit: 2019.03.2

## Conda Environemt Setup

```bash
conda create -n onmt36 python=3.6
conda activate onmt36
conda install -c rdkit rdkit=2019.03.2 -y
conda install -c pytorch pytorch=1.2.0 -y
git clone https://github.com/rxn4chemistry/OpenNMT-py.git
cd OpenNMT-py
git checkout carbohydrate_transformer
pip install -e .
```

## Quickstart

The training and evaluation was performed using [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).
The full documentation of the OpenNMT library can be found [here](http://opennmt.net/OpenNMT-py/). 


### Step 1: Preprocess the data

Start by merging the two uspto training source files into a single file using `python merge_src_splits.py` in `data/uspto_dataset`.

#### Single data sets
```bash
DATADIR=data/uspto_dataset
onmt_preprocess -train_src $DATADIR/src-train.txt -train_tgt $DATADIR/tgt-train.txt -valid_src $DATADIR/src-valid.txt -valid_tgt $DATADIR/tgt-valid.txt -save_data $DATADIR/uspto -src_seq_length 3000 -tgt_seq_length 3000 -src_vocab_size 3000 -tgt_vocab_size 3000 -share_vocab
```

```bash
DATADIR=data/transfer_dataset
onmt_preprocess -train_src $DATADIR/src-train.txt -train_tgt $DATADIR/tgt-train.txt -valid_src $DATADIR/src-valid.txt -valid_tgt $DATADIR/tgt-valid.txt -save_data $DATADIR/sequential -src_seq_length 3000 -tgt_seq_length 3000 -src_vocab_size 3000 -tgt_vocab_size 3000 -share_vocab
```

#### Multi-task data sets

```bash
DATASET=data/uspto_dataset
DATASET_TRANSFER=data/transfer_dataset

onmt_preprocess -train_src ${DATASET}/src-train.txt ${DATASET_TRANSFER}/src-train.txt -train_tgt ${DATASET}/tgt-train.txt ${DATASET_TRANSFER}/tgt-train.txt -train_ids uspto transfer  -valid_src ${DATASET_TRANSFER}/src-valid.txt -valid_tgt ${DATASET_TRANSFER}/tgt-valid.txt -save_data ${DATASET_TRANSFER}/multi_task -src_seq_length 3000 -tgt_seq_length 3000 -src_vocab_size 3000 -tgt_vocab_size 3000 -share_vocab

```


The files have been previously tokenized using the tokenization function for the reaction SMILES is available from https://github.com/pschwllr/MolecularTransformer.


The data consists of parallel precursors (`src`) and products (`tgt`) data containing one reaction per line with tokens separated by a space:

* `src-train.txt`
* `tgt-train.txt`
* `src-val.txt`
* `tgt-val.txt`


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
        -save_model  uspto_model_pretrained \
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
DATADIR=data/transfer_dataset
WEIGHT1=9
WEIGHT2=1

onmt_train -data $DATADIR/multi_task  \
        -save_model  multi_task_model \
        -data_ids uspto transfer --data_weights $WEIGHT1 $WEIGHT2
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


#### Sequential transfer learning

```bash
DATADIR=data/transfer_dataset

onmt_train -data $DATADIR/sequential  \
        -train_from models/upsto_model_pretrained.pt \
        -save_model  sequential_model \
        -seed $SEED -gpu_ranks 0  \
        -train_steps 6000 -param_init 0 \
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
onmt_translate -model uspto_model_pretrained.pt -src $DATADIR/src-test.txt -output predictions.txt  -n_best 1 -beam_size 5 -max_length 300 -batch_size 64 
```

## Pretrained Models

Pretrained models can be found in the `models`folder.

## Citation

```
@article{Pesciullesi2020,
author = "Giorgio Pesciullesi and Philippe Schwaller and Teodoro Laino and Jean-Louis Reymond",
title = "{Carbohydrate Transformer: Predicting Regio- and Stereoselective Reactions Using Transfer Learning}",
year = "2020",
month = "3",
url = "https://chemrxiv.org/articles/preprint/Carbohydrate_Transformer_Predicting_Regio-_and_Stereoselective_Reactions_Using_Transfer_Learning/11935635",
doi = "10.26434/chemrxiv.11935635.v1"
}
```


The Carbohydrate Transformer is based on OpentNMT-py, if you reuse this code please also cite the underlying code framework.

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
