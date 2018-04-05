# Adversarial

This repo contains code for our unsupervised domain adaptation method for relation extraction.

**Note:** Examples of the data format can be found in the data/ folder.

## Usage
### Training

```
python train_final_cnn.py --num_epochs 50 --checkpoint_dir /checkpoint/dir/experiments/checkpoints/ --checkpoint_name my_checkpoint --min_df 5 --lr 0.001 --penalty 0. --adv_train_data_X  /my/data/data1/all_train.txt --adv_test_data_X  /my/data/biogrid_train_test/all_test.txt --test_data /my/data/test_data.txt --train_data /my/data/train_data.txt --train_data_X /my/data/data2/train.txt --val_data_X /my/data/data2/test.txt --num_iters 10000 --num_disc_updates 1 --emb_reg --adv --pos_reg --hidden_state 128 --adv --seed 42
```

```
usage: train_final_cnn.py [-h] [--num_epochs NUM_EPOCHS]
                          [--hidden_state HIDDEN_STATE]
                          [--checkpoint_dir CHECKPOINT_DIR]
                          [--checkpoint_name CHECKPOINT_NAME]
                          [--min_df MIN_DF] [--lr LR] [--penalty PENALTY]
                          [--train_data_X TRAIN_DATA_X]
                          [--train_data TRAIN_DATA] [--test_data TEST_DATA]
                          [--val_data_X VAL_DATA_X]
                          [--adv_train_data_X ADV_TRAIN_DATA_X]
                          [--adv_test_data_X ADV_TEST_DATA_X]
                          [--num_iters NUM_ITERS] [--grad_clip GRAD_CLIP]
                          [--num_disc_updates NUM_DISC_UPDATES] [--seed SEED]
                          [--adv] [--emb_reg] [--pos_reg]

Train Neural Network.

optional arguments:
  -h, --help            show this help message and exit
  --num_epochs NUM_EPOCHS
                        Number of updates to make.
  --hidden_state HIDDEN_STATE
                        LSTM hidden state size.
  --checkpoint_dir CHECKPOINT_DIR
                        Checkpoint directory.
  --checkpoint_name CHECKPOINT_NAME
                        Checkpoint File Name.
  --min_df MIN_DF       Min word count.
  --lr LR               Learning Rate.
  --penalty PENALTY     Regularization Parameter.
  --train_data_X TRAIN_DATA_X
                        Training Data.
  --train_data TRAIN_DATA
                        Training Data.
  --test_data TEST_DATA
                        Training Data.
  --val_data_X VAL_DATA_X
                        Validation Data.
  --adv_train_data_X ADV_TRAIN_DATA_X
                        Validation Data.
  --adv_test_data_X ADV_TEST_DATA_X
                        Validation Data.
  --num_iters NUM_ITERS
                        Validation Data.
  --grad_clip GRAD_CLIP
                        Gradient Clip Value.
  --num_disc_updates NUM_DISC_UPDATES
                        Number of time to update discriminator.
  --seed SEED           Random seed.
  --adv                 Adversarial training?
  --emb_reg             Regularize word embeddings?
  --pos_reg             Regularize pos embeddings?
```

## Acknowledgements

> Anthony Rios, Ramakanth Kavuluru, and Zhiyong Lu. "Generalizing Biomedical Relation Classification with Neural Adversarial Domain Adaptation". Bioinformatics 2018

```
@article{rios2018advrel,
  title={Generalizing Biomedical Relation Classification with Neural Adversarial Domain Adaptation},
  author={Rios, Anthony and Kavuluru, Ramakanth and Lu, Zhiyong},
  journal={Bioinformatics},
  year={2018}
}
```

Written by Anthony Rios (anthonymrios at gmail dot com)
