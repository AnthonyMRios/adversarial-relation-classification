import argparse
import random
import pickle
from time import time
import sys
import gensim

import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from models.adversarial_cnn import CNN

from load_data import *

def main():
    parser = argparse.ArgumentParser(description='Train Neural Network.')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of updates to make.')
    parser.add_argument('--hidden_state', type=int, default=128, help='LSTM hidden state size.')
    parser.add_argument('--checkpoint_dir', default='./experiments/exp1/checkpoints/',
                        help='Checkpoint directory.')
    parser.add_argument('--checkpoint_name', default='checkpoint',
                        help='Checkpoint File Name.')
    parser.add_argument('--min_df', type=int, default=5, help='Min word count.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate.')
    parser.add_argument('--penalty', type=float, default=0.0, help='Regularization Parameter.')
    parser.add_argument('--train_data_X', help='Training Data.')
    parser.add_argument('--train_data', help='Training Data.')
    parser.add_argument('--test_data', help='Training Data.')
    parser.add_argument('--val_data_X', help='Validation Data.')
    parser.add_argument('--adv_train_data_X', help='Validation Data.')
    parser.add_argument('--adv_test_data_X', help='Validation Data.')
    parser.add_argument('--num_iters', type=int, default=1000, help='Validation Data.')
    parser.add_argument('--grad_clip', type=float, default=None, help='Gradient Clip Value.')
    parser.add_argument('--num_disc_updates', type=int, default=3, help='Number of time to update discriminator.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--adv', help='Adversarial training?', action='store_true')
    parser.add_argument('--emb_reg', help='Regularize word embeddings?', action='store_true')
    parser.add_argument('--pos_reg', help='Regularize pos embeddings?', action='store_true')
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    print args

    num_epochs = args.num_epochs
    mini_batch_size = 16
    val_mini_batch_size = 256
    t0 = time()
    lr = args.lr
    best_val = 0.

    all_relations = set()
    with open('/home/amri228/adversarial_ppi/data/to_use_test_relations', 'r') as in_file:
        for row in in_file:
            data = row.strip().split('\t')
            es = sorted([data[1], data[2]])
            all_relations.add((data[0], es[0], es[1]))

    ld = LoadData()
    source_train_ids = []
    with open(args.train_data_X,'r') as in_file:
        for row in in_file:
            source_train_ids.append(row.strip())

    source_dev_ids = []
    with open(args.val_data_X,'r') as in_file:
        for row in in_file:
            source_dev_ids.append(row.strip())
    train_pairs, train_e1, train_e2, train_y, train_ids, _, _ = ld.fit_transform(args.train_data,
            source_train_ids)
    print 'NUMBER TRAIN Tuples:', len(train_pairs)

    dev_pairs, dev_e1, dev_e2, dev_y, dev_ids, dev_e1_ids, dev_e2_ids = ld.transform(args.train_data,
            source_dev_ids)
    print 'NUMBER TEST Tuples:', len(dev_pairs)

    adv_train = []
    with open(args.adv_train_data_X,'r') as in_file:
        for row in in_file:
            adv_train.append(row.strip())

    adv_test = []
    test_relations = set()
    with open(args.adv_test_data_X,'r') as in_file:
        for row in in_file:
            adv_test.append(row.strip())
            for x in all_relations:
                if x[0] == row.strip():
                    test_relations.add(x)

    adv_train_pairs, adv_train_e1, adv_train_e2, adv_train_y, adv_train_ids, _, _ = ld.transform(args.test_data, adv_train)
    print 'NUMBER ADV TRAIN Tuples:', len(adv_train_pairs)

    adv_test_pairs, adv_test_e1, adv_test_e2, adv_test_y, adv_test_ids, adv_test_e1_ids, adv_test_e2_ids  = ld.transform(args.test_data, adv_test)
    print 'NUMBER ADV TEST Tuples:', len(adv_test_pairs)


    idxs = list(range(len(train_pairs)))
    dev_idxs = list(range(len(dev_pairs)))

    last_loss = None
    avg_loss = []
    avg_f1 = []
    check_preds = None
    mod = CNN(ld.embs, ld.pos, nc=2, disc_h=args.hidden_state, de=ld.embs.shape[1],
              emb_reg=args.emb_reg, pos_reg=args.pos_reg)
    best_dev_f1 = 0
    # Train Source Model
    best_dev_f1 = 0
    for epoch in range(1, num_epochs+1):
        mean_loss = []
        random.shuffle(idxs)
        for start, end in zip(range(0, len(idxs), mini_batch_size), range(mini_batch_size, len(idxs)+mini_batch_size,
                mini_batch_size)):
            idxs_sample = idxs[start:end]
            batch_labels = np.array(train_y[idxs_sample], dtype='int32')
            tpairs = ld.pad_data([train_pairs[i] for i in idxs_sample])
            te1 = ld.pad_data([train_e1[i] for i in idxs_sample])
            te2 = ld.pad_data([train_e2[i] for i in idxs_sample])
            cost = mod.train_batch_source(tpairs, te1, te2, 
                    train_y[idxs_sample].astype('int32'), 
                    np.float32(0.))
            mean_loss.append(cost)
            print("EPOCH: %d loss: %.4f train_loss: %.4f" % (epoch, cost, np.mean(mean_loss)))
            sys.stdout.flush()

        all_test_preds = []
        for start, end in zip(range(0, len(dev_idxs), val_mini_batch_size), range(val_mini_batch_size, len(dev_idxs)+val_mini_batch_size,
                    val_mini_batch_size)):
            if len(dev_idxs[start:end]) == 0:
                continue
            tpairs = ld.pad_data([dev_pairs[i] for i in dev_idxs[start:end]])
            te1 = ld.pad_data([dev_e1[i] for i in dev_idxs[start:end]])
            te2 = ld.pad_data([dev_e2[i] for i in dev_idxs[start:end]])
            preds = mod.predict_src_proba(tpairs, te1, te2,
                        np.float32(1.))
            all_test_preds += list((preds>0.5).flatten())

        dev_f1 = f1_score(dev_y, all_test_preds, average='binary')
        print("SOURCE_DEV: EPOCH: %d dev_f1: %.4f" % (epoch, dev_f1))
        sys.stdout.flush()

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            with open(args.checkpoint_dir+'/'+args.checkpoint_name+'.pkl','wb') as out_file:
                pickle.dump(mod.__getstate__(), out_file)

    del mod
    mod = CNN(ld.embs, ld.pos, nc=2, disc_h=args.hidden_state, de=ld.embs.shape[1],
              emb_reg=args.emb_reg, pos_reg=args.pos_reg)
    with open(args.checkpoint_dir+'/'+args.checkpoint_name+'.pkl','rb') as in_file:
        weights = pickle.load(in_file)
    mod.__setstate__(weights)
    mod.__settarget__()

    adv_test_idxs = list(range(len(adv_test_pairs)))

    adv_train_idxs = list(range(len(adv_train_pairs)))
    pos_adv_train_idxs = []
    neg_adv_train_idxs = []
    for i in adv_train_idxs:
        if adv_train_y[i] == 1:
            pos_adv_train_idxs.append(i)
        else:
            neg_adv_train_idxs.append(i)
    all_test_preds = []
    all_features_test = []
    all_txt_preds = []
    for start, end in zip(range(0, len(adv_test_idxs), val_mini_batch_size), range(val_mini_batch_size, len(adv_test_idxs)+val_mini_batch_size,
                val_mini_batch_size)):
        if len(adv_test_idxs[start:end]) == 0:
            continue
        tpairs = ld.pad_data([adv_test_pairs[i] for i in adv_test_idxs[start:end]])
        te1 = ld.pad_data([adv_test_e1[i] for i in adv_test_idxs[start:end]])
        te2 = ld.pad_data([adv_test_e2[i] for i in adv_test_idxs[start:end]])
        preds = mod.predict_src_proba(tpairs, te1, te2,
                    np.float32(1.))
        all_test_preds += list((preds>0.5).flatten())

    test_f1 = f1_score(adv_test_y, all_test_preds, average='binary')
    print("START: test_f1: %.4f sum: %d" % (test_f1, int(np.sum(np.array(all_test_preds)))))
    sys.stdout.flush()

    if args.adv:
        num_source_updates = args.num_disc_updates
        # Train Target Model
        best_f1 = 0.
        for update_iter in range(1, args.num_iters+1):
            cost_disc = []
            for k in range(args.num_disc_updates):
                src_sample_idxs = list(np.random.choice(idxs, 128, replace=False))
                src_pairs = ld.pad_data([train_pairs[i] for i in src_sample_idxs])
                src_e1 = ld.pad_data([train_e1[i] for i in src_sample_idxs])
                src_e2 = ld.pad_data([train_e2[i] for i in src_sample_idxs])

                #tgt_sample_idxs = list(np.random.choice(adv_train_idxs, 128, replace=False))
                batch_labels = np.array(train_y[src_sample_idxs], dtype='int32')
                ptgt_sample_idxs = list(np.random.choice(pos_adv_train_idxs, batch_labels.sum(), replace=False))
                ntgt_sample_idxs = list(np.random.choice(neg_adv_train_idxs, 128-batch_labels.sum(), replace=False))
                tgt_sample_idxs = ptgt_sample_idxs + ntgt_sample_idxs
                tgt_pairs = ld.pad_data([adv_train_pairs[i] for i in tgt_sample_idxs])
                tgt_e1 = ld.pad_data([adv_train_e1[i] for i in tgt_sample_idxs])
                tgt_e2 = ld.pad_data([adv_train_e2[i] for i in tgt_sample_idxs])

                cost = mod.train_batch_discriminator(tgt_pairs, src_pairs,
                         tgt_e1, tgt_e2, src_e1, src_e2, np.float32(0.))
                cost_disc.append(cost)

            tgt_sample_idxs = list(np.random.choice(adv_train_idxs, 128, replace=False))
            tgt_pairs = ld.pad_data([adv_train_pairs[i] for i in tgt_sample_idxs])
            tgt_e1 = ld.pad_data([adv_train_e1[i] for i in tgt_sample_idxs])
            tgt_e2 = ld.pad_data([adv_train_e2[i] for i in tgt_sample_idxs])
            cost = mod.train_batch_generator(tgt_pairs, tgt_e1, tgt_e2, np.float32(0.))
            print("ITER: %d discriminator_loss: %.4f generator_loss: %.4f" % (update_iter, np.mean(cost_disc), cost))
            sys.stdout.flush()

        # Predict
        all_test_preds = []
        all_features_test = []
        all_txt_preds = []
        for start, end in zip(range(0, len(adv_test_idxs), val_mini_batch_size), range(val_mini_batch_size, len(adv_test_idxs)+val_mini_batch_size,
                    val_mini_batch_size)):
            if len(adv_test_idxs[start:end]) == 0:
                continue
            tpairs = ld.pad_data([adv_test_pairs[i] for i in adv_test_idxs[start:end]])
            te1 = ld.pad_data([adv_test_e1[i] for i in adv_test_idxs[start:end]])
            te2 = ld.pad_data([adv_test_e2[i] for i in adv_test_idxs[start:end]])
            preds = mod.predict_proba(tpairs, te1, te2,
                        np.float32(1.))
            all_test_preds += list((preds>0.5).flatten())
            for test_id, e1_id, e2_id, p in zip(adv_test_ids[start:end], adv_test_e1_ids[start:end], adv_test_e2_ids[start:end], preds):
                if p > 0.5:
                    all_txt_preds.append((test_id, e1_id, e2_id))

        test_f1 = f1_score(adv_test_y, all_test_preds, average='binary')
        print("ITER: %d test_f1: %.4f sum: %d" % (update_iter, test_f1, int(np.sum(np.array(all_test_preds)))))
        sys.stdout.flush()
        #adv_test_ids, adv_test_e1_ids, adv_test_e2_ids
        best_f1 = test_f1
        #all_txt_preds = []
        all_features_val = []
        for start, end in zip(range(0, len(dev_idxs), val_mini_batch_size), range(val_mini_batch_size, len(dev_idxs)+val_mini_batch_size,
                    val_mini_batch_size)):
            if len(dev_idxs[start:end]) == 0:
                continue
            tpairs = ld.pad_data([dev_pairs[i] for i in dev_idxs[start:end]])
            te1 = ld.pad_data([dev_e1[i] for i in dev_idxs[start:end]])
            te2 = ld.pad_data([dev_e2[i] for i in dev_idxs[start:end]])
            preds = mod.predict_proba(tpairs, te1, te2,
                        np.float32(1.))

        with open(args.checkpoint_dir+'/'+'predictions.txt','w') as out_file:
            for i in all_txt_preds:
                out_file.write('%s\t%s\t%s\n' % (i[0], i[1], i[2]))

main()
