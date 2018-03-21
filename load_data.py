import random
import pickle
from time import time
import sys
from collections import defaultdict
import gensim

import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split

def dataRead(fname, filter_ids=None):
    print "Input File Reading"
    fp = open(fname, 'r')
    #samples = fp.read().strip().split('\r\n\r\n')
    samples = fp.read().strip().split('\n\n')
    sent_lengths   = []        #1-d array
    sent_contents  = []        #2-d array [[w1,w2,....] ...]
    sent_lables    = []        #1-d array
    entity1_list   = []        #2-d array [[e1,e1_t] [e1,e1_t]...]
    entity2_list   = []        #2-d array [[e1,e1_t] [e1,e1_t]...]
    doc_ids = []
    idents = []
    for sample in samples:
        #sent, entities = sample.strip().split('\r\n')
        sent, entities = sample.strip().split('\n')
        doc_id, ident, e1, e2, relation = entities.split('\t') 
        if filter_ids is not None:
            if ident not in filter_ids:
                continue
        sent_contents.append(sent.lower())
        entity1_list.append([e1.lower(), ident])
        entity2_list.append([e2.lower(), ident])
        sent_lables.append(relation)
        idents.append(ident)
        doc_ids.append(doc_id)

    return idents, sent_contents, entity1_list, entity2_list, sent_lables 

class LoadData(object):
    def __init__(self):
        self.word_index = {}
        self.pos_index = {}
        self.num_words = 1
        self.num_pos = 1
        self.embs = [np.zeros((300,))]
        self.pos  = [np.zeros((32,))]
        self.wv = gensim.models.Word2Vec.load('/home/amri228/i2b2_2016/ddi/word_vecs2/gensim_model_pubmed')
        self.max_u = self.wv.syn0.max()
        self.min_u = self.wv.syn0.min()

    def fit(self, filename, skip):
        all_data = dataRead(filename, skip)
        word_cnts = {}
        pos_cnts = {}
        for ident, tr, tl, e1, e2 in zip(all_data[0], all_data[1],
            all_data[-1], all_data[2], all_data[3]):
            final_string = tr.split(' ')
            e1_pos = None
            e2_pos = None
            cnt = 0
            for w in final_string:
                if w == 'proteina':
                    e1_pos = cnt
                elif w == 'proteinb':
                    e2_pos = cnt
                cnt += 1
            is_flip = False
            if e1_pos > e2_pos:
                is_flip = True 
                tmp_pos = e1_pos
                e1_pos = e2_pos
                e2_pos = tmp_pos
            if e1_pos is None or e2_pos is None:
                continue
            tmp = []
            final_e1_pos = []
            final_e2_pos = []
            cnt = 0
            error = False
            for w in final_string:
                if cnt-e1_pos in pos_cnts:
                    pos_cnts[cnt-e1_pos] += 1
                else:
                    pos_cnts[cnt-e1_pos] = 1
                if cnt-e2_pos in pos_cnts:
                    pos_cnts[cnt-e2_pos] += 1
                else:
                    pos_cnts[cnt-e2_pos] = 1
                cnt += 1
            for w in final_string:
                if w in word_cnts:
                    word_cnts[w] += 1
                else:
                    word_cnts[w] = 1
            for w in final_e1_pos:
                if w in pos_cnts:
                    pos_cnts[w] += 1
                else:
                    pos_cnts[w] = 1
            for w in final_e2_pos:
                if w in pos_cnts:
                    pos_cnts[w] += 1
                else:
                    pos_cnts[w] = 1

        for w, cnt in word_cnts.iteritems():
            if cnt > 5:
                if w in self.wv:
                    self.embs.append(self.wv[w])
                    self.word_index[w] = self.num_words
                    self.num_words += 1
                else:
                    #self.embs.append(np.random.uniform(self.min_u, self.max_u, (300,)))
                    self.embs.append(np.random.uniform(-1., 1., (300,)))
                    self.word_index[w] = self.num_words
                    self.num_words += 1
        for w, cnt in pos_cnts.iteritems():
            if cnt > 5:
                self.pos.append(np.random.uniform(-1., 1., (32,)))
                self.pos_index[w] = self.num_pos
                self.num_pos += 1

        self.pos_index['NegUNK'] = self.num_pos
        self.num_pos += 1
        self.pos.append(np.random.uniform(-1., 1., (32,)))
        self.pos_index['PosUNK'] = self.num_pos
        self.num_pos += 1
        self.pos.append(np.random.uniform(-1., 1., (32,)))

        self.word_index['UNK'] = self.num_words
        #self.embs.append(np.random.uniform(self.min_u, self.max_u, (300,)))
        self.embs.append(np.random.uniform(-1., 1., (300,)))
        self.num_words += 1

        #del self.wv
        self.embs = np.array(self.embs, dtype='float32')
        self.pos = np.array(self.pos, dtype='float32')
        return

    def fit_iter(self, filename, skip):
        all_data = dataRead(filename, skip)
        word_cnts = {}
        pos_cnts = {}
        for ident, tr, tl, e1, e2 in zip(all_data[0], all_data[1],
            all_data[-1], all_data[2], all_data[3]):
            final_string = tr.split(' ')
            e1_pos = None
            e2_pos = None
            cnt = 0
            for w in final_string:
                if w == 'proteina':
                    e1_pos = cnt
                elif w == 'proteinb':
                    e2_pos = cnt
                cnt += 1
            is_flip = False
            if e1_pos > e2_pos:
                is_flip = True 
                tmp_pos = e1_pos
                e1_pos = e2_pos
                e2_pos = tmp_pos
            tmp = []
            final_e1_pos = []
            final_e2_pos = []
            cnt = 0
            error = False
            for w in final_string:
                if cnt-e1_pos in pos_cnts:
                    pos_cnts[cnt-e1_pos] += 1
                else:
                    pos_cnts[cnt-e1_pos] = 1
                if cnt-e2_pos in pos_cnts:
                    pos_cnts[cnt-e2_pos] += 1
                else:
                    pos_cnts[cnt-e2_pos] = 1
                cnt += 1
            for w in final_string:
                if w in word_cnts:
                    word_cnts[w] += 1
                else:
                    word_cnts[w] = 1
            for w in final_e1_pos:
                if w in pos_cnts:
                    pos_cnts[w] += 1
                else:
                    pos_cnts[w] = 1
            for w in final_e2_pos:
                if w in pos_cnts:
                    pos_cnts[w] += 1
                else:
                    pos_cnts[w] = 1

        for w, cnt in word_cnts.iteritems():
            if cnt > 5 and w not in self.word_index:
                if w in self.wv:
                    self.embs = np.vstack([self.embs, self.wv[w]])
                else:
                    self.embs = np.vstack([self.embs, np.random.uniform(self.min_u, self.max_u, (300,))])
                self.word_index[w] = self.num_words
                self.num_words += 1
        for w, cnt in pos_cnts.iteritems():
            if cnt > 5 and w not in self.pos_index:
                self.pos = np.vstack([self.pos, np.random.uniform(-1., 1., (32,))])
                self.pos_index[w] = self.num_pos
                self.num_pos += 1
        return

    def transform(self, filename, skip):
        all_data = dataRead(filename, skip)
        pairs_idx = []
        pairs_idx_rev = []
        domain_labels = []
        pos_e2_idx = []
        pos_e1_idx = []
        e1_ids = []
        e2_ids = []
        y = []
        idents = []
        for ident, tr, tl, e1, e2 in zip(all_data[0], all_data[1],
            all_data[-1], all_data[2], all_data[3]):
            final_string = tr.split(' ')
            e1_pos = None
            e2_pos = None
            cnt = 0
            for w in final_string:
                if w == 'proteina':
                    e1_pos = cnt
                elif w == 'proteinb':
                    e2_pos = cnt
                cnt += 1
            is_flip = False
            if e1_pos is None or e2_pos is None:
                continue
            if e1_pos > e2_pos:
                is_flip = True 
                tmp_pos = e1_pos
                e1_pos = e2_pos
                e2_pos = tmp_pos
            tmp = []
            final_e1_pos = []
            final_e2_pos = []
            cnt = 0
            for w in final_string:
                final_e1_pos.append(cnt - e1_pos)
                final_e2_pos.append(cnt - e2_pos)
                cnt += 1
            idents.append(ident)
            y.append(tl)
            e1_ids.append(e1[0])
            e2_ids.append(e2[0])
            fstring = []
            for w in final_string:
                if w == 'proteina' and is_flip:
                    fstring.append('proteinb')
                elif w == 'proteinb' and is_flip:
                    fstring.append('proteina')
                else:
                    fstring.append(w)
            final_string = fstring
            str_idx = []
            for w in final_string:
                if w in self.word_index:
                    str_idx.append(self.word_index[w])
                else:
                    str_idx.append(self.word_index['UNK'])
            pairs_idx.append(str_idx)
            e1_idx = []
            for p in final_e1_pos:
                if p in self.pos_index:
                    e1_idx.append(self.pos_index[p])
                else:
                    if p < 0:
                        e1_idx.append(self.pos_index['NegUNK'])
                    else:
                        e1_idx.append(self.pos_index['PosUNK'])
            pos_e1_idx.append(e1_idx)
            e2_idx = []
            for p in final_e2_pos:
                if p in self.pos_index:
                    e2_idx.append(self.pos_index[p])
                else:
                    if p < 0:
                        e2_idx.append(self.pos_index['NegUNK'])
                    else:
                        e2_idx.append(self.pos_index['PosUNK'])
            pos_e2_idx.append(e2_idx)

        lab_lookup = {'True':1, 'False':0}
        lab_lookup_rev = {1:'True', 0:'False'}
        final_y = np.array([np.int32(lab_lookup[x]) for x in y])

        return pairs_idx, pos_e1_idx, pos_e2_idx, final_y, idents, e1_ids, e2_ids

    def fit_transform(self, filename, skip):
        self.fit(filename, skip)
        return self.transform(filename, skip)

    def fit_iter_transform(self, filename, skip):
        self.fit_iter(filename, skip)
        return self.transform(filename, skip)

    def pad_data(self, data):
        max_len = np.max([len(x) for x in data])
        padded_dataset = []
        for example in data:
            zeros = [0]*(max_len-len(example))
            padded_dataset.append(example+zeros)
        return np.array(padded_dataset)


