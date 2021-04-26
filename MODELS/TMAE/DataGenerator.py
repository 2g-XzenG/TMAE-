import tensorflow as tf
import pandas as pd
import numpy as np

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, seqs, vocab_sizes, list_IDs, max_visit, max_code, batch_size=100, shuffle=True):
        self.seqs = seqs
        self.code_vocab = vocab_sizes[0]
        self.list_IDs = list_IDs
        self.max_visit = max_visit
        self.max_code = max_code
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch' 
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        demo_feature, code_feature, ccs_feature, util_feature, date_feature, cost_feature, cost_label = self.seqs
        batch_demo, batch_code, batch_ccs, batch_util, batch_date, batch_cost, batch_cost_label = [], [], [], [], [], [], []
        for i, ID in enumerate(list_IDs_temp):
            batch_demo.append(demo_feature[ID])
            batch_code.append(code_feature[ID])
            batch_ccs.append(ccs_feature[ID])
            batch_util.append(util_feature[ID])
            batch_date.append(date_feature[ID])
            batch_cost.append(cost_feature[ID])
            batch_cost_label.append(cost_label[ID])
        
        batch_demo_feature = np.array(batch_demo)
        batch_code_feature = self.code_padding(batch_code)
        batch_ccs_feature = self.code_padding(batch_ccs)
        batch_util_feature = self.date_padding(batch_util)
        batch_date_feature = self.date_padding(batch_date)
        batch_cost_feature = self.date_padding(batch_cost)
        
        batch_code_label = self.code_labelling(batch_code)
        batch_cost_label = self.cost_labelling(batch_cost_label)
        dic = (
            {
                'demo_feature': batch_demo_feature,
                'code_feature': batch_code_feature,
                'ccs_feature':  batch_ccs_feature,
                'util_feature': batch_util_feature,
                'date_feature': batch_date_feature,
                'cost_feature': batch_cost_feature,
            },
            {
                'code_label': batch_code_label,
                'cost_label': batch_cost_label,
            })
        return dic
    
    def date_padding(self, seq):
        pad_seq = np.zeros((len(seq), self.max_visit))
        for i, p in enumerate(seq):
            pad_seq[i][:len(p)] = p[:self.max_visit]
        return pad_seq
    
    def code_padding(self, seq):
        X = np.zeros((len(seq), self.max_visit, self.max_code))
        for i, p in enumerate(seq):
            if len(p) > self.max_visit: 
                p = p[:self.max_visit]
            for j, claim in enumerate(p):
                claim = claim[:self.max_code]
                X[i][j][:len(claim)] = claim
        return X
    
    def code_labelling(self, seq):
        X = np.zeros((len(seq), self.max_visit, self.code_vocab))
        for i, p in enumerate(seq):
            if len(p) > self.max_visit: 
                p = p[:self.max_visit]
            for j, claim in enumerate(p):
                for c in claim:
                    X[i][j][c] = 1
        return X

    def cost_labelling(self, seq):
        pad_seq = np.zeros((len(seq), self.max_visit))
        for i, p in enumerate(seq):
            pad_seq[i][:len(p)] = p[:self.max_visit]
        return pad_seq

def get_ccs(seq, diag2cat, proc2cat, drug2cat):
    new_seq = []
    for p in seq:
        new_p = []
        for v in p:
            new_v = []
            for c in v:
                if c in diag2cat :
                    new_c = diag2cat[c]
                elif c in proc2cat:
                    new_c = proc2cat[c]
                elif c in drug2cat:
                    new_c = drug2cat[c]
                else:
                    new_c = c[:5]
                new_v.append(new_c)
            new_p.append(new_v)
        new_seq.append(new_p)
    return new_seq

def process_code(seq, PAD=True):
    new_seq = []
    if PAD: vocab2int = {"PAD":0}
    else: vocab2int = {}
    for p in seq:
        new_p = []
        for v in p:
            new_v = []
            for c in v:
                if c not in vocab2int: vocab2int[c] = len(vocab2int)
                new_v.append(vocab2int[c])
            new_p.append(new_v)
        new_seq.append(new_p)
    return new_seq, vocab2int

def process_util(seq):
    new_seq = []
    vocab2int = {"PAD":0,"IP":1,"RX":2,"OP":3}
    for p in seq:
        new_p = []
        for v in p:
            if "IP" in v:
                new_v=1
            elif "RX" in v:
                new_v=2
            else:
                new_v=3
            new_p.append(new_v)
        new_seq.append(new_p)
    return new_seq, vocab2int
    
def process_demo(age_seq, sex_seq):
    new_seq = []
    vocab2int = {}
    for age, sex in zip(age_seq, sex_seq):
        p = []
        if age not in vocab2int: vocab2int[age] = len(vocab2int)
        if sex not in vocab2int: vocab2int[sex] = len(vocab2int)
        p.append(vocab2int[age])
        p.append(vocab2int[sex])
        new_seq.append(p)
    return np.array(new_seq), vocab2int

def cost_helper(costs, split_num):
    'discretize cost values'
    v_costs = sorted([c for p in costs for c in p])
    stop_v = sum(v_costs)/split_num
    summ=0
    boundary = [-100, 0]   # 0 cost -> "1"
    for c in v_costs:
        summ+=c
        if summ>=stop_v:
            boundary.append(c)
            summ =0
    boundary += [10000000] # max is 10 million
    return boundary

def process_cost(seq, split_num=50):
    boundary = cost_helper(seq, split_num)
    bucket_costs = []
    for p in seq:
        b = pd.cut(p, bins=boundary, labels=range(len(boundary) - 1), right=False).astype(float)
        bucket_costs.append(b)
    return bucket_costs, boundary

def get_embedding(code2int, model, embed_size=100):
    embedding = np.zeros((len(code2int), embed_size))
    for code in code2int:
        if code=="PAD":continue
        embedding[code2int[code]] = model.wv[code]
    return embedding

