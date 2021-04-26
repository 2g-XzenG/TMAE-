import numpy as np
import _pickle as pickle
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from gensim.models import Word2Vec
from TMAE import *
from DataGenerator import *

print("------LOADING DATA------")
age_seq = pickle.load(open("../../DATA/age_seq","rb"))
sex_seq = pickle.load(open("../../DATA/sex_seq","rb"))

code_seq_2013 = pickle.load(open("../../DATA/code_seq_2013","rb"))
date_seq_2013 = pickle.load(open("../../DATA/date_seq_2013","rb"))
util_seq_2013 = pickle.load(open("../../DATA/util_seq_2013","rb"))
cost_seq_2013 = pickle.load(open("../../DATA/cost_seq_2013","rb"))

proc2cat = pickle.load(open("../../DATA/CCS/proc2cat","rb"))
diag2cat = pickle.load(open("../../DATA/CCS/diag2cat","rb"))
drug2cat = pickle.load(open("../../DATA/CCS/drug2cat","rb"))

MAX_VISIT=30
MAX_CODE=10
MAX_DEMO=2
BATCH_SIZE = 100
TRAIN_RATIO = 0.8
DATA_SIZE = len(age_seq)
EPOCHS = 1000

ccs_seq = get_ccs(code_seq_2013, diag2cat, proc2cat, drug2cat)
ccs_feature, ccs2int = process_code(ccs_seq)  
code_feature, code2int = process_code(code_seq_2013)
util_feature, util2int = process_util(util_seq_2013)
demo_feature, demo2int = process_demo(age_seq, sex_seq)
date_feature = date_seq_2013
cost_feature, cost2int = process_cost(cost_seq_2013)
cost_label = cost_seq_2013

code_embedding = get_embedding(code2int, Word2Vec.load("./pretrain/model_code"))
ccs_embedding = get_embedding(ccs2int, Word2Vec.load("./pretrain/model_ccs"))

params = {
    'seqs':[demo_feature, code_feature, ccs_feature, util_feature, date_feature, cost_feature, cost_label],
    'vocab_sizes': [len(code2int)],
    'batch_size':100,
    'max_visit':MAX_VISIT, 
    'max_code':MAX_CODE,
}

from sklearn.model_selection import train_test_split
train_IDs, valid_IDs = train_test_split(range(DATA_SIZE), train_size=TRAIN_RATIO, random_state=42)
train_generator = DataGenerator(list_IDs=train_IDs, shuffle=True, **params)
valid_generator = DataGenerator(list_IDs=valid_IDs, shuffle=False, **params)

model_losses = {
    "cost_label": cost_loss_fun,
    "code_label": code_loss_fun,
}

model_metrics = {
    "code_label": recall,
}

model_weights = {
    "cost_label": 2e-6,
    "code_label": 1,
}

for dim in [100,200,300,400,500]:
    print("*"*10, dim, "*"*10)
    m = model(
        patient_dim=dim,
        max_visit=MAX_VISIT,
        max_code=MAX_CODE,
        max_demo=MAX_DEMO,
        code_vocab=len(code2int),
        ccs_vocab=len(ccs2int),
        demo_vocab=len(demo2int),
        util_vocab=4,
        cost_vocab=51,
        date_vocab=365,
        code_embed_matrix=code_embedding,
        ccs_embed_matrix=ccs_embedding,
        )
    m.compile(optimizer='RMSprop', loss=model_losses, metrics=model_metrics, loss_weights=model_weights)
    print(m.summary())

    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min', restore_best_weights=True)
    m.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=valid_generator,
        verbose=1,
        callbacks=[earlyStopping],
    )

    data_generator = DataGenerator(list_IDs=range(DATA_SIZE), shuffle=False, **params)
    layer_name = 'patient_embedding'
    intermediate_layer_model = tf.keras.models.Model(inputs=m.input, outputs=m.get_layer(layer_name).output)
    encoded_features = intermediate_layer_model.predict(data_generator)
    pickle.dump(encoded_features, open("./TMAE_Embedding_"+str(dim),"wb"))





