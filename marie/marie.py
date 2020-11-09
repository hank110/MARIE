import json

import numpy as np
from numpy import inner, sqrt, dot
import textdistance
import editdistance

def _create_txt2idx(file_path):
    idx2txt=dict()
    idx=0
    
    with open(file_path, 'r') as f:
        for line in f:
            idx2txt[idx]=line.rstrip()
            idx+=1
    txt2idx={v:k for k, v in idx2txt.items()}
    
    return txt2idx, idx2txt

def _cal_cosine(vec1, vec2):
    return inner(vec1, vec2)/sqrt(np.dot(vec1, vec1)*dot(vec2, vec2))

def _cal_phrase_vector(json_entry, layers=1):
    phrase_vector=np.zeros(len(json_entry['features'][0]['layers'][0]['values']))
    
    for i in range (1, len(json_entry['features'])-1):
        word_vector=np.zeros(len(json_entry['features'][0]['layers'][0]['values']))
        for j in range(layers):
            word_vector+=np.array(json_entry['features'][i]['layers'][j]['values'])
        
        phrase_vector+=np.divide(word_vector, layers)
        
    return np.divide(phrase_vector, len(json_entry['features']) - 2)

def _get_bert_vectors(weight_path, label_path, txt2idx, layers=1):
    embedding_dict=dict()

    with open(weight_path, 'r') as f1, open(label_path, 'r') as f2:
        for line1,line2 in zip(f1, f2):
            embedding_dict[txt2idx[line2.rstrip()]]=_cal_phrase_vector(json.loads(line1), layers)
    
    return embedding_dict

def map_marie(input_data, target_data, input_bert_weights, target_bert_weights, string_match='edit', alpha=0.8, bert_layers=1, top_n=5):
    inp_txt2idx, inp_idx2txt=_create_txt2idx(input_data)
    tgt_txt2idx, tgt_idx2txt=_create_txt2idx(target_data)

    inp_bert_vectors=_get_bert_vectors(input_bert_weights, input_data, inp_txt2idx, bert_layers)
    tgt_bert_vectors=_get_bert_vectors(target_bert_weights, target_data, tgt_txt2idx, bert_layers)

    mapper=dict()

    for cnt, (inp_txt, inp_idx) in enumerate(inp_txt2idx.items()):
        inp_bert_vector=inp_bert_vectors[inp_idx]
        cal_dist=[]

        cos_dist=[alpha*(_cal_cosine(inp_bert_vector, tgt_bert_vectors[tgt_idx])) for tgt_txt, tgt_idx in tgt_txt2idx.items()]

        if string_match=='edit':
            str_match=[(1-alpha)*(1-editdistance.eval(inp_txt, tgt_txt)/max(len(inp_txt), len(tgt_txt))) for tgt_txt, tgt_idx in tgt_txt2idx.items()]
        if string_match=='jaccard':
            str_match=[(1-alpha)*textdistance.jaccard(inp_txt, tgt_txt) for tgt_txt, tgt_idx in tgt_txt2idx.items()]
        if string_match=='ob':
            str_match=[(1-alpha)*textdistance.ratcliff_obershelp(inp_txt, tgt_txt) for tgt_txt, tgt_idx in tgt_txt2idx.items()]

        ord2idx=[tgt_idx for _, tgt_idx in tgt_txt2idx.items()]

        cal_dist=np.add(cos_dist, str_match)
        topn_ord_idx=cal_dist.argsort()[::-1][:top_n]

        mapper[inp_idx]=[(ord2idx[idx], cal_dist[idx]) for idx in topn_ord_idx]

        if cnt%100==0:
            print("...Processed %i mappings" %(cnt))

    return mapper, inp_idx2txt, tgt_idx2txt