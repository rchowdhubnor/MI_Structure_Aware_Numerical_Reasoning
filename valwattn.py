from nnsight import CONFIG
from nnsight import LanguageModel
import numpy as np
import torch
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
import time
import json
import csv
import os
import time
import json
import csv
import pickle
import os
import random 
from nnsight import CONFIG
from collections import Counter
import matplotlib.pyplot as plt
import nnsight
import gc
import copy
import torch
import torch.nn.functional as F

np.random.seed(0)

model=LanguageModel('meta-llama/Meta-Llama-3.1-8B',device_map="auto")
tok_id=np.zeros((1000))
tok_id_num=tok_id.astype(int)
for i in range(1000):
    tok_id_num[i]=model.tokenizer.encode(str(i))[1]
num=np.arange(0, 1000)
datasame=np.load('')
# print("datasame shape:", datasame.shape, np.min(datasame), np.max(datasame))
errsame=np.load('')
indices = [[] for i in range(4)]
# print(len(indices))
# print(errsame.shape)
errsame=errsame.astype(int)
for n in range(errsame.shape[0]):
    for i in range(errsame.shape[1]):
        if errsame[n,i] == 0:
            indices[i].append(n)
# for i in range(len(indices)):
#     print("indices",i,"length:",indices[i])
Number_Test=100
DataT=datasame.astype(int)
# print(DataT.shape,np.min(DataT),np.max(DataT))
import time

num_add=np.ones((1000))
layers = model.model.layers
# for tid,toknext in enumerate(list(range(7, 14, 2))):
tkindlst=list(range(7, 14, 2))
# print(tkindlst[0:3])
for l_sel in range(25,32):
    for tid,toknext in enumerate(tkindlst):
        # print(tid)
        len_context=len(model.tokenizer.encode(",".join(map(str, DataT[0,0,0:(16+toknext)])) +","))
        # print("len_context:", len_context)    
        vwattmap=np.zeros(( len(indices[tid][0:100]),32,len_context))
        
        for ind,n in enumerate(indices[tid][0:100]):
            start_time = time.time()
            if tid<3:
                continue
            else:
                wave = ",".join(map(str, DataT[0,n,0:(16+toknext)])) +","

                with model.trace() as tracer:
                    with tracer.invoke(wave) as invoker:
                        q = model.model.layers[l_sel].self_attn.q_proj.output[0,:,:].save()
                        k = model.model.layers[l_sel].self_attn.k_proj.output[0,:,:].save()
                        v_proj = model.model.layers[l_sel].self_attn.v_proj.output.save()
                        output = model.output.save()
                        layer_output = model.lm_head(model.model.norm(model.model.layers[-1].output[0]))
            
                        probs = torch.nn.functional.softmax(layer_output, dim=-1).save()
                num_add=np.ones((1000))
                new_prob=np.dot(probs[0,-1,tok_id_num].detach().cpu().numpy(),num_add)
                new_probdist=probs[0,-1,tok_id_num].detach().cpu().numpy()/new_prob
                pred_mean=np.dot(new_probdist,num)
                max_pred=int(num[np.argmax(new_probdist)])
                input_ids = model.tokenizer(wave, return_tensors='pt').input_ids[0]
                decoded_chars = [model.tokenizer.decode([tid]) for tid in input_ids]
                # Hyperparameters
                num_query_heads = 32
                num_key_heads = 8
                head_dim = 128
                group_size = num_query_heads // num_key_heads
                # --- Step 2: Reshape ---
                # Queries: [seq_len, 4096] -> [seq_len, 32, 128]
                q_reshaped = q.view(-1, num_query_heads, head_dim)
                # Keys: [seq_len, 1024] -> [seq_len, 8, 128]
                k_reshaped = k.view(-1, num_key_heads, head_dim)
                # Values
                v_reshaped = v_proj.view(-1, num_key_heads, head_dim)

                # --- Step 3: Select Head 1 ---
                # head_idx = 0
                for head_idx in range(32):
                    group_id = head_idx // group_size  # Which key group this head uses (1//4=0)

                    # Get all queries for this key group (heads 0-3 use key group 0)
                    q_group = q_reshaped[:, group_id*group_size:(group_id+1)*group_size]  # [seq_len, 4, 128]
                    k_group = k_reshaped[:, group_id]  # [no. of tokens, 128]
                    v_group = v_reshaped[:, group_id]  # [no. of tokens, 128]
                    # --- Step 4: Compute Attention for Head 1 ---
                    q_head = q_reshaped[:, head_idx]  # [no. of tokens, 128]#
                    # print(q_head.shape,k_group.transpose(-2, -1).shape)
                    d_k = head_dim

                    # Broadcast matmul: [seq_len, 128] @ [128, seq_len] -> [seq_len, seq_len]
                    scores = torch.matmul(q_head, k_group.transpose(-2, -1)) / (d_k ** 0.5)
                    attention_map = F.softmax(scores, dim=-1)
                    seq_len = q_head.shape[0]
                    positions = np.arange(seq_len)
                    v_group = v_group.detach().cpu().numpy()
                    normv_group = np.linalg.norm(v_group, axis=1)
                    # --- Step 5: Plot ---
                    atmp=attention_map.detach().cpu().numpy()
                    attmap=atmp[-1,:]
                    vwattmap[ind,head_idx,:]=attmap*normv_group
            torch.cuda.empty_cache()
            gc.collect()
