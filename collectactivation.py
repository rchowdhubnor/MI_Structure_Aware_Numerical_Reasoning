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
np.random.seed(42)

model=LanguageModel('meta-llama/Meta-Llama-3.1-8B',device_map="auto")
tok_id=np.zeros((1000))
tok_id_num=tok_id.astype(int)
for i in range(1000):
    tok_id_num[i]=model.tokenizer.encode(str(i))[1]
num=np.arange(0, 1000)
datasame=np.load()
print("datasame shape:", datasame.shape, np.min(datasame), np.max(datasame))
errsame=np.load('')
indices = [[] for i in range(4)]
print(len(indices))
print(errsame.shape)
errsame=errsame.astype(int)
for n in range(errsame.shape[0]):
    for i in range(errsame.shape[1]):
        if errsame[n,i] == 0:
            indices[i].append(n)

Number_Test=100
DataT=datasame.astype(int)
print(DataT.shape,np.min(DataT),np.max(DataT))
import time

num_add=np.ones((1000))
layers = model.model.layers
tkindlst=list(range(7, 14, 2))
# print(tkindlst[0:3])
# li=5
for tid,toknext in enumerate(tkindlst[3:4]):
    # print(tid)
    len_context=len(model.tokenizer.encode(",".join(map(str, DataT[0,0,0:(16+toknext)])) +","))
    # print("len_context:", len_context)    
    activation=np.zeros((5000,3,32,4096))
    # for ind,n in enumerate(list(range(5000))):
    print("5000 samples")
    for ind,n in enumerate(list(range(5000))):
        
        
        start_time = time.time()

        wave = ",".join(map(str, DataT[0,n,0:(16+toknext)])) +","
        patchtoklen=len(model.tokenizer.encode(wave))
        # print(ind)

        max_start = patchtoklen 
        
        # for pattok in range(1, max_start):
        indices=[25,27,-2]
        for li in range(32):
            with model.trace() as tracer:
                with tracer.invoke(wave) as invoker:
                    collect_activation=model.model.layers[li].output[0][0, indices,:].save()
            activation[ind,:,li,:]=collect_activation.detach().cpu().numpy()
            # print(collect_activation.shape)
            torch.cuda.empty_cache()
            gc.collect()
        print(ind,"--- %s seconds ---" % (time.time() - start_time))    
