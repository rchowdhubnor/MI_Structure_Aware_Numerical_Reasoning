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
np.random.seed(0)


model=LanguageModel('meta-llama/Meta-Llama-3.1-8B',device_map="auto")
tok_id=np.zeros((1000))
tok_id_num=tok_id.astype(int)
for i in range(1000):
    tok_id_num[i]=model.tokenizer.encode(str(i))[1]
num=np.arange(0, 1000)
DataT=np.load('')
# indices = np.random.choice(DataT.shape[0], size=100, replace=False)
# Data=DataT[indices,:]
Data=DataT[0,:,:].astype(int)
print(Data.shape,np.min(Data),np.max(Data))

import time
start_time = time.time()
num_add=np.ones((1000))
r2_data=np.zeros((Data.shape[0],2))
for tid,toknext in enumerate(list(range(7, 14, 2))):
    if tid == 3:
        for n in range(Data.shape[0]):
            start_time = time.time()
            wave = ",".join(map(str, Data[n,0:(16+toknext)])) +","
            layers = model.model.layers
            contextlen=len(model.tokenizer.encode(wave))
            with model.trace() as tracer:
                with tracer.invoke(wave) as invoker:
                    output = model.output.save()
                    layer_output = model.lm_head(model.model.norm(model.model.layers[-1].output[0]))
                    probs = torch.nn.functional.softmax(layer_output, dim=-1).save()

            num_add=np.ones((1000))
            new_prob=np.dot(probs[0,-1,tok_id_num].detach().cpu().numpy(),num_add)
            new_probdist=probs[0,-1,tok_id_num].detach().cpu().numpy()/new_prob
            pred_mean=np.dot(new_probdist,num)
            max_pred=int(num[np.argmax(new_probdist)])

            # print(n,Data[n,(16+toknext)],max_pred)
            r2_data[n,0]=Data[n,(16+toknext)]
            r2_data[n,1]=max_pred
            
            del output,probs
            torch.cuda.empty_cache()
            gc.collect()

            print(n,"--- %s seconds ---" % (time.time() - start_time))

