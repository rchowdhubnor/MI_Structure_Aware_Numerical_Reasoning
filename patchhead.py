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
import einops
np.random.seed(42)

model=LanguageModel('meta-llama/Meta-Llama-3.1-8B',device_map="auto")
tok_id=np.zeros((1000))
tok_id_num=tok_id.astype(int)
for i in range(1000):
    tok_id_num[i]=model.tokenizer.encode(str(i))[1]
num=np.arange(0, 1000)
datasame=np.load('')
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
print("updated: 100 samples")
toknext=tkindlst[-1]
tid=3
len_context=len(model.tokenizer.encode(",".join(map(str, DataT[0,0,0:(16+toknext)])) +","))
# print("len_context:", len_context)    
for ind,n in enumerate(indices[tid][0:100]):
    start_time = time.time()
    # total=len_context-1-6
    maegrad=np.zeros((32,32))
    accgrad=np.zeros((32,32))
    logit_gaingrad=np.zeros((32,32))
    mae_netoutput=np.zeros((32,32))
    logit_lossnetoutput=np.zeros((32,32))
    accnet=np.zeros((32,32))
    NumberIndices=np.zeros((16+toknext))
    DiffIndices=np.zeros((16+toknext-1))
    prob_O = np.zeros((1000))
    prob_I_B = np.zeros((1000))
    prob_I_A = np.zeros((32,32,1000))
    start_time = time.time()
    InterveneData=np.ones((len(DataT[0,n,0:(16+toknext)])))
    delta=DataT[0,n,(16+toknext)]-DataT[0,n,(16+toknext-1)]
    # if delta>=0:
    #     InterveneData=InterveneData*99
    # else:
    InterveneData=InterveneData*100
    InterveneData=InterveneData.astype(int)
    
    # print("data",DataT[0,n,0:(16+toknext+1)])
    # print("intervene data",InterveneData[0:(16+toknext+1)])
    wave = ",".join(map(str, DataT[0,n,0:(16+toknext)])) +","
    waveI = ",".join(map(str, InterveneData[0:(16+toknext)])) +","
    NumberIndices=DataT[0,n,0:(16+toknext)]
    DiffIndices=np.diff(DataT[0,n,0:(16+toknext+1)])
    # print("wave:", wave)
    # print("waveI:", waveI)
    # for t,tokcheck in enumerate(model.tokenizer.encode(wave)):
    #     print(t,model.tokenizer.decode(tokcheck))
    with model.trace() as tracer:
        with tracer.invoke(wave) as invoker:
            output = model.output.save()
            layer_output = model.lm_head(model.model.norm(model.model.layers[-1].output[0]))
            probs = torch.nn.functional.softmax(layer_output, dim=-1).save()
    
    num_add=np.ones((1000))
    new_prob=np.dot(probs[0,-1,tok_id_num].detach().cpu().numpy(),num_add)
    prob_O=probs[0,-1,tok_id_num].detach().cpu().numpy()
    new_probdist=probs[0,-1,tok_id_num].detach().cpu().numpy()/new_prob
    pred_mean=np.dot(new_probdist,num)
    max_pred=int(num[np.argmax(new_probdist)])
    with model.trace() as tracer:
        with tracer.invoke(waveI) as invoker:
            outputnoint = model.output.save()
            layer_output = model.lm_head(model.model.norm(model.model.layers[-1].output[0]))
            probsnit = torch.nn.functional.softmax(layer_output, dim=-1).save()
    num_add=np.ones((1000))
    new_probnoint=np.dot(probsnit[0,-1,tok_id_num].detach().cpu().numpy(),num_add)
    prob_I_B=probsnit[0,-1,tok_id_num].detach().cpu().numpy()
    new_probnoint=probsnit[0,-1,tok_id_num].detach().cpu().numpy()/new_probnoint
    max_prednoint=int(num[np.argmax(new_probnoint)])
    torch.cuda.empty_cache()
    gc.collect()
    patchtoklen=len(model.tokenizer.encode(wave))
    for l in range(len(layers)):
        for head_idx in range(32):
            with model.trace() as tracer:
                with tracer.invoke(wave) as invoker:
                    # if l>0:
                    #     prevhidden=model.model.layers[l-1].output[0][0,-1,:].save()
                    att=model.model.layers[l].self_attn.o_proj.input[0,-1,:].save()
                with tracer.invoke(waveI) as invoker:
                    # if l>0:
                    #     model.model.layers[l-1].output[0][0,-1,:]=prevhidden
                    zin = model.model.layers[l].self_attn.o_proj.input[0,-1,:]
                    z_reshapedin = einops.rearrange(zin, '(nh dh) -> nh dh',nh=32)
                    z = att
                    z_reshaped = einops.rearrange(z, '(nh dh) -> nh dh',nh=32)
                    z_reshapedin[head_idx,:] = z_reshaped[head_idx,:]
                    layer_outputI = model.lm_head(model.model.norm(model.model.layers[-1].output[0]))
                    outputI = model.output.save()
                    probsI = torch.nn.functional.softmax(layer_outputI, dim=-1).save()

            new_probI=np.dot(probsI[0,-1,tok_id_num].detach().cpu().numpy(),num_add)
            prob_I_A[l,head_idx]=probsI[0,-1,tok_id_num].detach().cpu().numpy()
            new_probdistI=probsI[0,-1,tok_id_num].detach().cpu().numpy()/new_probI
            pred_meanI=np.dot(new_probdistI,num)
            max_predI=int(num[np.argmax(new_probdistI)])
            # print(max_predI)
            grad_pred=max_predI-InterveneData[(16+toknext-1)]
            grad_target=max_pred-DataT[0,n,(16+toknext-1)]
            # print(grad_target)
            target_grad=InterveneData[(16+toknext-1)]+grad_target
            maegrad[l,head_idx]=abs(grad_target-grad_pred)
            mae_netoutput[l,head_idx]=abs(max_pred-max_predI)
            pred_llm=output["logits"][-1,-1,:]
            llm_pred = torch.argmax(pred_llm)
            labelgrad=model.tokenizer.encode(str(int(target_grad)))[1]
            labelnetoutput=model.tokenizer.encode(str(int(max_pred)))[1]
            logit_gaingrad[l,head_idx]=(outputI["logits"][-1,-1,labelgrad].detach().cpu().numpy()-outputnoint["logits"][-1,-1,labelgrad].detach().cpu().numpy())
            logit_lossnetoutput[l,head_idx]=(output["logits"][-1,-1,labelnetoutput].detach().cpu().numpy()-outputI["logits"][-1,-1,labelnetoutput].detach().cpu().numpy())
            # print(ind,pattok,l,(outputI["logits"][-1,-1,labelgrad].detach().cpu().numpy()-outputnoint["logits"][-1,-1,labelgrad].detach().cpu().numpy()))
            # print(n,tid,l,maegrad[tid,l]/(n+1),mae_netoutput[tid,l]/(n+1),logit_gaingrad[tid,l]/(n+1),logit_lossnetoutput[tid,l]/(n+1))
            if maegrad[l,head_idx]==0:
                accgrad[l,head_idx]=1
                # print("yes maegrad")
            if mae_netoutput[l,head_idx]==0:
                accnet[l,head_idx]=1

            # del q,restarget
            torch.cuda.empty_cache()
            gc.collect()

    print(ind,"--- %s seconds ---" % (time.time() - start_time))    
