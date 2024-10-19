import numpy as np
import matplotlib.pyplot as plt
import random

def analyze(samples):
    array = np.array(samples)
    #average index for all cultures as a whole
    total_ave = np.mean(array, axis=0)
    subs=[[],[],[],[],[]]
    ages_ave=[]
    adaptability_ave=[]
    health_ave=[]
    degree_ave=[]
    money_ave=[]
    children_ave=[]
    society_type_ave=[]
    population=[]
    for i in range(len(samples)):
        #t=samples[i][6]
        if samples[i][6]==0:
            subs[0].append(samples[i])
        elif samples[i][6]==1:
            subs[1].append(samples[i])
        elif samples[i][6]==2:
            subs[2].append(samples[i])
        elif samples[i][6]==3:
            subs[3].append(samples[i])
        elif samples[i][6]==4:
            subs[4].append(samples[i])
    subs=np.array(subs,dtype=object)
    for i in range(len(subs)):
        if len(subs[i])==0:
            tmp_ave=np.array([0,0,0,0,0,0,random.choice([0, 1])])
        elif len(subs[i])>0:
            tmp_ave = np.mean(subs[i], axis=0)
        ages_ave.append(tmp_ave[0])
        adaptability_ave.append(tmp_ave[1])
        health_ave.append(tmp_ave[2])
        degree_ave.append(tmp_ave[3])
        money_ave.append(tmp_ave[4])
        children_ave.append(tmp_ave[5])
        society_type_ave.append(tmp_ave[-1])
        population.append(len(subs[i]))
    return total_ave, ages_ave,adaptability_ave,health_ave,degree_ave,money_ave,children_ave,society_type_ave,population
