import numpy as np
import random
from scipy.stats import truncnorm
from classifier import model
from initial_country import samples
from collect_data import analyze
import matplotlib.pyplot as plt

global samples,sum_child,all_pos,occ_pos,ava_pos
sum_child=0

mu_Ys=[50,25,35,40,35]
sigma_Ys=[10,10,10,10,10]
mu_A0s=[60,100,85,90,70]
sigma_As=[15,15,15,15,15]
ks=[0.5,0.5,0.5,0.5,0.5]
H0s=[70,100,80,60,80]
lambda_decays=[0.1,0.1,0.1,0.1,0.1]
mu_Ms=[100000,50000,70000,60000,60000]
sigma_Ms=[10000,10000,10000,10000,10000]
rhos=[0.5,0.5,0.5,0.5,0.5]
mu_Ds=[18,14,14,18,18]
sigma_Ds=[2,2,2,2,2]
mu_Cs=[1,2,1,1,3]
sigma_Cs=[1,1,1,1,1]
min_childs=[0,0,0,0,0]
max_childs=[2,5,2,2,5]
types=[1,0,0,0,1]
num_samples=[200,200,200,200,200]

def has_interaction(s):
    p=s[1]/120
    return 1 if random.random() < p else 0
def update_for_culture():
    global samples, sum_child, occ_pos, ava_pos
    dead_flag=[]
    full=0
    for i in range(len(samples)):
        s=samples[i]
        neighbors=random.sample(samples,10)
        new_s, ifdead, children=update_for_individual(s,neighbors)
        if ifdead==1:
            dead_flag.append(i)
        elif ifdead==0:
            samples[i]=new_s
        if len(children)>0:
            for c in children:
                if len(ava_pos)==0:
                    full=1
                    return full
                sum_child+=1
                samples.append(c)
                occ_pos.append(ava_pos[0])
                ava_pos.pop(0)
    if len(dead_flag)>0:
        tmp=samples.copy()
        samples=np.delete(np.array(tmp),dead_flag,axis=0).tolist()
        for dind in dead_flag:
            ava_pos.append(occ_pos[dind])
        pos_tmp=occ_pos.copy()
        occ_pos=np.delete(np.array(pos_tmp),dead_flag,axis=0).tolist()
    return full
def update_for_individual(s,neighbors):
    if len(neighbors)==0:
        res = update_single(s)
    elif len(neighbors)>0:
        ind=has_interaction(s)
        if ind==1:
            res=update_group(s,neighbors)
        elif ind==0:
            res=update_single(s)
    return res
def update_group(s,neighbors):
    s,ifdead,_=update_single(s)
    children = []
    if ifdead==1:
        return s, ifdead,children
    name=int(s[6])
    for neighbor in neighbors:
        #adaptability
        adp_lr = min(max(s[1]/120,0.2),1)  # Learning rate for adaptability
        if s[-1]!=neighbor[-1]:
            if s[-1]==1:
                adp_lr=adp_lr/2
        s[1] += adp_lr * (neighbor[1]-s[1])
        #health: health cannot be learnt in interaction
        '''
        heal_lr = min(max(s[2] / 100, 0), 0.2)  # Learning rate for health
        if s[-1] != neighbor[-1]:
            if s[-1] == 1:
                heal_lr = heal_lr / 2
        s[2] += heal_lr * (neighbor[2] - s[2])
        '''
        #degree
        deg_lr = min(max(s[3] / 20, 0), 0.2)  # Learning rate for degree
        if s[-1] != neighbor[-1]:
            if s[-1] == 1:
                deg_lr = deg_lr / 2
        if neighbor[3]>s[3]:
            s[3] += deg_lr * (neighbor[3] - s[3])
        #money
        diff=3*(s[3]-neighbor[3])+(s[0]-neighbor[0])
        if diff>5:
            p=0.8
        elif 0<diff<=5:
            p=0.6
        elif diff==0:
            p=0.5
        elif -5<=diff<0:
            p=0.4
        elif diff<-5:
            p=0.2
        exchange_amount = 0.05 * (s[4] + neighbor[4])  # Money exchange factor
        if s[-1] == 1 and neighbor[-1] == 1:  # Both are socialists
            s[4] += exchange_amount / 2
            neighbor[4] -= exchange_amount / 2
        else:
            if random.random() < p:
                s[4] += exchange_amount
                neighbor[4] -= exchange_amount
            else:
                s[4] -= exchange_amount
                neighbor[4] += exchange_amount
        s[4] = max(0, s[4])
        #children
        if 20 <= s[0] <= 35 and int(s[5]) > 0:
            for i in range(int(s[5])):
                child=s.copy()
                child[0]=0
                child[2]=H0s[name]
                child[3]=0
                child[4]=s[4]/(int(s[5]+1))
                child[5]=0
                children.append(child)
            s[4]=s[4]/(int(s[5]+1))
            s[5]=0
        #social type
        if s[0] < 35:
            if s[-1]==1:
                shift_probability = 0.1 * (s[1] / 100)
                if np.random.rand() < shift_probability and neighbor[-1]==0:
                    s[-1]=0
            elif s[-1]==0:
                shift_probability = 0.2 * (s[1] / 100)
                if np.random.rand() < shift_probability and neighbor[-1]==1:
                    s[-1]=1
    #culture name
    res=model.predict(np.array([s[0],s[1],s[2],s[3],s[4],s[5],s[7]]).reshape(1,-1))
    s[6]=int(res[0])
    return s, ifdead, children

def update_single(s):
    name=int(s[6])
    if s[2]==0:
        return s,1,[]
    ifdead=0
    #age
    s[0] += 1
    #adaptability
    #s[1] += s[1] - np.exp(0.1 * s[0] / 100) * s[1]
    #s[1] = max(0, min(120, s[1]))
    s[1]=np.random.normal(mu_A0s[name] - ks[name] * (s[0] - mu_Ys[name]), sigma_As[name])
    #health
    decay_rate = 0.04  # Exponential decay factor for health
    if s[4] > 70000:
        decay_rate = decay_rate / min((s[4] / 10000 - 6),5)
    elif s[4] < 30000:
        decay_rate = 2*decay_rate
    s[2] += s[2] - np.exp(decay_rate * s[0] / 100) * s[2]  # Age factor for health decline
    # If health falls below threshold, the individual is considered dead
    if s[2] < 35:
        s[2] = 0  # Mark as dead
        ifdead=1
    #degree
    if s[0] < 35 and s[1] > 80 and s[4] > 60000:  # Younger individuals may still be pursuing education
        s[3] += 0.6
    s[3] = max(0, min(20, s[0], s[3]))
    #money
    if s[0]>30:
        s[4] += np.exp(s[0]/100+(s[3]-16)/4)*4000
    else:
        s[4] += np.exp((s[3]-16)/4)*4000
    #children, new born children will not apper if there is no interaction
    if s[0]>35:
        s[5]=0
    elif s[0]==20:
        s[5]=truncnorm.rvs((min_childs[name] - mu_Cs[name]) / sigma_Cs[name], (max_childs[name] - mu_Cs[name]) / sigma_Cs[name],loc=mu_Cs[name], scale=sigma_Cs[name])
    elif s[0]<20:
        s[5]=0
    #culture_type, culture type will not change if there is no interaction
    return s, ifdead, []

'''
p1=[51.89053755897509, 54.02151168541031, 41.23293365204797, 20.96578816965122, 101534.36564656672, 0.0, 0, 1]
p2=[33.74619082291396, 113.62069699715215, 72.4328248932294, 14.324571917340847, 30041.09743846626, 0.0, 1, 0]
p3=[58.7186001937279, 73.11745657418892, 55.077021078589816, 15.284477403616766, 59437.577345679514, 0.0, 2, 0]
p4=[42.11432415158933, 81.97084321613485, 39.56126258541433, 18.167377824850995, 81742.67538012528, 0.0, 3, 0]
p5=[45.930367531441746, 75.93483609425027, 50.739360220787304, 17.07848040375459, 56187.2907522619, 0.0, 4, 1]
tmp=p1.copy()
'''
totals=[]
ages=[]
adapts=[]
healths=[]
degrees=[]
moneys=[]
childs=[]
types=[]
ts=[]
pops=[]
for t in range(100):
    total,age,adapt,health,degree,money,child,type,population=analyze(samples)
    ts.append(t)
    totals.append(total)
    ages.append(age)
    adapts.append(adapt)
    healths.append(health)
    degrees.append(degree)
    moneys.append(money)
    childs.append(child)
    types.append(type)
    pops.append(population)
    full=update_for_culture()
    if full:
        break
    print(len(samples))

#plot
country_name=['A','B','C','D','E']
min_pop=20
ages_masked = np.where(np.array(pops) >= min_pop, np.array(ages), np.nan)
adapts_masked = np.where(np.array(pops) >= min_pop, np.array(adapts), np.nan)
healths_masked = np.where(np.array(pops) >= min_pop, np.array(healths), np.nan)
degrees_masked = np.where(np.array(pops) >= min_pop, np.array(degrees), np.nan)
moneys_masked = np.where(np.array(pops) >= min_pop, np.array(moneys), np.nan)
childs_masked = np.where(np.array(pops) >= min_pop, np.array(childs), np.nan)
types_masked = np.where(np.array(pops) >= min_pop, np.array(types), np.nan)
plot_data=[pops,ages_masked,adapts_masked,healths_masked,degrees_masked,moneys_masked,childs_masked,types_masked]
#plot_data=[pops,ages,adapts,healths,degrees,moneys,childs,types]
plot_title=['population','average age','average adaptability','average health condition','average education','average property','average new-borns','average social types']
fig, axs = plt.subplots(2, 4, figsize=(12, 6))
for i in range(8):
    row = i // 4
    col = i % 4
    for j in range(5):
        axs[row, col].plot(ts, np.array(plot_data[i])[:, j], label=f'Culture{country_name[j]}')
    axs[row, col].set_title(f'{plot_title[i]}')
    axs[row, col].legend()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()
