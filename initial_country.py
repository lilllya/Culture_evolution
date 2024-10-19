import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde,truncnorm
import random

class country():
    def __init__(self,mu_Y,sigma_Y,mu_A0,sigma_A,k,H0,lambda_decay,mu_M,sigma_M,rho,mu_D,sigma_D,mu_C,sigma_C,min_child,max_child,type,num_samples,name):
        # Parameters for age distribution, normal distribution
        self.mu_Y = mu_Y          # Mean age
        self.sigma_Y= sigma_Y       # Standard deviation of age
        # Parameters for adaptability distribution, adaptability=-k*age...
        self.mu_A0= mu_A0        # Initial mean adaptability
        self.sigma_A= sigma_A       # Standard deviation of adaptability
        self.k= k            # Rate of adaptability decrease with age
        # Parameters for health condition, exponential distribution
        self.H0= H0           # Initial health condition when age is zero
        self.lambda_decay= lambda_decay  # Decay rate for health condition
        # Parameters for Money distribution
        self.mu_M= mu_M      # Mean money
        self.sigma_M= sigma_M   # Standard deviation of money
        # Parameters for Degree distribution
        self.rho= rho # Correlation coefficient between money and degree
        self.mu_D= mu_D         # Mean degree (e.g., 16 years of education)
        self.sigma_D= sigma_D       # Standard deviation of degree
        # Parameters for the number of children distribution
        self.mu_C= mu_C              # Mean number of children
        self.sigma_C= sigma_C           # Standard deviation
        self.min_child= min_child       # Minimum number of children
        self.max_child= max_child       # Maximum number of children
        # Parameters for society type,socialism=1, individualism=0
        self.type= type
        # Parameters for culture name
        self.name = name
        # Generate samples for age
        self.num_samples= num_samples
    def calculate_ages(self):
        #return np.random.normal(self.mu_Y, self.sigma_Y, self.num_samples)
        return truncnorm.rvs(0, 100, loc=self.mu_Y, scale=self.sigma_Y, size=self.num_samples)
    def calculate_adaptabilities(self):
        ages=self.calculate_ages()
        return np.array([np.random.normal(self.mu_A0 - self.k * (age - self.mu_Y), self.sigma_A) for age in ages])
    def calculate_healths(self):
        ages = self.calculate_ages()
        return self.H0 * np.exp(-self.lambda_decay * ages)
    def calculate_degrees(self):
        ages = self.calculate_ages()
        degrees=np.array([truncnorm.rvs((0-self.mu_D)/self.sigma_D, (age-self.mu_D)/self.sigma_D, loc=self.mu_D, scale=self.sigma_D) for age in ages])
        return degrees
    def calculate_moneys(self):
        return np.array([np.random.normal(self.mu_M + self.rho * (self.sigma_M / self.sigma_D) * (d - self.mu_D), self.sigma_M * np.sqrt(1 - self.rho**2)) for d in self.calculate_degrees()])
    def calculate_children(self):
        ages = self.calculate_ages()
        return np.array([bool(19<age<36)*(truncnorm.rvs((self.min_child - self.mu_C) / self.sigma_C, (self.max_child - self.mu_C) / self.sigma_C, loc=self.mu_C, scale=self.sigma_C)) for age in ages])
    def calculate_society_types(self):
        return [self.type]*self.num_samples
    def calculate_culture_name(self):
        return [self.name]*self.num_samples
def sampled_data(c):
    return list(zip(c.calculate_ages(), c.calculate_adaptabilities(), c.calculate_healths(), c.calculate_degrees(), c.calculate_moneys(), c.calculate_children(),c.calculate_culture_name(),c.calculate_society_types()))

'''
A: old country, rich property, low adaptability, medium health condition, well-educated,low born rate, socialism
B: young country, medium property, high adaptability, high health condition, medium-educated, medium born rate, individualism
C: age-balanced country, rich property, high adapatbility, medium health condition, medium-educated, low born rate, individualism
D: old country, medium property,high adaptability, low health condition, low born rate, well-educated, individualism
E: age-balanced country, medium property, medium adaptability, medium health condition, high born rate, well-educated, socialism
'''
mu_Ys=[50,25,35,40,35]
sigma_Ys=[10,10,10,10,10]
mu_A0s=[60,100,85,90,70]
sigma_As=[15,15,15,15,15]
ks=[0.5,0.5,0.5,0.5,0.5]
H0s=[70,100,80,60,80]
lambda_decays=[0.01,0.01,0.01,0.01,0.01]
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
names=[0,1,2,3,4]

country_A=country(50,10,60,15,0.5,70,0.01,100000,10000,0.5,18,2,1,1,0,2,1,200,0)
country_B=country(25,10,100,15,0.5,100,0.01,50000,10000,0.5,14,2,2,1,0,5,0,200,1)
country_C=country(35,10,85,15,0.5,80,0.01,70000,10000,0.5,14,2,1,1,0,2,0,200,2)
country_D=country(40,10,90,15,0.5,60,0.01,60000,10000,0.5,18,2,1,1,0,2,0,200,3)
country_E=country(35,10,70,15,0.5,80,0.01,60000,10000,0.5,18,2,3,1,0,5,1,200,4)

sample_A=sampled_data(country_A)
sample_B=sampled_data(country_B)
sample_C=sampled_data(country_C)
sample_D=sampled_data(country_D)
sample_E=sampled_data(country_E)

tmp=sample_A+sample_B+sample_C+sample_D+sample_E
samples=[list(i) for i in tmp]
all_pos=[[i,j] for i in range(40) for j in range(40)]
occ_pos=random.sample(all_pos,len(samples))
ava_pos=[item for item in all_pos if item not in occ_pos]

'''
for idx, (age, adaptability, health, degree, money, children,name,society_type) in enumerate(sample_A):
    print(f"Person {idx + 1} Age = {age:.2f}, Adaptability = {adaptability:.2f}, Health = {health:.2f}, Money = {money:.2f}, Children = {children:.2f}, culture_name={name}, society_type={society_type:.2f}")

print(sample_A[0])
print(sample_B[0])
print(sample_C[0])
print(sample_D[0])
print(sample_E[0])
'''