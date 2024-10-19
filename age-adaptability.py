import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde

mu_Y = 50
sigma_Y = 10

mu_A0 = 100
sigma_A = 15
k = 0.5

num_samples = 200
ages = np.random.normal(mu_Y, sigma_Y, num_samples)

adaptabilities = np.array([np.random.normal(mu_A0 - k * (age - mu_Y), sigma_A) for age in ages])
'''
sampled_data = list(zip(ages, adaptabilities))
for idx, (age, adaptability) in enumerate(sampled_data):
    print(f"Person {idx + 1}: Age = {age:.2f}, Adaptability = {adaptability:.2f}")
'''
age_range = np.linspace(20, 80, 100)
adaptability_range = np.array([np.random.normal(mu_A0 - k * (age - mu_Y), sigma_A, num_samples) for age in age_range])

joint_hist, xedges, yedges = np.histogram2d(ages, adaptabilities, bins=[30, 30])
joint_prob = joint_hist / np.sum(joint_hist)
X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
joint_prob_contour = joint_prob.T  # 转置以获得正确的方向
fig = plt.figure(figsize=(18, 6))

#age marginal prob
plt.subplot(1, 3, 1)
age_kde = gaussian_kde(ages)
x_age = np.linspace(min(ages), max(ages), 100)
plt.plot(x_age, age_kde(x_age), color='blue', label='Age Density')
plt.xlabel('Age')
plt.ylabel('Density')
plt.title('Distribution of Age')
plt.axvline(mu_Y, color='red', linestyle='dashed', linewidth=1, label='Mean Age')
plt.legend()

#adaptability marginal prob
plt.subplot(1, 3, 2)
adaptability_kde = gaussian_kde(adaptabilities)
x_adaptability = np.linspace(min(adaptabilities), max(adaptabilities), 100)
plt.plot(x_adaptability, adaptability_kde(x_adaptability), color='orange', label='Adaptability Density')
plt.xlabel('Adaptability')
plt.ylabel('Density')
plt.title('Distribution of Adaptability')
plt.axvline(mu_A0, color='red', linestyle='dashed', linewidth=1, label='Initial Mean Adaptability')
plt.legend()

# 3D joint probability
ax = fig.add_subplot(1, 3, 3, projection='3d')
ax.plot_surface(X, Y, joint_prob_contour, cmap='viridis', edgecolor='none')
ax.set_xlabel('Age')
ax.set_ylabel('Adaptability')
ax.set_zlabel('Joint Probability Density')
ax.set_title('3D Joint Probability Distribution')

plt.tight_layout()
plt.show()
