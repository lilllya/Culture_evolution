import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde

# Parameters for age distribution
mu_A = 50          # Mean age
sigma_A = 10       # Standard deviation of age

# Parameters for health condition
H0 = 100           # Initial health condition when age is zero
lambda_decay = 0.05  # Decay rate for health condition

# Sample 20 individuals
num_samples = 2000
ages = np.random.normal(mu_A, sigma_A, num_samples)

# Calculate health condition for each sampled age
health_conditions = H0 * np.exp(-lambda_decay * ages)

# Print the sampled data
sampled_data = list(zip(ages, health_conditions))
'''
print("Sampled Data (Age, Health Condition):")
for idx, (age, health) in enumerate(sampled_data):
    print(f"Person {idx + 1}: Age = {age:.2f}, Health Condition = {health:.2f}")
'''
# Plotting the distributions
plt.figure(figsize=(18, 6))

# Age distribution
plt.subplot(1, 3, 1)
age_kde = gaussian_kde(ages)
x_age = np.linspace(min(ages) - 5, max(ages) + 5, 100)
plt.plot(x_age, age_kde(x_age), color='blue', label='Age Density')
plt.xlabel('Age')
plt.ylabel('Density')
plt.title('Distribution of Age')
plt.axvline(mu_A, color='red', linestyle='dashed', linewidth=1, label='Mean Age')
plt.legend()

# Health condition distribution
plt.subplot(1, 3, 2)
health_kde = gaussian_kde(health_conditions)
x_health = np.linspace(min(health_conditions) - 5, max(health_conditions) + 5, 100)
plt.plot(x_health, health_kde(x_health), color='orange', label='Health Condition Density')
plt.xlabel('Health Condition')
plt.ylabel('Density')
plt.title('Distribution of Health Condition')
mean_health = H0 * np.exp(-lambda_decay * mu_A)
plt.axvline(mean_health, color='red', linestyle='dashed', linewidth=1, label='Mean Health Condition')
plt.legend()

# Joint distribution in 3D
ax = plt.subplot(1, 3, 3, projection='3d')
hist, xedges, yedges = np.histogram2d(ages, health_conditions, bins=10, density=True)
xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construct arrays with the dimensions for the bars
dx = dy = 0.5 * np.ones_like(zpos)
dz = hist.ravel()

# Create the 3D bar plot
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', color='lightblue', alpha=0.7)
ax.set_xlabel('Age')
ax.set_ylabel('Health Condition')
ax.set_zlabel('Density')
ax.set_title('Joint Distribution of Age and Health Condition')

plt.tight_layout()
plt.show()
