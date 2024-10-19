import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd  # Import pandas for DataFrame

# Parameters for Money distribution
mu_M = 50000      # Mean money
sigma_M = 10000   # Standard deviation of money

# Parameters for Degree distribution
mu_D = 16         # Mean degree (e.g., 16 years of education)
sigma_D = 2       # Standard deviation of degree

# Correlation coefficient between Money and Degree
rho = 0.5         # Adjust this value based on desired correlation

# Covariance matrix
covariance_matrix = np.array([[sigma_M**2, rho * sigma_M * sigma_D],
                               [rho * sigma_M * sigma_D, sigma_D**2]])

# Sample 20 individuals from the joint distribution
num_samples = 200
samples = np.random.multivariate_normal([mu_M, mu_D], covariance_matrix, num_samples)

# Extract money and degree from samples
money_samples = samples[:, 0]
degree_samples = samples[:, 1]

# Print the sampled data
sampled_data = list(zip(money_samples, degree_samples))
print("Sampled Data (Money, Degree):")
for idx, (money, degree) in enumerate(sampled_data):
    print(f"Person {idx + 1}: Money = {money:.2f}, Degree = {degree:.2f}")

# Plotting the results
plt.figure(figsize=(12, 6))

# Scatter plot of the joint distribution
plt.subplot(1, 2, 1)
plt.scatter(money_samples, degree_samples, color='blue', alpha=0.6)
plt.title('Joint Distribution of Money and Degree')
plt.xlabel('Money')
plt.ylabel('Degree')

# Create a DataFrame for KDE plot
data = pd.DataFrame({'Money': money_samples, 'Degree': degree_samples})

# KDE plot
plt.subplot(1, 2, 2)
sns.kdeplot(data=data, x='Money', y='Degree', cmap='Blues', fill=True, thresh=0, levels=10)
plt.title('Joint Distribution')
plt.xlabel('Money')
plt.ylabel('Degree')

plt.tight_layout()
plt.show()
