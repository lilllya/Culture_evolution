import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import truncnorm

# Parameters for Age distribution
mu_A = 30             # Mean age
sigma_A = 5           # Standard deviation of age

# Parameters for Years of Study distribution
mu_S = 12             # Mean years of study
sigma_S = 3           # Standard deviation of years of study

# Number of samples
num_samples = 1000

# Sample ages from a normal distribution
ages = np.random.normal(mu_A, sigma_A, num_samples)
ages = np.clip(ages, 20, 40)  # Limit age between 20 and 40

# Calculate years of study for each individual, truncated by their age
years_of_study = []
'''
for age in ages:
    # Set bounds for years of study based on age
    lower_bound = 0
    upper_bound = age
    a, b = (lower_bound - mu_S) / sigma_S, (upper_bound - mu_S) / sigma_S

    # Sample years of study from the truncated normal distribution
    study_years = truncnorm.rvs(a, b, loc=mu_S, scale=sigma_S)
    years_of_study.append(study_years)

years_of_study = np.array(years_of_study)
'''
years_of_study=np.array([truncnorm.rvs((0-mu_S)/sigma_S, (age-mu_S)/sigma_S, loc=mu_S, scale=sigma_S) for age in ages])
# Plotting the results
plt.figure(figsize=(12, 6))

# Scatter plot of Age vs. Years of Study
plt.subplot(1, 2, 1)
plt.scatter(ages, years_of_study, color='blue', alpha=0.5)
plt.title('Age vs. Years of Study')
plt.xlabel('Age')
plt.ylabel('Years of Study')

# KDE plot for the joint distribution
plt.subplot(1, 2, 2)
sns.kdeplot(x=ages, y=years_of_study, cmap='Blues', fill=True, thresh=0, levels=10)
plt.title('KDE of Joint Distribution of Age and Years of Study')
plt.xlabel('Age')
plt.ylabel('Years of Study')

plt.tight_layout()
plt.show()
