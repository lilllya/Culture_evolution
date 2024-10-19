import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import truncnorm

# Parameters for the number of children distribution
mu_C = 1              # Mean number of children
sigma_C = 1           # Standard deviation
lower_bound = 0       # Minimum number of children
upper_bound = 5       # Maximum number of children

# Truncated normal distribution for the number of children
# Calculate the truncated normal parameters
a, b = (lower_bound - mu_C) / sigma_C, (upper_bound - mu_C) / sigma_C

# Generate the truncated normal distribution
number_of_children = truncnorm.rvs(a, b, loc=mu_C, scale=sigma_C, size=1000)

# Age range (20 to 35)
#ages = np.arange(20, 36)
ages = np.arange(0, 80)
# Calculate the probability distribution for each age group
# Assume a uniform distribution of ages for illustration
#children_counts = [truncnorm.rvs(a, b, loc=mu_C, scale=sigma_C, size=1000) for _ in ages]
children_counts = [bool(19<age<36)*(truncnorm.rvs((0 - mu_C) / sigma_C, (5 - mu_C) / sigma_C, loc=mu_C, scale=sigma_C)) for age in ages]

# Calculate the mean number of children for visualization
mean_children = [np.mean(count) for count in children_counts]

# Plotting the distribution of the number of children
plt.figure(figsize=(12, 6))

# Plotting the mean number of children by age
plt.subplot(1, 2, 1)
plt.bar(ages, mean_children, color='lightblue', alpha=0.7)
plt.xticks(range(0,ages[-1]+1,5))
plt.title('Mean Number of Children by Age')
plt.xlabel('Age')
plt.ylabel('Mean Number of Children')

# Plotting the distribution of the number of children
plt.subplot(1, 2, 2)
sns.histplot(number_of_children, bins=30, kde=True, color='blue', alpha=0.5)
plt.title('Distribution of Number of Children')
plt.xlabel('Number of Children')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
