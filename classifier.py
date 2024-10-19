import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from initial_country import sample_A, sample_B, sample_C, sample_D, sample_E

data=[]
label=[]
samples=[sample_A, sample_B, sample_C, sample_D, sample_E]
for sample in samples:
    for i in range(len(sample)):
        tmp=list(sample[i][:6]+sample[i][7:])
        data.append(tmp)
        label.append(sample[i][6])
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(np.array(data), np.array(label), test_size=0.2, random_state=42)

# Create the Gaussian Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
'''
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Example prediction
sample = np.array([40, 70, 70, 40000, 10, 0, 0])  # Sample to classify
predicted_culture = model.predict(sample.reshape(1,-1))
predicted_proba = model.predict_proba(sample.reshape(1,-1))

print(f"Predicted culture for the sample: {predicted_culture[0]}")
print(f"Prediction probabilities: {predicted_proba[0]}")

'''