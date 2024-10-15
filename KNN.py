import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the data

data = pd.read_csv('D:\\SRH\\SEM 2\\AI\\ElectrGridStab.csv') #please give your own path
print(data.info())
print(data.describe())

# Drop rows with missing values
data = data.dropna()

# Normalize the data
scaler = MinMaxScaler()
data_normalized = pd.DataFrame(scaler.fit_transform(data.iloc[:, :-1]), columns=data.columns[:-1])

# Add the target variable back to the normalized data
data_normalized['stabf'] = data['stabf']

# Split the data into training and testing sets
train, test = train_test_split(data_normalized, test_size=0.2, random_state=4321)
train_features = train.iloc[:, :-1]
train_labels = train.iloc[:, -1]
test_features = test.iloc[:, :-1]
test_labels = test.iloc[:, -1]

# List of k-values to evaluate
k_values = [31]  # Add more k-values to this list if needed
for k in k_values:
    # Initialize and train the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_features, train_labels)

    # Make predictions on the test set
    predictions = knn.predict(test_features)

    # Calculate the confusion matrix and accuracy
    conf_matrix = confusion_matrix(test_labels, predictions)
    accuracy = accuracy_score(test_labels, predictions)

    # Print the results
    print(f'k={k}')
    print('Confusion Matrix:\n', conf_matrix)
    print('Accuracy:', accuracy)



