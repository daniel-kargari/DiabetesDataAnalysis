import pandas as pd
import numpy as np

# Data Preprocessing 
diabetes_data = pd.read_csv('Diabetes_Dataset.csv')

# Inspect the dataset
print(diabetes_data.head())

# Check for missing values in each column
print(diabetes_data.isnull().sum())

# Check for duplicated rows
print(diabetes_data.duplicated().sum())


# Drop rows with missing values
diabetes_data_dropped = diabetes_data.dropna()

# Drop duplicated rows
diabetes_data_no_duplicates = diabetes_data.drop_duplicates()

# Save the cleaned dataset
diabetes_data_cleaned = diabetes_data_no_duplicates.dropna()  # Combined cleaning
diabetes_data_cleaned.to_csv('Diabetes_Dataset_Cleaned.csv', index=False)


# Scaling Numerical Features
from sklearn.preprocessing import StandardScaler
import pandas as pd 

scaler = StandardScaler()

# List of features to scale
features_to_scale = ['Age', 'BMI', 'Chol', 'TG', 'HDL', 'LDL', 'Cr', 'BUN']

# Perform scaling
scaled_features = scaler.fit_transform(diabetes_data[features_to_scale])

# Create a new DataFrame with scaled features
diabetes_data_scaled = pd.DataFrame(scaled_features, columns=features_to_scale)

# Print a preview of the scaled DataFrame
print("Scaled features:")
print(diabetes_data_scaled.head())  # Display the first 5 rows


# Splitting the Data
from sklearn.model_selection import train_test_split

X = diabetes_data_scaled
y = diabetes_data['Diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the resulting datasets
print("Data split completed.")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Decision Tree Implementations
# Simple Decision Tree:

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Creating and training the Decision Tree model
print('Simple Decision Tree:')

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Predicting test data
y_pred = dt_model.predict(X_test)

# Evaluating the model
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Visualizing the Decision Tree
fig = plt.figure(figsize=(40, 20))  
plot_tree(dt_model, feature_names=['Age', 'BMI', 'Chol', 'TG', 'HDL', 'LDL', 'Cr', 'BUN'], 
          class_names=['No Diabetes', 'Diabetes'], filled=True)
plt.title("Decision Tree Visualization")
fig.savefig('decision_tree_ultra_high_quality.png', dpi=600, bbox_inches='tight') 

# Decision Tree Implementations
# Random Forest:

from sklearn.ensemble import RandomForestClassifier

# Creating and training the Random Forest model
print('Random Forest:')
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predicting test data
y_pred_rf = rf_model.predict(X_test)

# Evaluating the model
print(classification_report(y_test, y_pred_rf))
print("Accuracy:", accuracy_score(y_test, y_pred_rf))

# Visualizing the Random Forest

import matplotlib.pyplot as plt

feature_importance = rf_model.feature_importances_
features = ['Age', 'BMI', 'Chol', 'TG', 'HDL', 'LDL', 'Cr', 'BUN']

fig = plt.figure(figsize=(40, 20))  
plt.barh(features, feature_importance, color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Feature Importance - Random Forest')
plt.gca().invert_yaxis()

fig.savefig('random_forest_ultra_high_quality.png', dpi=600, bbox_inches='tight') 


# K-Nearest Neighbors (KNN) Implementation
from sklearn.neighbors import KNeighborsClassifier

# Testing KNN with different numbers of neighbors (k)
print('KNN Results:')
for k in [3, 5, 7]:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    print(f'\nK={k}')
    print(classification_report(y_test, y_pred_knn))
    print("Accuracy:", accuracy_score(y_test, y_pred_knn))


# Support Vector Machine (SVM) Implementation
from sklearn.svm import SVC

# Testing SVM with different kernels
print('\nSVM Results:')
for kernel in ['linear', 'rbf', 'poly']:
    svm_model = SVC(kernel=kernel, random_state=42)
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    print(f'\nKernel={kernel}')
    print(classification_report(y_test, y_pred_svm))
    print("Accuracy:", accuracy_score(y_test, y_pred_svm))


# Naive Bayes Implementation
from sklearn.naive_bayes import GaussianNB

print('\nNaive Bayes Results:')
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

print(classification_report(y_test, y_pred_nb))
print("Accuracy:", accuracy_score(y_test, y_pred_nb))