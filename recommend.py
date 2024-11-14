import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pickle

# Step 1: Load the dataset
df = pd.read_csv("Crop_recommendation.csv")

# Step 2: Explore the data
print("First 5 rows of the dataset:")
print(df.head())

# Check for missing values and data types
print("\nData Info (Data Types and Missing Values):")
print(df.info())

# Summary statistics of numerical columns
print("\nSummary Statistics:")
print(df.describe())

# Step 3: Data Cleaning (handle missing values, duplicates)
# If there are any missing values, fill them with the mean of the column
df.fillna(df.mean(), inplace=True)

# Remove any duplicates
df = df.drop_duplicates()

# Step 4: Convert categorical column ('label') to numerical values using LabelEncoder
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Step 5: Scaling numerical features (temperature, humidity, pH, rainfall)
scaler = StandardScaler()
df[['temperature', 'humidity', 'ph', 'rainfall']] = scaler.fit_transform(df[['temperature', 'humidity', 'ph', 'rainfall']])

# Step 6: Visualizing Data
# Correlation Heatmap
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Distribution of crops (labels)
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='label')
plt.title('Crop Distribution')
plt.show()

# Step 7: Prepare data for Machine Learning
# Split the data into features (X) and target (y)
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]  # Feature columns
y = df['label']  # Target column (crop type)

# Step 8: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9: Train a Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 10: Evaluate the model
accuracy = model.score(X_test, y_test)
print(f'\nModel Accuracy: {accuracy:.2f}')

# Confusion Matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
with open('crop_model.pkl', 'wb') as file:
    pickle.dump(model, file)
loaded_model=pickle.load(open('crop_model.pkl','rb'))
# Optionally, load the saved model and use it to make predictions
sample_data = X_test.iloc[0].values.reshape(1, -1)  # Reshape a single sample from the test set
predicted_crop = label_encoder.inverse_transform(loaded_model.predict(sample_data))
print(predicted_crop)
