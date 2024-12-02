import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn import tree

# Step 1: Load the dataset
df = pd.read_csv(r'C:\shree project\p\Task_3\bank.csv', delimiter=';')

# Step 2: Inspect the data
print(df.head())
print(df.info())

# Step 3: Handle missing values (if any)
df.replace('unknown', pd.NA, inplace=True)
df.dropna(inplace=True) 

# Step 4: Convert categorical variables to numeric (using LabelEncoder for simplicity)
labelencoder = LabelEncoder()

# Convert all object type columns to numeric codes
for column in df.select_dtypes(include=['object']).columns:
    df[column] = labelencoder.fit_transform(df[column])

# Step 5: Split data into features (X) and target (y)
X = df.drop('y', axis=1)  
y = df['y']  

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 7: Initialize and train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 8: Make predictions
y_pred = clf.predict(X_test)

# Step 9: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 10: Visualize the Decision Tree
plt.figure(figsize=(20,10))
tree.plot_tree(clf, feature_names=X.columns, class_names=['Not Purchased', 'Purchased'], filled=True)
plt.show()
