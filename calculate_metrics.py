#Calculate classification reort
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset 
df = pd.read_csv('dataset.csv', encoding= 'unicode_escape')
df.dropna(inplace=True)
X = df.drop(['Target'], axis=1)
y = df['Target']

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# Create a decision tree classifier
classifier = DecisionTreeClassifier()
# Fit the classifier to the new training data and predict on the test data
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Calculate metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))