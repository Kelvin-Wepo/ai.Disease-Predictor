import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

# Importing the dataset
dataset = pd.read_csv('./data/breast_cancer.csv')
X = dataset.iloc[:, 1:-1].values  # Independent variable
y = dataset.iloc[:, -1].values  # Dependent variable

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the Random Forest Classifier model on the Training set
classifier = RandomForestClassifier(n_estimators=100, random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Calculating Accuracy
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Saving model to disk
joblib.dump(classifier, './models/BreastCancer_model.joblib')

# Loading model to compare the results
loaded_model = joblib.load('./models/BreastCancer_model.joblib')
