from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

data = pd.read_csv('convexity_results.csv')  
data=data.dropna()

X = data.drop(['Patient_ID', 'Slice_Number', 'Output_Value'], axis=1)
y = data['Output_Value']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svm_model = SVR(kernel='rbf')  
svm_model.fit(X_train, y_train)

# Predict on the test set
svm_predictions = svm_model.predict(X_test)

pickle.dump(svm_model, open('svm.pkl','wb'))


