import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

data = pd.read_csv("file:///C:\\Users\\Sayyed_Talha_Bacha\\Desktop\\practus_dl\\diabetes-dataset.csv")
print(data)
print(data.shape)

featu_data = data.iloc[:,0:8]
targe_data = data['Outcome']
print(featu_data,targe_data)

xtrain ,xtest, ytrain, ytest = train_test_split(featu_data,targe_data, test_size=0.2)

model = Sequential()
model.add(Dense(12, input_dim = 8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = 'accuracy')
model.fit(xtrain,ytrain, epochs = 150, batch_size = 10)

from sklearn.metrics import accuracy_score
y_pred = (model.predict(xtest) > 0.5).astype("int32")
print(accuracy_score(ytest, y_pred))


import pickle
pickle.dump(model, open('model.pkl', 'wb'))


Pregnancies = int(input('inter pragnance # : '))
Glucose = int(input('inter glucose level : '))
BloodPressure = int(input('inter BP : '))
SkinThickness = int(input('Inter sk : '))
Insulin = int(input('Inter insulin level : '))
BMI = int(input('BMI : '))
DiabetesPedigreeFunction = int(input('Inter the DPF : '))
Age = int(input('Inter Age : '))
arr = ([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI])
print(arr)


arr = [Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]
arr = np.expand_dims(arr, axis = 0) # Reshape to add a batch dimension

# Load the pickled model
with open('model.pkl', 'rb') as model_file:
    pickled_model = pickle.load(model_file)

# Convert the list 'arr' to a NumPy array
arr = np.array(arr)

# Make predictions
pre = pickled_model.predict(arr)
class_labels = (pre > 0.5).astype(int)
if class_labels == [1]:
  print('diabates')
else:
  print('No diabates') 


