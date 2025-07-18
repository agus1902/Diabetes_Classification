
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split

def sigmoid_function(X):
  return 1/(1+math.e**(-X))

def log_regression4(X, y, alpha, epochs):
  y_ = np.reshape(y, (len(y), 1)) # shape (150,1)
  N = len(X)
  theta = np.random.randn(len(X[0]) + 1, 1) #* initialize theta
  X_vect = np.c_[np.ones((len(X), 1)), X] #* Add x0 (column of 1s)
  avg_loss_list = []
  loss_last_epoch = 9999999
  for epoch in range(epochs):
    sigmoid_x_theta = sigmoid_function(X_vect.dot(theta)) # shape: (150,5).(5,1) = (150,1)
    grad = (1/N) * X_vect.T.dot(sigmoid_x_theta - y_) # shapes: (5,150).(150,1) = (5, 1)
    best_params = theta
    theta = theta - (alpha * grad)
    hyp = sigmoid_function(X_vect.dot(theta)) # shape (150,5).(5,1) = (150,1)
    avg_loss = -np.sum(np.dot(y_.T, np.log(hyp) + np.dot((1-y_).T, np.log(1-hyp)))) / len(hyp)
    avg_loss_list.append(avg_loss)
    loss_step = abs(loss_last_epoch - avg_loss) #*
    loss_last_epoch = avg_loss #*
  return best_params

import numpy as np
import pandas as pd
df = pd.read_csv('diabetes.csv')

X = df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]


y_no = (df["Outcome"] == 0).astype(int) 
y_yes = (df["Outcome"] == 1).astype(int)

y_diabetes_types = [y_no, y_yes]
y_diabetes_types = {'No':y_no,
                'Yes':y_yes}
predicted_probs = {'No':0.0,
                   'Yes':0.0}
actual_y = {'No':0,
            'Yes':0}

for key, y_diabetes_type in y_diabetes_types.items():

  X_train, X_test, y_train, y_test = train_test_split(X, y_diabetes_type, test_size=0.2, random_state=0)

  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)
  # Train model
  epochs = 10000
  alpha = 0.01
  best_params = log_regression4(X_train, y_train, alpha, epochs)

  index_ = 10
  X_to_predict = [list(X_test[index_])]

  X_to_predict = np.c_[np.ones((len(X_to_predict), 1)), X_to_predict] # add x0 for bias
  pred_probability = sigmoid_function(X_to_predict.dot(best_params))
  print(pred_probability)
  predicted_probs[key] == pred_probability[0][0]
  print('Our model calculated probability of sample being {}, is: {}%'.format(key, round(pred_probability[0][0]*100,2)))
  actual_y[key] = y_test[index_]

max_key = max(predicted_probs, key=predicted_probs.get)
print('\n', predicted_probs)
print('\nModel Prediction: {}'.format(max_key))
max_actual_y = max(actual_y, key=actual_y.get)
print('Real value is: {}'.format(max_actual_y))


