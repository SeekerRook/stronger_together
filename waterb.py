import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      # self.conv1 = nn.Conv2d(1, 32, 3, 1)
      # self.conv2 = nn.Conv2d(32, 64, 3, 1)
      # self.dropout1 = nn.Dropout2d(0.25)
      # self.dropout2 = nn.Dropout2d(0.5)
      self.fc1 = nn.Linear(23, 128)
      self.fc2 = nn.Linear(128, 10)


    # x represents our data
    def forward(self, x):
      # Pass data through conv1
      # x = self.conv1(x)
      # # Use the rectified-linear activation function over x
      # x = F.relu(x)

      # x = self.conv2(x)
      # x = F.relu(x)

      # # Run max pooling over x
      # x = F.max_pool2d(x, 2)
      # # Pass data through dropout1
      # x = self.dropout1(x)
      # # Flatten x with start_dim=1
      # x = torch.flatten(x, 1)
      # Pass data through fc1
      x = self.fc1(x)
      x = F.relu(x)
      x = self.dropout2(x)
      x = self.fc2(x)

      # Apply softmax to x
      # output = F.log_softmax(x, dim=1)
      output = x
      return output


def preprocess(df):
  from matplotlib import transforms
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import OneHotEncoder
  from sklearn.impute import SimpleImputer
  df.pop('id')
  df.pop('Date')
  df.pop('Location')
  
  categorical_features = ['WindGustDir', 'WindDir3pm', 'WindDir9am']
  categorical_transformer = Pipeline(steps=[
      ('imputer', SimpleImputer(strategy='most_frequent')),
      ('scaler', OneHotEncoder(handle_unknown='ignore'))]) 

  from sklearn.impute import KNNImputer
  from sklearn.preprocessing import StandardScaler


  numeric_features = ['MinTemp','MaxTemp','Rainfall', 'Evaporation', 'Sunshine', 
                    'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm',
                    'Humidity9am', 'Humidity3pm', 'Pressure9am', 
                    'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
                    'Temp3pm','RainToday']


  numeric_transformer = Pipeline(steps=[
      ('imputer', KNNImputer(n_neighbors=5)), #SimpleImputer(strategy = "mean")
      ('scaler', StandardScaler())])
  
  
  from sklearn.compose import ColumnTransformer

  
  preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

  X = df.drop("RainTomorrow", axis = 1)
  y = df.RainTomorrow

  print("starting pipeline....")


  preprocess = Pipeline(steps=[('preprocessor', preprocessor)])


  dfPP = preprocess.fit_transform(X)


  print("Done")


  dfPP = pd.DataFrame(dfPP)
  # dfPP.to_csv('test_clean.csv')
  return dfPP,y

def getxy():
  
  X = pd.read_csv('train_clean.csv')
  X.pop('Unnamed: 0')
  df = pd.read_csv('train.csv')
  y = df.RainTomorrow

  return X,y


import pandas as pd

# test_set = preprocess(pd.read_csv('test.csv'))



X,y =getxy()

print(X.info())
test_set = pd.read_csv('test.csv')

# print(train_set)