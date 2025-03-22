
import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd
import seaborn as sns
import joblib as jb
import missingno
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn_features.transformers import DataFrameSelector
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import Ridge , Lasso
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.pipeline import Pipeline
df= pd.read_csv("D:\work\housing_project\housing.csv")

df['ocean_proximity']=  df['ocean_proximity'].replace('<1H OCEAN','1H OCEAN')

num_cols = df.select_dtypes(include = 'number').columns
# Calculate the ratio of total_bedrooms to total_rooms
df['rooms_per_household'] = df['total_rooms'] / df['households']
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
df['population_per_household'] = df['population'] / df['households']

# Display the updated DataFrame with the new features

num_cols = df.select_dtypes(include = 'number').columns
correlation_matrix = df[num_cols].corr()
df[num_cols].corr()['median_house_value'].sort_values(ascending=False)

x = df.drop('median_house_value',axis=1)
y = df['median_house_value']
x_train_full , x_test_full , y_train_full, y_test_full  = train_test_split(x,y , shuffle = True , random_state=42,test_size=.15)

#preprocessing
#simple imputer

imputer = SimpleImputer (strategy='median')
num_col= x_train_full.select_dtypes(include=['int32','float32','int64','float64']).columns
imputer .fit(x_train_full[num_col])

x_train_filled  = imputer.transform(x_train_full[num_col])
x_test_filled  = imputer.transform(x_test_full[num_col])
pd.DataFrame(x_train_filled,columns=num_col).isna().sum()

# Create DataFrames from the NumPy arrays
x_train_filled = pd.DataFrame(x_train_filled, columns=num_col)
x_test_filled = pd.DataFrame(x_test_filled, columns=num_col)
scaler = StandardScaler()
scaler.fit(x_train_filled[num_col])


from sklearn.preprocessing import LabelEncoder
lbl_encoder  = LabelEncoder()
x_train_filled ['ocean_proximity'] = df['ocean_proximity']
x_test_filled ['ocean_proximity'] = df['ocean_proximity']
cat_cols = x_train_filled.select_dtypes(include='object').columns
lbl_encoder.fit(x_train_filled[cat_cols])
x_train_encodeing  = lbl_encoder.transform(x_train_filled[cat_cols])
x_test_encodeing  = lbl_encoder.transform(x_test_filled[cat_cols])
ohe = OneHotEncoder()
ohe.fit(x_train_filled[cat_cols].values)
x_train_ohe = ohe.transform(x_train_filled[cat_cols])
x_test_ohe = ohe.transform(x_test_filled[cat_cols])

from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]

categ_pipline = Pipeline([('selector',DataFrameSelector(cat_cols)),('Imputer', SimpleImputer(strategy='constant',fill_value='missing')), ('ohe',OneHotEncoder(sparse_output= False ))])
nums_pipline = Pipeline([('selector',DataFrameSelector(num_col)),('imputer',imputer),('scaler',scaler)])
total_pipline  = FeatureUnion([('num_pipline',nums_pipline),('cate_pipline',categ_pipline)])
x_train_final = total_pipline.fit_transform(x_train_filled)
x_test_final = total_pipline.transform(x_test_filled)


def preprocess_newinstance (x_new):
    return total_pipline.transform(x_new)
