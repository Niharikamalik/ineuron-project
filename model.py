import numpy as np
import pandas as pd 
from sklearn import model_selection,preprocessing
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression 

import pickle
# import warnings


df = pd.read_csv('adult.csv')
def hrs_edit(val):
    if (val<40):
        return ('<40 hrs')
    elif (val==40):
        return ('40 hrs')
    else:
        return ('>40hrs') 
df['hours-per-week']=df['hours-per-week'].apply(hrs_edit)
    
    ####### drop #####
df = df.drop(['education','fnlwgt','race','relationship','country'],axis = 1)

# replace#####

df['workclass'].replace(' ?',0,inplace = True)
df['occupation'].replace(' ?',0,inplace = True)

 
df['workclass'].replace(0,np.nan,inplace = True)
df['occupation'].replace(0,np.nan,inplace = True)

df["workclass"] = df["workclass"].fillna(df["workclass"].mode()[0])
df["occupation"] = df["occupation"].fillna(df["occupation"].mode()[0])

# marital status ##########################
def married(val):
    if val==' Never-married':
        return 'not-married'
    elif val==' Divorced':
        return 'not-married'
    elif val==' Separated':
        return 'not-married'
    elif val==' Widowed':
        return 'not-married'
    else:
        return 'married'
df['marital-status']=df['marital-status'].apply(married)    

def income(val):
    if val == '<=50K':
        return  0
    else :
        return  1
df['salary'] = df['salary'].apply(income)                                                               
# label encoding ###################################                                                                               
encoder = preprocessing.LabelEncoder()
def encode(df):
    df['workclass'] = encoder.fit_transform(df['workclass'])
    df['marital-status'] = encoder.fit_transform(df['marital-status'])
    df['occupation'] = encoder.fit_transform(df['occupation'])
    df['sex'] = encoder.fit_transform(df['sex'])
    df['salary'] = encoder.fit_transform(df['salary'])
    df['hours-per-week'] = encoder.fit_transform(df['hours-per-week'])
    return df

encoded_df = encode(df)

# logistic ##########################################

X = encoded_df.iloc[:,:-1]
y = encoded_df.iloc[:,-1]

X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y)
random_forest = RandomForestClassifier(n_estimators=10,
                            random_state=0)
random_forest.fit(X_train, y_train)

pickle.dump(random_forest,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
pickle.dump(encoded_df,open('label_encoding.pkl','wb'))
encoding = pickle.load(open('label_encoding.pkl','rb'))
