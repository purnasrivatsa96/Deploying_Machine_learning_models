import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, roc_auc_score

import joblib


# Individual pre-processing and training functions
# ================================================

def load_data(df_path):
    # Function loads data for training
    data = pd.read_csv(df_path)
    return data
    #pass



def divide_train_test(df, target):
    # Function divides data set in train and test
    X_train, X_test, y_train, y_test = train_test_split(
    df.drop(target, axis=1),  # predictors
    df[target],  # target
    test_size=0.2,  # percentage of obs in test set
    random_state=0) 
    return X_train, X_test, y_train, y_test
    #pass
    



def extract_cabin_letter(df, var):
    # captures the first letter
    df[var] = df[var].str[0] # captures the first letter

    return df
    #pass 



# def add_missing_indicator(df, var):
#     # function adds a binary missing value indicator
#     if(df[var].dtypes=='O'):
#     	df[var] = df[var].fillna('Missing')
#     else:
#     	df[var+'_NA'] = np.where(df[var].isnull(), 1, 0)
#     return df[var]
#     #pass


    
def impute_na(df,var,replacement='Missing'):
    # function replaces NA by value entered by user
    # or by string Missing (default behaviour)
    if(replacement=='Missing'):
        df[var] = df[var].fillna('Missing')
    else:
	    df[var+'_NA'] = np.where(df[var].isnull(), 1, 0)
	    df[var].fillna(replacement, inplace=True)
    return df[var]
    #pass



def remove_rare_labels(df, var,frequent_labels):
    # groups labels that are not in the frequent list into the umbrella
    # group Rare
    return np.where(df[var].isin(frequent_labels), df[var], 'Rare')

	#pass



def encode_categorical(df, var):
    # adds ohe variables and removes original categorical variable
    df = pd.concat([df,
                         pd.get_dummies(df[var], prefix=var, drop_first=True)
                         ], axis=1)
    df.drop(labels=var, axis=1, inplace=True)
    return df

    
    #pass


def check_dummy_variables(df, dummy_list):
    
    # check that all missing variables where added when encoding, otherwise
    # add the ones that are missing
    for var in dummy_list:
    	df[var] = 0
    return df
    #pass
    

def train_scaler(df, output_path):
    # train and save scaler
    scaler = StandardScaler()
    scaler.fit(df)
    joblib.dump(scaler, output_path)
    return scaler
    #pass
  
    

def scale_features(df, scaler):
    # load scaler and transform data
    scaler = joblib.load(scaler) # with joblib probably
    return scaler.transform(df)
    #pass



def train_model(df, target, output_path):
    # train and save model
    # initialise the model
    log_model = LogisticRegression(C=0.0005, random_state=0)
    
    # train the model
    log_model.fit(df, target)
    
    # save the model
    joblib.dump(log_model, output_path)

    # make predictions for test set
    class_ = log_model.predict(df)
    pred = log_model.predict_proba(df)[:,1]

	# determine mse and rmse
    print('train roc-auc: {}'.format(roc_auc_score(target, pred)))
    print('train accuracy: {}'.format(accuracy_score(target, class_)))
    
    return None
    #pass



def predict(df, model):
    # load model and get predictions
    model = joblib.load(model)

    return model.predict(df)
    

