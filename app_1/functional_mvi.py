#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


# In[5]:


def sep_na_rows(df):
        df=df.reset_index()
        df_train=df.dropna()
        index=df_train["index"]
        index_missing =[]
        for i in range(len(df)):
            if i not in index:
                index_missing.append(i)
        df_test = df.iloc[index_missing,:]
        return(df_train,df_test)

def robscaler(df):
        df_train,df_test=sep_na_rows(df)
        df_scaled=df_train[['SALES/REVENUE', 'FACEBOOK', 'GOOGLE', 'INSTAGRAM', 'E-MAIL',
       'NEWSPAPER', 'BANNERS & POSTERS', 'TV MEDIA', 'SMS',
       'TELEPHONE MARKETING', 'ROADSHOWS/EVENTS', 'OTHERS']]
        scaler=RobustScaler()
        robust_scaled_df = scaler.fit_transform(df_scaled)
        robust_scaled_df = pd.DataFrame(robust_scaled_df, columns=['SALES/REVENUE', 'FACEBOOK', 'GOOGLE', 'INSTAGRAM', 'E-MAIL',
       'NEWSPAPER', 'BANNERS & POSTERS', 'TV MEDIA', 'SMS',
       'TELEPHONE MARKETING', 'ROADSHOWS/EVENTS', 'OTHERS'])
        return(df_train,df_test,robust_scaled_df)

def rmse_calculator(df,feature,algo):
        X=df.drop(feature,axis=1)
        y=df[feature]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)
        
        
        
        if algo=='svr':
            model = SVR()
            param_grid ={'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['linear']} 
        if algo=="knn":
            model = KNeighborsRegressor()
            param_grid= {"n_neighbors":[1,3,5,7,9,11]}
        elif algo=="dec_tree":
            model = DecisionTreeRegressor()
            param_grid={}
        elif algo=="random_forest":
            model = RandomForestRegressor()
            param_grid={"n_estimators":[9,10,11,12]}
        else :
            model = LinearRegression()
            param_grid={}
        

        from sklearn import metrics
        
        
        from sklearn.model_selection import GridSearchCV
        grid = GridSearchCV(model,param_grid,refit=True,verbose=3)
        grid.fit(X_train,y_train)
        prediction=grid.predict(X_test)
        
        rmse=np.sqrt(metrics.mean_squared_error(y_test,prediction))
        return(rmse)

def model_selector(feature,robust_scaled_df):
        rmse_feature=[]
        algo=["svr","knn","dec_tree","random_forest","lin_reg"]
        for i in range(len(algo)):
            rmse=rmse_calculator(robust_scaled_df,feature,algo[i])
            rmse_feature.append(rmse)


        for i in range(len(rmse_feature)):
            if rmse_feature[i]==min(rmse_feature):
                selected_model=(algo[i])
                return(selected_model)  

def imputer (df_train,df_test,feature,selected_model):
        
        X_train=df_train.drop(feature,axis=1)
        y_train=df_train[feature]
        
        X_test=df_test[df_test[feature].isna()].drop(feature,axis=1)
        X_test=X_test.fillna(value=X_train.mean())
        y_test=df_test[df_test[feature].isna()][feature]
        
        if selected_model=='svr':
            model = SVR()
        elif selected_model=="knn":
            model = KNeighborsRegressor()
        elif selected_model=="dec_tree":
            model = DecisionTreeRegressor()
        elif selected_model=="random_forest":
            model=RandomForestRegressor()
        else:
            model=LinearRegression()
            
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        return(y_pred) 

def final_imputation(feature,data,final_value):
        a=data[data[feature].isna()]
        index=a["index"]
        index=np.array(index)
        
        for i in range(len(final_value)):
            j=index[i]
            data.loc[j,feature]=final_value[i]

def outliers_1(X):
        q1,q3=np.percentile(X,[25,75])
        iqr=q3-q1
        lower_range=q1-(iqr*1.5)
        upper_range=q3+(iqr*1.5)

        outliers_qm=[]
        for y in X:
            if (y < lower_range) | (y > upper_range):
                outliers_qm.append(y)
        return outliers_qm

def outliers_2(data):
    threshold=3
    mean_data=np.mean(data)
    std_data=np.std(data)
    outliers=[]
    for y in data:
        z_score=(y-mean_data)/std_data
        if np.abs(z_score)>threshold:
            outliers.append(y)
    return(outliers)


# In[6]:


def data_cleaner(df):

    column_list=df.columns

    df.dropna(axis=1,thresh=(int(df.shape[0]/2)),inplace=True)
    df.dropna(axis=0,thresh=(int(df.shape[1]/2)),inplace=True)
    Date_backup=df["DATE"]
    Date_df=df["DATE"].str.split("/",expand=True)
    Date_df.rename(columns={0:"DATE",1:"Month",2:"Year"},inplace=True)

    df.drop("DATE",axis=1,inplace=True)
    df["Month"]=Date_df["Month"]
    df["Last_week"]=0
    for i in range(len(df["Month"])-1):
        if df["Month"][i]!= df["Month"][i+1]:
            df["Last_week"][i]=1


    df.drop(["Month"],axis=1,inplace=True)   
    df=df.reset_index()

    missing_feature=[]
    for i in range(len(df.columns)):
        if df.isnull().any()[i]:
            missing_feature.append(df.columns[i])

    feature=missing_feature
    df_train,df_test,robust_scaled_df=robscaler(df)


    for a in feature:
        selected_model=model_selector(a,robust_scaled_df)
        final_value=imputer(df_train,df_test,a,selected_model)
        final_imputation(a,df,final_value)
    
    for i in df.columns[1:11]:
        outlier_list_1=outliers_1(df[i])
        outlier_list_2=outliers_2(df[i])

        for j in (outlier_list_1 and outlier_list_2):
            df[i].replace(j,np.nan,inplace=True)

    missing_feature=[]
    for i in range(len(df.columns)):
        if df.isnull().any()[i]:
            missing_feature.append(df.columns[i])

    feature=missing_feature
    df_train,df_test,robust_scaled_df=robscaler(df)


    for a in feature:
        selected_model=model_selector(a,robust_scaled_df)
        final_value=imputer(df_train,df_test,a,selected_model)
        final_imputation(a,df,final_value)

    df=df.drop("index",axis=1)
    df=df.iloc[:,:-1]
    frames=[Date_backup,df]
    df=pd.concat(frames,axis=1)
    return df




