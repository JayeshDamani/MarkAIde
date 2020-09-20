import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
init_notebook_mode(connected=True)
cf.go_offline()
import plotly.graph_objs as go
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


df=pd.read_csv("Cleaned_MMM.csv")


def Pie_plotter(df):
    df_req=df[df["Month_year"]==Month_year]

    Marketing_spend=["Door_drops","Banners","Press","RCSD","Email","SMS"]
    spend=df_req.sum()[Marketing_spend]
    labels = Marketing_spend
    values = spend.values
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    plot_div = plot(figure,output_type="div",include_plotlyjs=False)
    return plot_div


def Plot_all(df):

    Date_df=df["Date"].str.split("/",expand=True)
    Date_df.rename(columns={0:"DATE",1:"Month",2:"Year"},inplace=True)
    df.drop("Date",axis=1,inplace=True)
    df["Month"]=Date_df["Month"]
    df["Year"]=Date_df["Year"]


    df["Month_year"]=df["Year"]+"-"+df["Month"]


    plt.figure(figsize=(12,8))
    month_grouped = df.groupby('Month_year')['Sales'].sum()
    month_grouped.plot.bar()

    Pie_plotter("2017-09")

    df.drop(["revenue","Location"],axis=1,inplace=True)

    feature=[]
    cor=df.corr().iloc[0,:]>0
    for i in range(len(cor)):
        if cor[i]:
            feature.append(cor.index[i])
        

    feature=feature[1:-1]

    lm=LinearRegression()
    scaler=StandardScaler()

    X=df[feature]
    scaled_X=scaler.fit_transform(X)
    y=df.Sales

    lm.fit(scaled_X+1,y)
    coef=lm.coef_


    labels = feature
    values = coef
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    plot_div = plot(figure,output_type="div",include_plotlyjs=False)
    return plot_div



    y=np.log(y)
    X=np.log(X)

    lm.fit(X,y)

    coef=lm.coef_
    coef_sum=coef.sum()
    coef_per=(coef/coef_sum)*100
    labels = feature
    values = coef
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    plot_div = plot(figure,output_type="div",include_plotlyjs=False)
    return plot_div


    base_sale=np.power(np.e,lm.intercept_)
    avg_sale=df["Sales"].mean()

    coef_per=coef_per/100
    final_contri=(coef_per*(avg_sale-base_sale))

    a=[]
    for i in final_contri:
        a.append(i)
    a.append(base_sale)

    feature.append("Base Sales")

    labels = feature
    values = a
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    plot_div = plot(figure,output_type="div",include_plotlyjs=False)
    return plot_div

Plot_all(df)








