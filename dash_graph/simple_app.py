import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from django_plotly_dash import DjangoDash
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np 
import matplotlib.pyplot as plt

def plot(data):


    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    app = DjangoDash('SimpleExample', external_stylesheets=external_stylesheets)
    Date_data=data["Date"].str.split("/",expand=True)
    Date_data.rename(columns={0:"DATE",1:"Month",2:"Year"},inplace=True)
    data.drop("Date",axis=1,inplace=True)
    data["Month"]=Date_data["Month"]
    data["Year"]=Date_data["Year"]


    data["Month_year"]=data["Year"]+"-"+data["Month"]


    data_req=data[data["Month_year"]=="2017-09"]

    Marketing_spend=["Door_drops","Banners","Press","RCSD","Email","SMS"]
    spend=data_req.sum()[Marketing_spend]
    labels = Marketing_spend
    values = spend.values
    data_1 =go.Pie(labels=labels, values=values)
    layout1 =go.Layout(title='Month_pie')


    plt.figure(figsize=(12,8))
    month_grouped = data.groupby('Month_year')['Sales'].sum()
    month_grouped.plot.bar()


    data.drop(["revenue","Location"],axis=1,inplace=True)

    feature=[]
    cor=data.corr().iloc[0,:]>0
    for i in range(len(cor)):
        if cor[i]:
            feature.append(cor.index[i])
        

    feature=feature[1:-1]

    lm=LinearRegression()
    scaler=StandardScaler()

    X=data[feature]
    scaled_X=scaler.fit_transform(X)
    y=data.Sales

    lm.fit(scaled_X+1,y)
    coef=lm.coef_


    labels = feature
    values = coef
    data_2 =go.Pie(labels=labels, values=values)
    layout2 =go.Layout(title='Marketing_driver_lm')


    y=np.log(y)
    X=np.log(X)

    lm.fit(X,y)

    coef=lm.coef_
    coef_sum=coef.sum()
    coef_per=(coef/coef_sum)*100
    labels = feature
    values = coef
    data_3 =go.Pie(labels=labels, values=values)
    layout3 =go.Layout(title='Marketing_driver_log')



    base_sale=np.power(np.e,lm.intercept_)
    avg_sale=data["Sales"].mean()

    coef_per=coef_per/100
    final_contri=(coef_per*(avg_sale-base_sale))

    a=[]
    for i in final_contri:
        a.append(i)
    a.append(base_sale)

    feature.append("Base Sales")

    labels = feature
    values = a
    data_4 =go.Pie(labels=labels, values=values)
    layout4 =go.Layout(title='Sales_revenue_decomposition')



    app.layout = html.Div([
        html.H1('Hello DataTale'),
        dcc.Graph(id='Month_pie',figure={'data':[data_1],'layout':layout1}),
        dcc.Graph(id='Marketing_driver_lm',figure={'data':[data_2],'layout':layout2}),
        dcc.Graph(id='Marketing_driver_log',figure={'data':[data_3],'layout':layout3}),
        dcc.Graph(id='Sales_revenue_decomposition',figure={'data':[data_4],'layout':layout4})
    ])


