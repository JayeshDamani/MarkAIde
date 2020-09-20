import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from django_plotly_dash import DjangoDash
import pandas as pd 

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
data =go.Pie(labels=labels, values=values)
layout =go.Layout(title='Pie chart')


app.layout = html.Div([
    html.H1('Hello DataTale'),
    dcc.Graph(id='Pie_chart',figure={'data':[data],'layout':layout})
])


