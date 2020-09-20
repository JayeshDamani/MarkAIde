from django.shortcuts import render,redirect,reverse 
from django.http import HttpResponse,HttpResponseRedirect
from django.core.files.storage import FileSystemStorage
import pandas as pd
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import UserCreationForm
from .forms import SignUpForm,LoginForm
from django.contrib.auth import authenticate, login, logout
import plotly.graph_objs as go
from plotly.offline import plot
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from app_1.functional_mvi import data_cleaner,robscaler,sep_na_rows,rmse_calculator,model_selector,imputer,final_imputation,outliers_1,outliers_2
from .models import data_new
from .forms import fileForm
from django.contrib.auth.decorators import login_required

# from plot import Pie_plotter
from django.views.generic import TemplateView



"""def index(request):
	form = SignUpForm(request.POST)
	return render(request,'index.html',{'form':form}) """
def index(request):
	login_form =LoginForm(request.POST)
	if request.method=="POST" and login_form.is_valid():
		user =login_form.login(request)
		if user:
			if user.is_active:
				login(request,user)
				return redirect('index')
			else:
				return HttpResponse('<h1 style="text-align:center;color:red"> your account is not active</h1>')
		else:
			return HttpResponse("Invalid login details supplied.")
	else:
		form = SignUpForm(request.POST)
		return render(request,'Latest_index.html',{'form':form,'login_form':login_form})
@login_required
def home(request):
	return render(request,'home.html')	
	
def Register(request):
	register =False
	if request.method=="POST":
		#send the login page
		user_data =SignUpForm(data=request.POST)
		if user_data.is_valid():
			user =user_data.save()
			user.set_password(user.password)
			user.save()
			register =True
			return redirect('index')
		else:
			pass
			# print(user_data.errors)
			# return HttpResponse('data invalid')
	else:
		return redirect('index')
@login_required
def user_logout(request):
	logout(request)
	return redirect('index')

@login_required
def vis(request):
	if request.method=="POST":
		uploaded_file= request.FILES['file-csv']
		fs = FileSystemStorage()
		#filename = fs.save(uploaded_file.name,uploaded_file)
		user = request.user 
		user_id = user.id 
		username = user.username 
		title = uploaded_file.name
		file = uploaded_file
		data_in = data_new(user_id=user_id,username=username,title=title,file_csv=file)
		data_in.save()
		file_name = data_in.file_csv
		df=pd.read_csv(file_name, na_values=['#DIV/0!',"0"])
		data=data_cleaner(df)	
		Date_data=data["DATE"].str.split("/",expand=True)
		Date_data.rename(columns={0:"DATE",1:"Month",2:"Year"},inplace=True)
		data.drop("DATE",axis=1,inplace=True)
		data["Month"]=Date_data["Month"]
		data["Year"]=Date_data["Year"]
		Dict = {
		    '01':'Jan',
		    '02':'Feb',
		    '03':'Mar',
		    '04':'Apr',
		    '05':'May',
		    '06':'Jun',
		    '07':'Jul',
		    '08':'Aug',
		    '09':'Sep',
		    '10':'Oct',
		    '11':'Nov',
		    '12':'Dec'
		}
		data['Month']=data['Month'].map(Dict)
		data["Month_year"]=data["Year"]+"-"+data["Month"]

		fig_region=[]
		for i in data["Month_year"]:
			data_req=data[data["Month_year"]=="2016-Dec"]
			Marketing_spend=['FACEBOOK', 'GOOGLE', 'INSTAGRAM', 'E-MAIL',
	       'NEWSPAPER', 'BANNERS & POSTERS', 'TV MEDIA', 'SMS',
	       'TELEPHONE MARKETING', 'ROADSHOWS/EVENTS', 'OTHERS']
			spend=data_req.sum()[Marketing_spend]
			labels = Marketing_spend
			values = spend.values
			fig1 =go.Figure()
			fig1.add_traces(go.Pie(labels=labels, values=values))
			fig1.update_layout(width=500)
			fig1_region =plot(fig1,output_type='div',include_plotlyjs=False)
			fig_region.append(fig1_region)

		month_grouped = data.groupby('Month_year')['SALES/REVENUE'].sum()
		df2=pd.DataFrame(month_grouped,columns=["SALES/REVENUE"])
		df2=df2.reset_index()
		fig2 =px.bar(df2,x='Month_year',y="SALES/REVENUE")
		fig2.update_layout(width=500)
		fig2_region =plot(fig2,output_type='div',include_plotlyjs=False)

		feature=[]
		cor=data.corr().iloc[0,:]>0
		for i in range(len(cor)):
			if cor[i]:
				feature.append(cor.index[i])

		feature=feature[1:]
			

		lm=LinearRegression()
		scaler=StandardScaler()

		X=data[feature]
		# scaled_X=scaler.fit_transform(X)
		
		y=data["SALES/REVENUE"]

		# lm.fit(scaled_X+1,y)
		# coef=lm.coef_


		# labels = feature
		# values = coef
		# data_2 =go.Pie(labels=labels, values=values)
		# layout2 =go.Layout(title='Marketing_driver_lm')


		y=np.log(y)
		X=np.log(X)

		lm.fit(X,y)

		coef=lm.coef_
		coef_sum=coef.sum()
		coef_per=(coef/coef_sum)*100
		labels = feature
		values = coef
		fig3 =go.Figure()
		fig3.add_traces(go.Pie(labels=labels, values=values))
		fig3.update_layout(width=500,title_text='Marketing Driver')
		fig3_region =plot(fig3,output_type='div',include_plotlyjs=False)
		# data_3 =go.Pie(labels=labels, values=values)
		# layout3 =go.Layout(title='Marketing_driver_log')



		base_sale=np.power(np.e,lm.intercept_)
		avg_sale=data["SALES/REVENUE"].mean()

		coef_per=coef_per/100
		final_contri=(coef_per*(avg_sale-base_sale))

		a=[]
		for i in final_contri:
			a.append(i)
		a.append(base_sale)
		feature.append("Base Sales")
		labels = feature
		values = a
		fig4 =go.Figure()
		fig4.add_traces(go.Pie(labels=labels, values=values))
		fig4.update_layout(width=500,title_text='Sales_revenue_decomposition')
		fig4_region =plot(fig4,output_type='div',include_plotlyjs=False)
		# data_4 =go.Pie(labels=labels, values=values)
		# layout4 =go.Layout(title='Sales_revenue_decomposition')

	


		return render(request,'vis.html',{'fig':fig_region,'fig2':fig2_region,'fig3':fig3_region,'fig4':fig4_region,'MY':data["Month_year"].unique()})
		# return HttpResponse(filename)
	else:
		return HttpResponse('There was an error while loading') 


@login_required
def file_upload(request):
	if request.method=="POST":
		file_form =fileForm(request.POST,request.FILES)
		if file_form.is_valid():
			user =request.user
			f =file_form.save(commit=False)
			f.user_id = user.id
			f.username =user.username
			f.save()
			return redirect(reverse('update'))

	else:
		file_form = fileForm()
		user =request.user
		current_id = user.id
		file_list = data_new.objects.filter(user_id=current_id)

		return render(request,'settings.html',{"form":file_form,'file_list':file_list})




