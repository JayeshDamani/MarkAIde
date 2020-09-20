from django.urls import path
from . import views

urlpatterns=[
	path('',views.index,name='index'),
	path('reg/',views.Register,name='reg'),
	path('home',views.home,name='home'),
	path('logout/',views.user_logout,name='logout'),
	path('update',views.file_upload,name='update'),

	#home contains the register
]
