from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from app_1.models import UserProfileInfo
from django.contrib.auth import login, authenticate
from app_1.models import data_new

class SignUpForm(forms.ModelForm):
	password = forms.CharField(widget=forms.PasswordInput())

	class Meta():
		model = User
		fields = ('username', 'email', 'password')

	def __init__(self, *args, **kwargs):
		super(SignUpForm, self).__init__(*args, **kwargs)
		self.fields['username'].widget.attrs.update({'class' : 'input-field user','placeholder':'Username'})
		self.fields['email'].widget.attrs.update({'class' : 'input-field mail','placeholder':'Email'})
		self.fields['password'].widget.attrs.update({'class' : 'input-field pwd','placeholder':'password'})

class LoginForm(forms.ModelForm):
	password = forms.CharField(widget=forms.PasswordInput())
	class Meta():
		model =User
		fields=('username','password')
	def __init__(self,*args,**kwargs):
		super(LoginForm,self).__init__(*args,**kwargs)
		self.fields['username'].widget.attrs.update({'class' : 'input-field user','placeholder':'Username'})
		self.fields['password'].widget.attrs.update({'class' : 'input-field pwd','placeholder':'Password'})
	def clean(self):
		username = self.cleaned_data.get('username')
		password = self.cleaned_data.get('password')
		user = authenticate(username=username, password=password)
		if not user or not user.is_active:
			raise forms.ValidationError("Sorry, that login was invalid. Please try again.")
			return self.cleaned_data
	def login(self, request):
		username = self.cleaned_data.get('username')
		password = self.cleaned_data.get('password')
		user = authenticate(username=username, password=password)
		return user

class UserProfileInfoForm(forms.ModelForm):
	class Meta():
		model = UserProfileInfo
		fields =('website',)
	def __init__(self,*args,**kwargs):
		super(UserProfileInfoForm,self).__init__(*args,**kwargs)
		self.fields['website'].widget.attrs.update({'class':'input-field mail'})
class fileForm(forms.ModelForm):
	class Meta():
		model = data_new
		fields=('title','file_csv')
