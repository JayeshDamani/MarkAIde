from django.db import models
from django.contrib import admin
from django.contrib.auth.models import User

# Create your models here.
class UserProfileInfo(models.Model):
    user = models.OneToOneField(User,on_delete=models.CASCADE)
    website = models.URLField(blank=True)
    def __str__(self):
        return self.user.username

class data_new(models.Model):
    user_id =models.IntegerField(unique=False)
    username = models.CharField(max_length=30)
    title = models.CharField(max_length=100)
    file_csv = models.FileField(upload_to='csv_data/')
    date = models.DateTimeField(auto_now_add=True)
