from django.contrib import admin
from django.contrib.auth.models import User
from app_1.models import UserProfileInfo
from app_1.models import data_new

# Register your models here.
admin.site.register(data_new)
# Register your models here.
admin.site.register(UserProfileInfo)
