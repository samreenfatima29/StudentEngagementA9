from django.db import models

# Create your models here.
class Student(models.Model):
    confused=models.IntegerField(default=0)
    lookingaway=models.IntegerField(default=0)
    drowsy=models.IntegerField(default=0)
    frustated=models.IntegerField(default=0)
    engaged=models.IntegerField(default=0)
    bored=models.IntegerField(default=0)