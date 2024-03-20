from django.db import models

# Create your models here.
class Student(models.Model):
    confused=models.IntegerField()
    lookingaway=models.IntegerField()
    drowsy=models.IntegerField()
    frustated=models.IntegerField()
    engaged=models.IntegerField()
    bored=models.IntegerField()