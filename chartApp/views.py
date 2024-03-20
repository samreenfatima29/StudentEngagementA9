from django.shortcuts import render
from . models import *
from django.db.models import Count
# def index(request):
    # confused_count = Student.objects.filter(confused__gt=0).count()
    # looking_away_count = Student.objects.filter(lookingaway__gt=0).count()
    # drowsy_count = Student.objects.filter(drowsy__gt=0).count()
    # frustrated_count = Student.objects.filter(frustated__gt=0).count()
    # engaged_count = Student.objects.filter(engaged__gt=0).count()
    # bored_count = Student.objects.filter(bored__gt=0).count()
 # context = {
    #     'confused': confused_count,
    #     'lookingaway': looking_away_count,
    #     'drowsy': drowsy_count,
    #     'frustated': frustrated_count,
    #     'engaged': engaged_count,
    #     'bored': bored_count
    # }
    # return render(request, 'index.html', context)
    

from django.db.models import Sum

def index(request):
    # Aggregate counts for each attribute across all student objects
    attribute_counts = Student.objects.aggregate(
        total_confused=Sum('confused'),
        total_looking_away=Sum('lookingaway'),
        total_drowsy=Sum('drowsy'),
        total_frustated=Sum('frustated'),
        total_engaged=Sum('engaged'),
        total_bored=Sum('bored'),
    )

    context = {
        'confused': attribute_counts['total_confused'],
        'lookingaway': attribute_counts['total_looking_away'],
        'drowsy': attribute_counts['total_drowsy'],
        'frustated': attribute_counts['total_frustated'],
        'engaged': attribute_counts['total_engaged'],
        'bored': attribute_counts['total_bored']
    }
    return render(request, 'index.html', context)

    
   


