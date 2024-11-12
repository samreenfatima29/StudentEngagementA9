# Student Engagement Analysis in Online Meetings

## Overview

The **Student Engagement Analysis in Online Meetings** project leverages facial expression analysis to gain insights into student engagement during online meetings or classes. The objective is to understand a listener's attentiveness through individual facial emotions. This project aims to provide a comprehensive view of student engagement and attentiveness and offer detailed feedback of the entire session to the speaker through a detailed dashboard.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Dependencies](#dependencies)

## Introduction

In the era of virtual learning and remote meetings, it becomes challenging to gauge the engagement levels of participants. This project addresses this challenge by using computer vision and machine learning techniques to analyze facial expressions and provide real-time feedback on engagement levels. 

## Installation

### Prerequisites

- Python 3.10
- Django
- OpenCV
- TensorFlow
- NumPy

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/samreenfatima29/StudentEngagementA9.git
   cd STUDENTENGAGEMENTA9
2. **Set up a virtual environment:**
    ```bash
    python3.10 -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
3. **Install the required dependencies:**
    ```bash
    pip install django
    pip install opencv
    pip install tensorflow 
4. **Set up Django:**
    ```bash
    python manage.py makemigrations
    python manage.py migrate
5. **Generate the pre-trained model:**
    Run the classification.ipynb notebook in the project to train the model and generate the facial_recognizer.h5 file.
6. **Run the django server:**
    ```bash
    python manage.py runserver
### Usage
- Log in to the system:
- Open your web browser and navigate to http://127.0.0.1:8000/login/. 
- Enter your username, password, and ID to log in.
- Start the video capture and analysis:
Upon successful login, the camera will open to capture your expressions. The system will analyze the facial expressions and update the engagement attributes in the Django database.

- View the dashboard:
Open your web browser and navigate to http://127.0.0.1:8000/teacher to view the detailed dashboard, which shows the results of the engagement analysis.
Select the Student tab to view detailed attentiveness analysis of each student.

### Features
- Real-time facial expression recognition using a pre-trained Machine Learning learning model.
- Analysis of various engagement levels such as Looking Away, Bored, Confused, Drowsy, Engaged, and Frustrated.
- Integration with Django to store and update student engagement data.
- Detailed dashboard for visualizing the engagement metrics of participants.
- Individual student attentiveness information.

### Software Dependencies
- OpenCV
- TensorFlow
- NumPy
- Django
  
### Hardware Requirements
- Intel Core i5 processor
- 64-bit Operating System
- 16GB RAM
