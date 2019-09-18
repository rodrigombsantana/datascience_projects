# Sparkify project
Spark and machine learning to predict users churn

### Motivation
This is the Capstone project for the Udacity Data Scientist nanodegree

In this project we need to analyse a tiny subset representation of a much larger spark dataset (12 Gb) containing information about users behaviours of a music streaming app/service (Sparkify)
The small dataset version contains only 2 months of logs
The ultimate goal is to identify which parameters are the most likely to conduct the users to churn / leave the service

### Requirements
This project requires the following python libraries

- pyspark
- pandas
- Matplotlib
- Seaborn
- numpy
- datetime
- jupyter notebook

### Files Description
- Sparkify.ipynb	: jupyter notebook

### Result

I could get 85% (f1 score) with Random Forest

I was able to see that as much as the application is been used and the Thumbs Down is select for it more likely users will unsubscribe from the app.

So a good counter measure will be giving discounts or free month subscriptions for long time users that started disliking the service.

### Licensing, Authors, Acknowledgements
All credits go to Udacity for setting up this study

