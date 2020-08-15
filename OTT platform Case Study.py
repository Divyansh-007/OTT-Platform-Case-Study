# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 11:25:05 2020

@author: Dell
"""

# Importing all required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error

# Importing the dataset
media = pd.read_csv('mediacompany.csv')
# Removing the garbage column
media = media.drop('Unnamed: 7',axis = 1)

# Exploring the dataset
media.head()
#Information about media dataframe
media.info()

#Converting date to Pandas datetime format
media['Date'] = pd.to_datetime(media['Date'])

# Deriving "days since the show started"
basedate = pd.Timestamp('2017-2-28')
def time_since(x):
    return (x-basedate).days
media['days'] = media['Date'].apply(time_since)

# Plot of days vs Views_show
media.plot.line(x='days', y='Views_show')

# Plot for days vs Views_show and days vs Ad_impressions
fig = plt.figure()
host = fig.add_subplot(111)

par1 = host.twinx()
par2 = host.twinx()

host.set_xlabel("Days")
host.set_ylabel("View_Show")
par1.set_ylabel("Ad_impression")

color1 = plt.cm.viridis(0)
color2 = plt.cm.viridis(0.5)
color3 = plt.cm.viridis(.9)

p1, = host.plot(media.days,media.Views_show, color=color1,label="View_Show")
p2, = par1.plot(media.days,media.Ad_impression,color=color2, label="Ad_impression")

lns = [p1, p2]
host.legend(handles=lns, loc='best')

# right, left, top, bottom
par2.spines['right'].set_position(('outward', 60))      
# no x-ticks                 
par2.xaxis.set_ticks([])

# Sometimes handy, same for xaxis
#par2.yaxis.set_ticks_position('right')
host.yaxis.label.set_color(p1.get_color())
par1.yaxis.label.set_color(p2.get_color())

# Derived Metrics
# Weekdays are taken such that 1 corresponds to Sunday and 7 to Saturday
# Generate the weekday variable
media['weekday'] = (media['days']+3)%7
media.weekday.replace(0,7, inplace=True)
media['weekday'] = media['weekday'].astype(int)

# Running first model(lm1) Weekday and Visitors
# Putting feature variable to X
X = media[['Visitors','weekday']]
# Putting response variable to y
y = media['Views_show']

lm1=LinearRegression()
lm1.fit(X,y)

# We use the method sm.add_constant(X) in order to add a constant. 
X = sm.add_constant(X)
# create a fitted model in one line
lm_1 = sm.OLS(y,X).fit()

# Create Weekend variable, with value 1 at weekends and 0 at weekdays
def cond(i):
    if i % 7 == 5: return 1
    elif i % 7 == 4: return 1
    else :return 0
    return i
media['weekend']=[cond(i) for i in media['days']]

# Running second model(lm2) Weekend and Visitors
# Putting feature variable to X
X2 = media[['Visitors','weekend']]
# Putting response variable to y
y2 = media['Views_show']

lm2=LinearRegression()
lm2.fit(X2,y2)

# We use the method sm.add_constant(X) in order to add a constant. 
X2 = sm.add_constant(X2)
# create a fitted model in one line
lm_2 = sm.OLS(y2,X2).fit()

# Running third model(lm3) Weekend, Visitors and Character_A
# Putting feature variable to X
X3 = media[['Visitors','weekend','Character_A']]
# Putting response variable to y
y3 = media['Views_show']

lm3=LinearRegression()
lm3.fit(X3,y3)

# We use the method sm.add_constant(X) in order to add a constant. 
X3 = sm.add_constant(X3)
# create a fitted model in one line
lm_3 = sm.OLS(y3,X3).fit()

# Create lag variable
media['Lag_Views'] = np.roll(media['Views_show'], 1)
media.Lag_Views.replace(108961,0, inplace=True)

# Running fourth model(lm4) Weekend, Visitors, Character_ A and Lag_Views
# Putting feature variable to X
X4 = media[['Visitors','weekend','Character_A','Lag_Views']]
# Putting response variable to y
y4 = media['Views_show']

lm4=LinearRegression()
lm4.fit(X4,y4)

# We use the method sm.add_constant(X) in order to add a constant. 
X4 = sm.add_constant(X4)
# create a fitted model in one line
lm_4 = sm.OLS(y4,X4).fit()

# Plotting Heat-map
plt.figure(figsize = (20,10))        
sns.heatmap(media.corr(),annot = True)

#Ad impression in million
media['ad_impression_million'] = media['Ad_impression']/1000000

# Running fifth model(lm5) Weekend, Character_ A and Ad_impressions_million
# Putting feature variable to X
X5 = media[['weekend','Character_A','ad_impression_million']]
# Putting response variable to y
y5 = media['Views_show']

lm5=LinearRegression()
lm5.fit(X5,y5)

# We use the method sm.add_constant(X) in order to add a constant. 
X5 = sm.add_constant(X5)
# create a fitted model in one line
lm_5 = sm.OLS(y5,X5).fit()

# Making predictions using the fifth model
X_pred = media[['weekend','Character_A','ad_impression_million']]
X_pred = sm.add_constant(X_pred)
Predicted_views = lm_5.predict(X_pred)

# Performance measure
mse = mean_squared_error(media.Views_show, Predicted_views)
r_squared = r2_score(media.Views_show, Predicted_views)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)

#Actual vs Predicted plot
c = [i for i in range(1,81,1)]
fig = plt.figure()
plt.plot(c,media.Views_show, color="blue", linewidth=2.5, linestyle="-")
plt.plot(c,Predicted_views, color="red",  linewidth=2.5, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)               
plt.xlabel('Index', fontsize=18)                               
plt.ylabel('Views', fontsize=16)                               

# Error terms plot
c = [i for i in range(1,81,1)]
fig = plt.figure()
plt.plot(c,media.Views_show-Predicted_views, color="blue", linewidth=2.5, linestyle="-")
fig.suptitle('Error Terms', fontsize=20)               
plt.xlabel('Index', fontsize=18)                      
plt.ylabel('Views_show-Predicted_views', fontsize=16)