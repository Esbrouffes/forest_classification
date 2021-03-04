# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 17:43:16 2021

@author: YBlachonpro
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv",sep=",")
print(df.columns)

sns.heatmap(df.corr())


plt.figure()

sns.boxplot(data=df[["Elevation","Aspect","Slope", 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']])


sns.boxplot(data=df,y="Aspect",x="Cover_Type")

sns.boxplot(data=df,y="Elevation",x="Cover_Type")


sns.boxplot(data=df,y="Slope",x="Cover_Type")


sns.boxplot(data=df,y="Aspect",x="Cover_Type")

