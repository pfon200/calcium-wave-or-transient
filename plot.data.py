# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 11:48:28 2021

@author: pfon200
"""

import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')


# Setup the figure size
plt.rcParams['figure.figsize'] = (8,4)

filename = 'C:/Users/pfon200/Dropbox/Sarah Fong/Ca-waves and sparks/Rabbit/Data Analysis/spark analysis3_frequency.xlsx'
sheetname = 'Frequency'


# Load the dataset from a CSV file
df = pd.read_excel(filename, sheetname, usecols = 'B, C, D, E', skiprows=(0,1),
                   keep_default_na=True)

# to rename
df.columns = ['CTR.C','SS.C','CTR.M','SS.M']

val_dict = df.to_dict('list')

vals, names, xs = [],[],[]

for i,key in enumerate(val_dict):
    a = np.array(val_dict[key])
    b = a[np.logical_not(np.isnan(a))]
    val_dict[key] = b
    
    vals.append(b)
    xs.append(np.random.normal(i + 1, 0.04,b.shape[0]))  # alpha is the transparency of scatter points

# Define plot space
fig, ax = plt.subplots(figsize=(10, 6))

# Set plot title and axes labels
ax.set(title = "Spontaneous Calcium sparks under Control Conditions",
       xlabel = "", 
       ylabel = "Frequency (sparks/s/100um)")

# rectangular box plot
bp = ax.boxplot(val_dict.values(),
     patch_artist = True,
     showfliers=False)
ax.set_xticklabels(val_dict.keys())

# #changing color of box plots
colors = [(1,1,1), (0,0,0),  
          (1,1,1), (1,0,0)] 
for patch, color in zip(bp['boxes'], colors): patch.set_facecolor(color +(0.45,))

# changing color and linewidth of whiskers 
for whisker in bp['whiskers']: whisker.set(color ='#000000', linewidth = 1.5, linestyle =":") 

# changing color and linewidth of caps 
for cap in bp['caps']: cap.set(color ='#000000', linewidth = 2) 

# changing color and linewidth of medians 
for median in bp['medians']: median.set(color ='black', linewidth = 1) 

# changing style of fliers 
#for flier in bp['fliers']: flier.set(marker ='D', color ='#FFFFFF', alpha = 0.5) 

# x-axis labels 
ax.set_xticklabels(['C', 'SS', 'C', 'SS']) 


palette = ['k', 'k', 'r', 'r']
for x, val, c in zip(xs, vals, palette):
    ax.scatter(x, val, alpha=0.65, color=c, label = color)
#    ctl = ax.scatter(x, val, alpha=0.65, color = '', label = 'Control')
#    mets = ax.scatter(x, val, alpha=0.65, color = 'r', label = 'MetS')
     

ax.legend()
ax.grid(False)

plt.show()