import numpy as np 
import pandas as pd 
import pandas_profiling
import matplotlib.pyplot as plt 
import seaborn as sns
import os
from sklearn.decomposition import PCA 

FILENAME = "/home/dennis/Desktop/Link to datascience_job_portfolio/problem-set-show-case/data/transfusion.data"

df = pd.read_csv(FILENAME, header="infer")

# report = pandas_profiling.ProfileReport(df)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
fig.suptitle('Distribution Plots for each feature', fontsize=15)
sns.distplot(df['Monetary (c.c. blood)'], hist=True, norm_hist=True, ax=ax1)
ax1.set_xlabel('Monetary (c.c. blood)', fontsize=10)
sns.distplot(df['Recency (months)'], hist=True, norm_hist=True, ax=ax2)
ax2.set_xlabel('Recency (months)', fontsize=10)
sns.distplot(df['Frequency (times)'], hist=True, norm_hist=True, ax=ax3)
ax3.set_xlabel('Frequency (times)', fontsize=10)
sns.distplot(df['Time (months)'], hist=True, norm_hist=True, ax=ax4)
plt.show()

