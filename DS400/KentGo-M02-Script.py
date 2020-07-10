# M02 - Script

#import necessary packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import *

#reading Adult dataset from UCI repository & assign correct column titles
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'

origdf = pd.read_csv(url, header=None, delimiter=' *, *', engine='python')
origdf.drop(4, axis=1, inplace=True)
origdf.columns = ['Age', 'Work Class', 'Final Weight', 'Education', 'Marital Status', 'Occupation',
              'Relationship', 'Race', 'Sex', 'Capital Gain', 'Capital Loss', 'Hrs per Week', 'Native Country', 'Income']

#output original dataset to csv: KentGo-M02-Dataset.csv
origdf.to_csv('KentGo-M02-Dataset.csv', index=False)


#read original csv
df = pd.read_csv('KentGo-M02-Dataset.csv')


print('\nDataset from UCI \n')
print(df.head())

print('\nOriginal Dataset shape: ', df.shape, '\n')

#check unique values & counts for each variable
for col in df.columns:
    print(df[col].unique())
    print(df[col].value_counts())


#replacing missing data in Native Country with the most common (modal) entry
print('\nNo. of missing values in Native Country before replacement: ', df.loc[df['Native Country']=='?', 'Native Country'].count())
df['Native Country'].replace('?', 'United-States', inplace=True)
print('\nNo. of missing values in Native Country after replacement: ', df.loc[df['Native Country']=='?', 'Native Country'].count())


#drop missing values in columns Work Class & Occupation to simplify dataset
df.replace('?', np.NaN, inplace=True)
df.dropna(axis=0, inplace=True)
print('\nDataset shape after dropping missing values in Work Class & Occupation columns: ', df.shape, '\n')


#calculate mean & stdevs and account for outlier values in Capital Gain & Capital Loss columns
outlier = pd.DataFrame()
stats = pd.DataFrame()
df2 = df[['Capital Gain', 'Capital Loss']]

for i in ['Capital Gain', 'Capital Loss']:
    stats.loc[0,i] = np.mean(df[i]) + 2*np.std(df[i]) #high limit
    stats.loc[1,i] = np.mean(df[i]) - 2*np.std(df[i]) #low limit   
    outlier[i] = ((df[i] < stats.loc[1,i]) | (df[i] > stats.loc[0,i]))
    stats.loc[2,i] = np.mean(df2[~outlier][i]) #mean of non-outliers

print('\nNo. of outliers before replacement:')
print(df[outlier].count())


#replace outliers in Capital Gain & Capital Loss columns with mean
df[['Capital Gain', 'Capital Loss']] = df2.where(~outlier, stats.loc[2,['Capital Gain', 'Capital Loss']], axis=1) 

outlier2 = pd.DataFrame()
for j in ['Capital Gain', 'Capital Loss']: 
    outlier2[j] = ((df[j] < stats.loc[1,j]) | (df[j] > stats.loc[0,j]))

print('\nNo. of outliers after replacement:')
print(df[outlier2].count())


print('\nDataset after replacing outliers in Capital Gain & Capital Loss with mean')
print(df.head(), '\n')


#normalize Final Weight & Hrs per Week variables
df['Norm Final Weight'] = StandardScaler().fit(pd.DataFrame(df['Final Weight'])).transform(pd.DataFrame(df['Final Weight']))
df['Norm Hrs per Week'] = StandardScaler().fit(pd.DataFrame(df['Hrs per Week'])).transform(pd.DataFrame(df['Hrs per Week']))

print('\nNew Normalized Final Weight & Normalized Hrs per Week columns after using Standard Scaler:')
print(df[['Norm Final Weight', 'Norm Hrs per Week']].head(), '\n')


#bin Age into 8 bins of equal width
nb_age = 8
binwidth_age = 10
minbin_age = 10
maxbin1_age = minbin_age + 1*binwidth_age
maxbin2_age = minbin_age + 2*binwidth_age
maxbin3_age = minbin_age + 3*binwidth_age
maxbin4_age = minbin_age + 4*binwidth_age
maxbin5_age = minbin_age + 5*binwidth_age
maxbin6_age = minbin_age + 6*binwidth_age
maxbin7_age = minbin_age + 7*binwidth_age
maxbin8_age = minbin_age + 8*binwidth_age

df.loc[((df['Age']>minbin_age) & (df['Age']<=maxbin1_age)), 'Age Bin'] = '10-20'
df.loc[((df['Age']>maxbin1_age) & (df['Age']<=maxbin2_age)), 'Age Bin'] = '21-30'
df.loc[((df['Age']>maxbin2_age) & (df['Age']<=maxbin3_age)), 'Age Bin'] = '31-40'
df.loc[((df['Age']>maxbin3_age) & (df['Age']<=maxbin4_age)), 'Age Bin'] = '41-50'
df.loc[((df['Age']>maxbin4_age) & (df['Age']<=maxbin5_age)), 'Age Bin'] = '51-60'
df.loc[((df['Age']>maxbin5_age) & (df['Age']<=maxbin6_age)), 'Age Bin'] = '61-70'
df.loc[((df['Age']>maxbin6_age) & (df['Age']<=maxbin7_age)), 'Age Bin'] = '71-80'
df.loc[((df['Age']>maxbin7_age) & (df['Age']<=maxbin8_age)), 'Age Bin'] = '81-90'  



#bin Hrs per Week in to 10 bins of equal width
nb_hrs = 10
binwidth_hrs = 10
minbin_hrs = 0
maxbin1_hrs = minbin_hrs + 1*binwidth_hrs
maxbin2_hrs = minbin_hrs + 2*binwidth_hrs
maxbin3_hrs = minbin_hrs + 3*binwidth_hrs
maxbin4_hrs = minbin_hrs + 4*binwidth_hrs
maxbin5_hrs = minbin_hrs + 5*binwidth_hrs
maxbin6_hrs = minbin_hrs + 6*binwidth_hrs
maxbin7_hrs = minbin_hrs + 7*binwidth_hrs
maxbin8_hrs = minbin_hrs + 8*binwidth_hrs
maxbin9_hrs = minbin_hrs + 9*binwidth_hrs
maxbin10_hrs = minbin_hrs + 10*binwidth_hrs

df.loc[((df['Hrs per Week']>minbin_hrs) & (df['Hrs per Week']<=maxbin1_hrs)), 'Hrs per Week Bin'] = '0-10'
df.loc[((df['Hrs per Week']>maxbin1_hrs) & (df['Hrs per Week']<=maxbin2_hrs)), 'Hrs per Week Bin'] = '11-20'
df.loc[((df['Hrs per Week']>maxbin2_hrs) & (df['Hrs per Week']<=maxbin3_hrs)), 'Hrs per Week Bin'] = '21-30'
df.loc[((df['Hrs per Week']>maxbin3_hrs) & (df['Hrs per Week']<=maxbin4_hrs)), 'Hrs per Week Bin'] = '31-40'
df.loc[((df['Hrs per Week']>maxbin4_hrs) & (df['Hrs per Week']<=maxbin5_hrs)), 'Hrs per Week Bin'] = '41-50'
df.loc[((df['Hrs per Week']>maxbin5_hrs) & (df['Hrs per Week']<=maxbin6_hrs)), 'Hrs per Week Bin'] = '51-60'
df.loc[((df['Hrs per Week']>maxbin6_hrs) & (df['Hrs per Week']<=maxbin7_hrs)), 'Hrs per Week Bin'] = '61-70'
df.loc[((df['Hrs per Week']>maxbin7_hrs) & (df['Hrs per Week']<=maxbin8_hrs)), 'Hrs per Week Bin'] = '71-80'
df.loc[((df['Hrs per Week']>maxbin8_hrs) & (df['Hrs per Week']<=maxbin9_hrs)), 'Hrs per Week Bin'] = '81-90'
df.loc[((df['Hrs per Week']>maxbin9_hrs) & (df['Hrs per Week']<=maxbin10_hrs)), 'Hrs per Week Bin'] = '91-100'

print('\nNew Binned Age and Hrs per Week columns after binning:')
print(df[['Age Bin', 'Hrs per Week Bin']].head(), '\n')


#consolidate Education & Work Class
df.loc[df['Education']=='1st-4th','Education'] = 'Elmtry-school'
df.loc[df['Education']=='5th-6th','Education'] = 'Elmtry-school'
df.loc[df['Education']=='7th-8th','Education'] = 'Middle-school'
df.loc[df['Education']=='9th','Education'] = 'Some-HS'
df.loc[df['Education']=='10th','Education'] = 'Some-HS'
df.loc[df['Education']=='11th','Education'] = 'Some-HS'
df.loc[df['Education']=='12th','Education'] = 'Some-HS'
df.loc[df['Education']=='Assoc-acdm','Education'] = 'Some-college'
df.loc[df['Education']=='Assoc-voc','Education'] = 'Some-college'


df.loc[df['Work Class']=='Self-emp-not-inc','Work Class'] = 'Self-emp'
df.loc[df['Work Class']=='Self-emp-inc','Work Class'] = 'Self-emp'
df.loc[df['Work Class']=='Local-gov','Work Class'] = 'Gov'
df.loc[df['Work Class']=='State-gov','Work Class'] = 'Gov'
df.loc[df['Work Class']=='Federal-gov','Work Class'] = 'Gov'

print('\nEducation & Work Class columns after consolidation:')
print(df.head(), '\n')
 
    
#one-hot encode Race & Sex variables
df_ohe1 = pd.get_dummies(df['Race'], prefix_sep="__", columns='Race')
df_ohe2 = pd.get_dummies(df['Sex'], prefix_sep="__", columns='Sex')
df = pd.concat([df, df_ohe1, df_ohe2], axis=1)
print('\nNew Hot Endoded columns for Race & Sex variables:')
print(df[['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Female', 'Male']].head(), '\n')


#remove obsolete Race & Sex columns
df.drop(['Race', 'Sex'], axis=1, inplace=True)
print('\nDataset after preparation:')
print(df.head())
print(df.shape)
    
#output results to csv: KentGo-M02-Dataset.csv
df.to_csv('KentGo-M02-CleanedDataset.csv', index=False)