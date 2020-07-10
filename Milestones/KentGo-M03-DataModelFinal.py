# M03 - DataModelFinal

#import necessary packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import *
import matplotlib

#we're using shopping dataset from uci repository to see if we can predict if a user is going to make a purchase based on different attributes
#reading data using pandas read_csv function
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv'
df = pd.read_csv(url)
df.loc[df['Month']=='June','Month'] = 'Jun'


print('\nThis dataset & exercise try to predict if a website visitor is likely to make a purchase online based on various attributes')

print('\nDataset from UCI Repository')
print(df.head())
r, c =  df.shape
print('\nDataset Original Shape:', r, 'observations and', c, 'attributes')
print('\nData Type:')
print(df.dtypes)
print('\nDescription of Numeric Data')
print(df.describe())


#plot the distribution of numeric and categorical variables in the original dataset
categorical = ['SpecialDay', 'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend', 'Revenue']
df_numeric = df.drop(categorical, axis=1)
df_categorical = df[categorical]

print('\nDistribution of Numeric Variables')
for col_n in df_numeric.columns:
    plt.hist(df[col_n])
    plt.title(col_n + ' Distribution')
    plt.xlabel(col_n)
    plt.ylabel('No of Visitors')
    plt.show()
    
print('\nDistribution of Categorical Variables')
for col_c in ['SpecialDay', 'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType']:
    plt.hist(df[col_c])
    plt.title(col_c + ' Distribution')
    plt.xlabel(col_c)
    plt.ylabel('No of Visitors')
    plt.show()

for col_c in ['Weekend', 'Revenue']:
    df[col_c].value_counts().plot(kind='bar')
    plt.title(col_c + ' Distribution')
    plt.xlabel(col_c)
    plt.ylabel('No of Visitors')
    plt.show()


#calculate mean & stdevs and account for outlier values in the Duration columns (Administrative, Informational & ProductRelated)
outlier = pd.DataFrame()
stats = pd.DataFrame()
df2 = df[['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration']]

print('\nDescription of Duration Data')
print(df2.describe())

for i in df2.columns:
    stats.loc[0,i] = np.mean(df2[i]) - 3*np.std(df2[i]) #low limit
    stats.loc[1,i] = np.mean(df2[i]) + 3*np.std(df2[i]) #high limit   
    outlier[i] = ((df2[i] < stats.loc[0,i]) | (df2[i] > stats.loc[1,i]))
    stats.loc[2,i] = np.mean(df2[~outlier][i]) #mean of non-outliers
    stats.loc[3,i] = df2[i].min()
    stats.loc[4,i] = df2[i].max()
    
print('\nNo. of outliers before replacement:')
print(df[outlier].count())


#replace outliers with mean of non-outliers
df[df2.columns] = df2.where(~outlier, stats.loc[2,df2.columns], axis=1)

outlier2 = pd.DataFrame()
for j in df2.columns: 
    outlier2[j] = ((df[j] < stats.loc[0,j]) | (df[j] > stats.loc[1,j]))

print('\nNo. of outliers after replacement:')
print(df[outlier2].count())

print('\nDescription of Numeric Data after Replacement')
print(df.describe())


df['Total_Page_Viewed'] = df['Administrative'] + df['Informational'] + df['ProductRelated']
df['Total_Duration'] = df['Administrative_Duration'] + df['Informational_Duration'] + df['ProductRelated_Duration']


#normalize (Standard Scaler) Page Type, Duration columns (Administrative, Informational & ProductRelated), Rates(Bounce & Exit) & Page Values
df['Norm Administrative'] = StandardScaler().fit(pd.DataFrame(df['Administrative'])).transform(pd.DataFrame(df['Administrative']))
df['Norm Informational'] = StandardScaler().fit(pd.DataFrame(df['Informational'])).transform(pd.DataFrame(df['Informational']))
df['Norm ProductRelated'] = StandardScaler().fit(pd.DataFrame(df['ProductRelated'])).transform(pd.DataFrame(df['ProductRelated']))
df['Norm Total_Page_Viewed'] = StandardScaler().fit(pd.DataFrame(df['Total_Page_Viewed'])).transform(pd.DataFrame(df['Total_Page_Viewed']))
df['Norm Administrative_Duration'] = StandardScaler().fit(pd.DataFrame(df['Administrative_Duration'])).transform(pd.DataFrame(df['Administrative_Duration']))
df['Norm Informational_Duration'] = StandardScaler().fit(pd.DataFrame(df['Informational_Duration'])).transform(pd.DataFrame(df['Informational_Duration']))
df['Norm ProductRelated_Duration'] = StandardScaler().fit(pd.DataFrame(df['ProductRelated_Duration'])).transform(pd.DataFrame(df['ProductRelated_Duration']))
df['Norm Total_Duration'] = StandardScaler().fit(pd.DataFrame(df['Total_Duration'])).transform(pd.DataFrame(df['Total_Duration']))
df['Norm BounceRates'] = StandardScaler().fit(pd.DataFrame(df['BounceRates'])).transform(pd.DataFrame(df['BounceRates']))
df['Norm ExitRates'] = StandardScaler().fit(pd.DataFrame(df['ExitRates'])).transform(pd.DataFrame(df['ExitRates']))
df['Norm PageValues'] = StandardScaler().fit(pd.DataFrame(df['PageValues'])).transform(pd.DataFrame(df['PageValues']))
df['Norm SpecialDay'] = StandardScaler().fit(pd.DataFrame(df['SpecialDay'])).transform(pd.DataFrame(df['SpecialDay']))

#consolidate  Month
df.loc[df['Month']=='Jan','Month'] = 1
df.loc[df['Month']=='Feb','Month'] = 2
df.loc[df['Month']=='Mar','Month'] = 3
df.loc[df['Month']=='Apr','Month'] = 4
df.loc[df['Month']=='May','Month'] = 5
df.loc[df['Month']=='Jun','Month'] = 6
df.loc[df['Month']=='Jul','Month'] = 7
df.loc[df['Month']=='Aug','Month'] = 8
df.loc[df['Month']=='Sep','Month'] = 9
df.loc[df['Month']=='Oct','Month'] = 10
df.loc[df['Month']=='Nov','Month'] = 11
df.loc[df['Month']=='Dec','Month'] = 12

df.loc[df['Month']==1,'Quarter'] = 1
df.loc[df['Month']==2,'Quarter'] = 1
df.loc[df['Month']==3,'Quarter'] = 1
df.loc[df['Month']==4,'Quarter'] = 2
df.loc[df['Month']==5,'Quarter'] = 2
df.loc[df['Month']==6,'Quarter'] = 2
df.loc[df['Month']==7,'Quarter'] = 3
df.loc[df['Month']==8,'Quarter'] = 3
df.loc[df['Month']==9,'Quarter'] = 3
df.loc[df['Month']==10,'Quarter'] = 4
df.loc[df['Month']==11,'Quarter'] = 4
df.loc[df['Month']==12,'Quarter'] = 4


#bin Bounce & Exit Rates
nb_bounce = 4
binwidth_bounce = 0.05
minbin_bounce = 0
maxbin1_bounce = minbin_bounce + 1*binwidth_bounce
maxbin2_bounce = minbin_bounce + 2*binwidth_bounce
maxbin3_bounce = minbin_bounce + 3*binwidth_bounce
maxbin4_bounce = minbin_bounce + 4*binwidth_bounce

df.loc[((df['BounceRates']>=minbin_bounce) & (df['BounceRates']<=maxbin1_bounce)), 'BounceRates Bin'] = '0.000-0.050'
df.loc[((df['BounceRates']>maxbin1_bounce) & (df['BounceRates']<=maxbin2_bounce)), 'BounceRates Bin'] = '0.051-0.100'
df.loc[((df['BounceRates']>maxbin2_bounce) & (df['BounceRates']<=maxbin3_bounce)), 'BounceRates Bin'] = '0.101-0.150'
df.loc[((df['BounceRates']>maxbin3_bounce) & (df['BounceRates']<=maxbin4_bounce)), 'BounceRates Bin'] = '0.151-0.200'


nb_exit = 4
binwidth_exit = 0.05
minbin_exit = 0
maxbin1_exit = minbin_exit + 1*binwidth_exit
maxbin2_exit = minbin_exit + 2*binwidth_exit
maxbin3_exit = minbin_exit + 3*binwidth_exit
maxbin4_exit = minbin_exit + 4*binwidth_exit

df.loc[((df['ExitRates']>=minbin_exit) & (df['ExitRates']<=maxbin1_exit)), 'ExitRates Bin'] = '0.000-0.050'
df.loc[((df['ExitRates']>maxbin1_exit) & (df['ExitRates']<=maxbin2_exit)), 'ExitRates Bin'] = '0.051-0.100'
df.loc[((df['ExitRates']>maxbin2_exit) & (df['ExitRates']<=maxbin3_exit)), 'ExitRates Bin'] = '0.101-0.150'
df.loc[((df['ExitRates']>maxbin3_exit) & (df['ExitRates']<=maxbin4_exit)), 'ExitRates Bin'] = '0.151-0.200'


#one hot encode VisitorType
df.loc[df['VisitorType']=='Other', 'VisitorType'] = 'Other_Visitor'
df_ohe1 = pd.get_dummies(df['VisitorType'], prefix_sep="__", columns='VisitorType')
df = pd.concat([df, df_ohe1], axis=1)


#final dataset after preparation
print('\nFinal Dataset after preparation:')
print(df.head())
r, c = df.shape
print('\nDataset shape after preparation: ', r, ' observations and ', c, ' attributes\n')




### UNSUPERVISED LEARNING ###


#Cluster Analysis 1: cluster analysis on Norm Total Duration & Norm Total Page Viewed attributes (both numeric variables)
#Q1: Do visitors who spend more time on the website & viewing pages with higher values have a higher purchase rate (Revenue = True)?
print('\nQ1: Do visitors with high Total Duration & Page Viewed combination have a higher purchase rate (Revenue = True)?')

X1 = df[['Norm Total_Duration', 'Norm Total_Page_Viewed']]

kmeans1 = KMeans(n_clusters=4, random_state=0).fit(X1)
X1['Cluster Label'] =  kmeans1.labels_
df['Label1'] = X1['Cluster Label']
centroids1 = kmeans1.cluster_centers_

plt.scatter(X1['Norm Total_Duration'], X1['Norm Total_Page_Viewed'], c=kmeans1.labels_.astype(float), s=10, alpha=0.5)
plt.scatter(centroids1[:, 0], centroids1[:, 1], c='red', s=40)
plt.xlabel('Norm Total Duration')
plt.ylabel('Norm Total Page Viewed')
plt.title('Norm Total Duration v Norm Total Page Viewed')
plt.text(round(centroids1[0,0],2)+0.1, round(centroids1[0,1],2)+0.1, 'Cluster 0', fontweight='bold')
plt.text(round(centroids1[1,0],2)+0.1, round(centroids1[1,1],2)+0.1, 'Cluster 1', fontweight='bold')
plt.text(round(centroids1[2,0],2)+0.1, round(centroids1[2,1],2)+0.1, 'Cluster 2', fontweight='bold')
plt.text(round(centroids1[3,0],2)+0.1, round(centroids1[3,1],2)+0.1, 'Cluster 3', fontweight='bold')
plt.show()

print('\nCentroid coordinates:')
print('Label 0: x=', round(centroids1[0,0],2), ', y=', round(centroids1[0,1],2))
print('Label 1: x=', round(centroids1[1,0],2), ', y=', round(centroids1[1,1],2))
print('Label 2: x=', round(centroids1[2,0],2), ', y=', round(centroids1[2,1],2))
print('Label 3: x=', round(centroids1[3,0],2), ', y=', round(centroids1[3,1],2))

X1cluster0 = df.loc[(df['Label1'] == 0) & (df['Revenue']==True), 'Label1'].count() / df.loc[(df['Label1'] == 0), 'Label1'].count()
X1cluster1 = df.loc[(df['Label1'] == 1) & (df['Revenue']==True), 'Label1'].count() / df.loc[(df['Label1'] == 1), 'Label1'].count()
X1cluster2 = df.loc[(df['Label1'] == 2) & (df['Revenue']==True), 'Label1'].count() / df.loc[(df['Label1'] == 2), 'Label1'].count()
X1cluster3 = df.loc[(df['Label1'] == 3) & (df['Revenue']==True), 'Label1'].count() / df.loc[(df['Label1'] == 3), 'Label1'].count()
print('\nPurchase Rate:')
print('Cluster 0:', round(X1cluster0*100,1),'%')
print('Cluster 1:', round(X1cluster1*100,1),'%')
print('Cluster 2:', round(X1cluster2*100,1),'%')
print('Cluster 3:', round(X1cluster3*100,1),'%')
print('\nBased on our clustering analysis, visitors with high Total Duration & Page Viewed combination do not have a higher Purchase Rate than other clusters')
print('\nVisitors visiting high Total Page Viewed have tend to have a higher Purchase Rate regardless of Duration')


#Cluster Analysis 2: cluster analysis on Norm Product Related & Norm Informational attributes (both numeric variables)
#Q2: What type of pages do visitors with a higher Purchase Rate view?
print('\nQ2: What type of pages do visitors with a higher Purchase Rate view?')

X2 = df[['Norm ProductRelated', 'Norm Informational']]

kmeans2 = KMeans(n_clusters=4, random_state=0).fit(X2)
X2['Cluster Label'] =  kmeans2.labels_
df['Label2'] = X2['Cluster Label']
centroids2 = kmeans2.cluster_centers_

plt.scatter(X2['Norm ProductRelated'], X2['Norm Informational'], c=kmeans2.labels_.astype(float), s=20, alpha=0.5)
plt.scatter(centroids2[:, 0], centroids2[:, 1], c='red', s=40)
plt.xlabel('Norm ProductRelated')
plt.ylabel('Norm Informational')
plt.title('Norm ProductRelated v Norm Informational')
plt.text(round(centroids2[0,0],2)+0.1, round(centroids2[0,1],2)+0.1, 'Cluster 0', fontweight='bold')
plt.text(round(centroids2[1,0],2)+0.1, round(centroids2[1,1],2)+0.1, 'Cluster 1', fontweight='bold')
plt.text(round(centroids2[2,0],2)+0.1, round(centroids2[2,1],2)+0.1, 'Cluster 2', fontweight='bold')
plt.text(round(centroids2[3,0],2)+0.1, round(centroids2[3,1],2)+0.1, 'Cluster 3', fontweight='bold')
plt.show()

print('\nCentroid coordinates:')
print('Label 0: x=', round(centroids2[0,0],2), ', y=', round(centroids2[0,1],2))
print('Label 1: x=', round(centroids2[1,0],2), ', y=', round(centroids2[1,1],2))
print('Label 2: x=', round(centroids2[2,0],2), ', y=', round(centroids2[2,1],2))
print('Label 3: x=', round(centroids2[3,0],2), ', y=', round(centroids2[3,1],2))

X2cluster0 = df.loc[(df['Label2'] == 0) & (df['Revenue']==True), 'Label2'].count() / df.loc[(df['Label2'] == 0), 'Label2'].count()
X2cluster1 = df.loc[(df['Label2'] == 1) & (df['Revenue']==True), 'Label2'].count() / df.loc[(df['Label2'] == 1), 'Label2'].count()
X2cluster2 = df.loc[(df['Label2'] == 2) & (df['Revenue']==True), 'Label2'].count() / df.loc[(df['Label2'] == 2), 'Label2'].count()
X2cluster3 = df.loc[(df['Label2'] == 3) & (df['Revenue']==True), 'Label2'].count() / df.loc[(df['Label2'] == 3), 'Label2'].count()
print('\nPurchase Rate:')
print('Cluster 0:', round(X2cluster0*100,1),'%')
print('Cluster 1:', round(X2cluster1*100,1),'%')
print('Cluster 2:', round(X2cluster2*100,1),'%')
print('Cluster 3:', round(X2cluster3*100,1),'%')
print('\nBased on our clustering analysis, visitors who visit more Product Related page have a higher Purchase Rate than other clusters')


#Cluster Analysis 3: cluster analysis on Norm Special Day & Norm Total Duration attributes (categorical & numeric variables, respectively)
#Q3: Do visitors who visit the website around Special Days have a higher Purchase Rate?
print('\nQ3: Do visitors who visit the website around Special Days have a higher Purchase Rate?')

X3 = df[['Norm SpecialDay', 'Norm Total_Duration']]

kmeans3 = KMeans(n_clusters=2, random_state=0).fit(X3)
X3['Cluster Label'] =  kmeans3.labels_
df['Label3'] = X3['Cluster Label']
centroids3 = kmeans3.cluster_centers_

plt.scatter(X3['Norm SpecialDay'], X3['Norm Total_Duration'], c=kmeans3.labels_.astype(float), s=20, alpha=0.5)
plt.scatter(centroids3[:, 0], centroids3[:, 1], c='red', s=40)
plt.xlabel('Norm Special Day')
plt.ylabel('Norm Total Duration')
plt.title('Norm Special Day v Norm Total Duration')
plt.text(round(centroids3[0,0],2)+0.1, round(centroids3[0,1],2)+0.1, 'Cluster 0', fontweight='bold')
plt.text(round(centroids3[1,0],2)+0.1, round(centroids3[1,1],2)+0.1, 'Cluster 1', fontweight='bold')
plt.show()

print('\nCentroid coordinates:')
print('Label 0: x=', round(centroids3[0,0],2), ', y=', round(centroids3[0,1],2))
print('Label 1: x=', round(centroids3[1,0],2), ', y=', round(centroids3[1,1],2))

X3cluster0 = df.loc[(df['Label3'] == 0) & (df['Revenue']==True), 'Label3'].count() / df.loc[(df['Label3'] == 0), 'Label3'].count()
X3cluster1 = df.loc[(df['Label3'] == 1) & (df['Revenue']==True), 'Label3'].count() / df.loc[(df['Label3'] == 1), 'Label3'].count()
print('\nPurchase Rate:')
print('Cluster 0:', round(X3cluster0*100,1),'%')
print('Cluster 1:', round(X3cluster1*100,1),'%')
print('\nBased on our clustering analysis, visitors who visit the website around Special Days have a lower Purchase Rate than the other cluster')




### SUPERVISED LEARNING ###


#in this exercise, we're using RandomForest Classification & Logistic Regression techniques & calculate the prediction probability as well as accuracy score for each model

#Classification Analysis 1
#Q1: Is visitor 5487 (#1 in the test set) going to make a purchase (Revenue=True) based on what pages they view and other attributes?
print('\nQ1: Is visitor 5487 (#1 in the test set) going to make a purchase (Revenue=True) based on what pages they view and other attributes?')

        
#splitting data into train & test sets
#our target column is Revenue
X = df.drop(['Revenue', 'Label1', 'Label2', 'Label3', 'Quarter', 'BounceRates Bin', 'ExitRates Bin', 'OperatingSystems', 'Browser', 'VisitorType', 
             'Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 
             'Administrative', 'Informational', 'ProductRelated', 'SpecialDay'], axis=1)
y = df['Revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print('\nTraining set shape:', X_train.shape)
print('\nTesting set shape:', X_test.shape)


## RANDOM FOREST ##
print('\nRANDOM FOREST CLASSIFICATION')

estimator = 200
mss = 20
print ('\nRandom Forest classifier with', estimator, 'estimators and', mss, 'min sample split')
print('\n')

rfc = RandomForestClassifier(n_estimators=estimator, min_samples_split=mss)

#training the model using train dataset
trained_rfc = rfc.fit(X_train, y_train)

#applying the model using test dataset
y_pred_rfc = rfc.predict(X_test)

y_pred_rfc_df = pd.DataFrame(y_pred_rfc).set_index(y_test.index)
y_pred_rfc_df.columns = ['Revenue_pred_rfc']

pred_proba_rfc = rfc.predict_proba(X_test)
pred_proba_rfc_df = pd.DataFrame(pred_proba_rfc).set_index(y_test.index)
pred_proba_rfc_df.columns = ['RanFor Prob 0', 'RanFor Prob 1']

results_df = pd.concat([y_test, y_pred_rfc_df], axis=1)
print(results_df)

print('\nBased on our Random Forest Classification model, visitor #5487 (#1 in test set) will not make a purchase. This is the same as the original test set result')
      
#calculating Accuracy, Confusion Matrix & Metrics for Random Forest result
print('\nAccuracy Score, Confusion Matrix & ROC/AUC for Random Forest Model')  

score_rfc = round(metrics.accuracy_score(y_test, y_pred_rfc),3)
print('\nRandom Forest accuracy score:', score_rfc)

Threshold = 0.5 # Some number between 0 and 1
print('\nProbability Threshold is chosen to be:', Threshold)

probabilities_rfc = pred_proba_rfc_df['RanFor Prob 1']
predictions_rfc = (probabilities_rfc > Threshold).astype(int)

CM_rfc = confusion_matrix(y_test, predictions_rfc)

tn_rfc, fp_rfc, fn_rfc, tp_rfc = CM_rfc.ravel()
print('\nTP, TN, FP, FN: ', tp_rfc, ",", tn_rfc, ",", fp_rfc, ",", fn_rfc)

AR_rfc = accuracy_score(y_test, predictions_rfc)
print('\nAccuracy rate: ', np.round(AR_rfc, 2))

P_rfc = precision_score(y_test, predictions_rfc)
print('\nPrecision: ', np.round(P_rfc, 2))

R_rfc = recall_score(y_test, predictions_rfc)
print('\nRecall: ', np.round(R_rfc, 2))

F1_rfc = f1_score(y_test, predictions_rfc)
print ("\nF1 score:", np.round(F1_rfc, 2))

#False Positive Rate, True Posisive Rate, probability thresholds for Random Forest
fpr_rfc, tpr_rfc, th_rfc = roc_curve(y_test, probabilities_rfc)
AUC_rfc = auc(fpr_rfc, tpr_rfc)

#plotting the ROC & AUC for Random Forest
plt.rcParams["figure.figsize"] = [8, 8] # Square
font = {'family' : 'normal', 'weight' : 'bold', 'size' : 18}
matplotlib.rc('font', **font)
plt.figure()
plt.title('ROC Curve: Random Forest')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(fpr_rfc, tpr_rfc, LW=3, label='ROC curve (AUC = %0.2f)' % AUC_rfc)
plt.plot([0, 1], [0, 1], color='navy', LW=3, linestyle='--') # reference line for random classifier
plt.legend(loc="lower right")
plt.show()

#our Random Forest model has an accuracy score around 91%, which is fairly accurate.
#our precision, recall rates and F1 score are 72%, 59% and 65%, respectively, which are fairly low
#our AUC is 0.93 which is very high
#we can improve this by adjusting a few parameters used in the model as well as modifying the dataset (simplifying dataset, changing some parameters, omiting certain attributes, etc.)



## LOGISTIC REGRESSION ##
print('\nLOGISTIC REGRESSION CLASSIFICATION')
print('\n')

logreg = LogisticRegression()

#training the model using train dataset
trained_logreg = logreg.fit(X_train, y_train) 

#applying the model using test dataset
y_pred_logreg = logreg.predict(X_test)

y_pred_logreg_df = pd.DataFrame(y_pred_logreg).set_index(y_test.index)
y_pred_logreg_df.columns = ['Revenue_pred_logreg']

pred_proba_logreg = logreg.predict_proba(X_test)
pred_proba_logreg_df = pd.DataFrame(pred_proba_logreg).set_index(y_test.index)
pred_proba_logreg_df.columns = ['LogReg Prob 0', 'LogReg Prob 1']

results_df = pd.concat([results_df, y_pred_logreg_df], axis=1)
print(results_df)

#calculating Accuracy, Confusion Matrix & Metrics for Logistic Regression result
print('\nAccuracy Score, Confusion Matrix & ROC/AUC for Logistic Regression Model')  

score_logreg = round(metrics.accuracy_score(y_test, y_pred_logreg),3)
print('\nLogistic Regression accuracy score:', score_logreg)

Threshold = 0.5 # Some number between 0 and 1
print('\nProbability Threshold is chosen to be:', Threshold)

probabilities_logreg = pred_proba_logreg_df['LogReg Prob 1']
predictions_logreg = (probabilities_logreg > Threshold).astype(int)

CM_logreg = confusion_matrix(y_test, predictions_logreg)

tn_logreg, fp_logreg, fn_logreg, tp_logreg = CM_logreg.ravel()
print('\nTP, TN, FP, FN: ', tp_logreg, ",", tn_logreg, ",", fp_logreg, ",", fn_logreg)

AR_logreg = accuracy_score(y_test, predictions_logreg)
print('\nAccuracy rate: ', np.round(AR_logreg, 2))

P_logreg = precision_score(y_test, predictions_logreg)
print('\nPrecision: ', np.round(P_logreg, 2))

R_logreg = recall_score(y_test, predictions_logreg)
print('\nRecall: ', np.round(R_logreg, 2))

F1_logreg = f1_score(y_test, predictions_logreg)
print ("\nF1 score:", np.round(F1_logreg, 2))

#False Positive Rate, True Posisive Rate, probability thresholds for Random Forest
fpr_logreg, tpr_logreg, th_logreg = roc_curve(y_test, probabilities_logreg)
AUC_logreg = auc(fpr_logreg, tpr_logreg)

#plotting the ROC & AUC for Random Forest
plt.rcParams["figure.figsize"] = [8, 8] # Square
font = {'family' : 'normal', 'weight' : 'bold', 'size' : 18}
matplotlib.rc('font', **font)
plt.figure()
plt.title('ROC Curve: Logistic Regression')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.plot(fpr_logreg, tpr_logreg, LW=3, label='ROC curve (AUC = %0.2f)' % AUC_logreg)
plt.plot([0, 1], [0, 1], color='navy', LW=3, linestyle='--') # reference line for random classifier
plt.legend(loc="lower right")
plt.show()

#our Logistic Regression model has an accuracy score around 89%, which is fairly accurate.
#our precision, recall rates and F1 score are 71%, 37% and 49%, respectively, which are fairly low
#our AUC is 0.90
#we can improve this by adjusting a few parameters used in the model as well as modifying the dataset (simplifying dataset, changing some parameters, omiting certain attributes, etc.)
