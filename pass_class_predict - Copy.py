import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Placement_Info.csv')

data.drop(['S_no'],axis = 1, inplace = True)

#data.shape

c1={}
c2={}
c3={}
c4={}
comp={}

def placed_class (row):
    if row['Package']>=11.5:
        c4[row['Company_Placed']]=row['Package']
        return 'class4'
   
    if row['Package']>=9 and row['Package']<11.5:
        c3[row['Company_Placed']]=row['Package']
        return 'class3'
        
    if row['Package']>=5 and row['Package']<9:
        c2[row['Company_Placed']]=row['Package']
        return 'class2'
        
    if row['Package']>=0 and row['Package']<5:
        c1[row['Company_Placed']]=row['Package']
        return 'class1'

def gender (row):
    if row['Gender']=='Male':
        return 0
    else:
        return 1

def intern (row):
    if row['Internship_Experience']=='No':
        return 0
    else:
        return 10

data['Company_Placed']=data.apply(lambda row: placed_class(row), axis=1)

data['Internship_Experience']=data.apply(lambda row: intern(row), axis=1)

data['Gender']=data.apply(lambda row: gender(row), axis=1)

for i in range(3):
    c3['Micron (ETD)']=8.1
    c3['HD Works']=12.0
    c3['Barclays']=12.75
    c3['Accolite']=8.0

def getList(dict):
    return list(dict.keys())

c_c1=getList(c1)
c_c2=getList(c2)
c_c3=getList(c3)
c_c4=getList(c4)

#data.info()

#Correlation heatmap
plt.figure(figsize=(30, 15))
plt.title('Correlation between features')
sns.heatmap(data.corr(), annot = True)

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

corr_features = correlation(data, 0.51)


x=data[['Gender','CGPA','marks_12','Extra_Curricular','Data_Structures', 'Internship_Experience', 'Python','Speaking','Java','C','CPP','DBMS','OOPS','SQL','Operating_Systems','Data_Communication_and_Computer_Networks','Machine_Learning','Deep_Learning','Mobile_App_Development','Web_App_Development']]
y=data[['Company_Placed']]

d={}
for i in data['Company_Placed']:
    if i in d:
        d[i]+=1
    else:
        d[i]=1

company={'Cognizant GenC': 4.0, 'Wipro': 3.6, 'Accenture (ASE)': 4.5, 'Capgemini (Analyst)': 4.0, 'Infosys (SE)': 4.0, 'TCS': 3.6, 'Mindtree': 4.0,'PwC Acceleration Center': 6.0, 'Accolite': 8.0, 'Micron (ETD)': 8.1, 'Modak Analytics': 6.0, 'LTI (Level 2)': 6.5, 'Keka': 7.0, 'Infosys (DSE)': 5.0, 'LTI (Level 1)': 5.0, 'Accenture (Adv ASE)': 6.5, 'Capgemini (Sr.Analyst)': 7.5, 'Deloitte (Tax)': 7.6, 'Collins Aerospace (Internship)': 6.25, 'Cognizant GenC': 5.0, 'Advance Auto Parts': 9.0, 'Micron (EPS)': 9.1, 'Care Allianz': 11.0, 'Mathworks': 15.5, 'JP Morgan Chase': 14.0, 'Barclays': 12.75, 'Darwin Box (Internship)': 13.0, 'Oracle (GBU)': 16.6, 'HD Works': 12.0, 'Oracle (FSGBU)': 16.6}

import random
def predict_companies(ypred):
    if ypred[0]=='class4':
        return random.sample(c_c4,5)
    elif ypred[0]=='class3':
        return random.sample(c_c3,5)
    elif ypred[0]=='class2':
        return random.sample(c_c2,5)
    else:
        return random.sample(c_c1,5)

def predict_packages(pred):
    d={}
    for i in pred:
        d[i]=company[i]
    return d

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=10)

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

R_m =RandomForestClassifier(n_estimators = 50,criterion="entropy",min_samples_split=4,max_depth=15,max_samples=75,bootstrap=True)
R_model = AdaBoostClassifier(n_estimators=50,base_estimator=R_m,learning_rate=0.1)
R_model.fit(x_train,y_train.values.ravel())

y_pred=R_model.predict(x_test)

z=R_model.predict_proba(x_test)

print(R_model.score(x_test,y_test))
