# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1

2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.

3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.

4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.

The feature selection techniques used are:

1.Filter Method

2.Wrapper Method

3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd 
from scipy import stats 
import numpy as np
df=pd.read_csv("bmi.csv") 
df.head()
```
<img width="297" height="202" alt="image" src="https://github.com/user-attachments/assets/8514c978-aabe-46c3-a6ba-5ad4a9880887" />

```
df_null_sum=df.isnull().sum() 
df_null_sum
```
<img width="145" height="102" alt="image" src="https://github.com/user-attachments/assets/002a69e4-f50e-43f1-833a-20c4b8e0affa" />

```
df.dropna()
```
<img width="327" height="465" alt="image" src="https://github.com/user-attachments/assets/20246665-d9ae-4a2a-befa-4ad6d0819737" />

```
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0) 
max_vals
```
<img width="131" height="71" alt="image" src="https://github.com/user-attachments/assets/8e878604-ac74-4375-b818-b7069d61f4ff" />

```
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("bmi.csv")
df1.head()
```
<img width="282" height="206" alt="image" src="https://github.com/user-attachments/assets/1d19915c-b5fe-40ca-9bdd-155320ce3495" />
```
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']]) 
df1.head(10)
```
<img width="341" height="376" alt="image" src="https://github.com/user-attachments/assets/10ddf11f-5ff2-4c31-ac8b-7afe036be817" />

```
from sklearn.preprocessing import MinMaxScaler 
scaler=MinMaxScaler() 
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']]) 
df.head(10)
```
<img width="316" height="381" alt="image" src="https://github.com/user-attachments/assets/52c949b7-9a4b-471c-b7d2-f7c764f7e1a5" />

```
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']]) 
print(df)
```
<img width="352" height="267" alt="image" src="https://github.com/user-attachments/assets/e6aa8601-1384-4191-a2d8-d56cc75caf11" />

```
from sklearn.preprocessing import RobustScaler 
scaler = RobustScaler() 
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']]) 
df3.head()
```
<img width="311" height="232" alt="image" src="https://github.com/user-attachments/assets/c8776bb6-52d3-4979-b1cc-1184a49f6a88" />

```
df=pd.read_csv("income(1) (1).csv") 
df.info()
```
<img width="387" height="400" alt="image" src="https://github.com/user-attachments/assets/a6bafbd9-f951-424f-85e1-280b5c699d30" />
df_null_sum=df.isnull().sum() 
df_null_sum
<img width="196" height="277" alt="image" src="https://github.com/user-attachments/assets/226447a2-2671-4204-966e-524ae1b25292" />
```
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
<img width="803" height="660" alt="image" src="https://github.com/user-attachments/assets/33185446-4752-4de5-b902-ced9817b2d00" />

```
df[categorical_columns] = df[categorical_columns].astype('category') 
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```

<img width="730" height="468" alt="image" src="https://github.com/user-attachments/assets/031206bd-0516-41a7-a4ee-81a6c3e6d786" />

```
X = df.drop(columns=['SalStat']) 
y = df['SalStat']
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 
rf = RandomForestClassifier(n_estimators=100, random_state=42) 
rf.fit(X_train, y_train)
```
<img width="381" height="101" alt="image" src="https://github.com/user-attachments/assets/8b6b9781-36f8-403c-8f1b-4dd8d89ca2bb" />

```
y_pred = rf.predict(X_test) 
df=pd.read_csv("income(1) (1).csv") 
df.info()
```
<img width="425" height="388" alt="image" src="https://github.com/user-attachments/assets/9f5ea442-909a-4044-aa38-1aa105678e20" />
```
import pandas as pd 
from sklearn.feature_selection import SelectKBest, chi2, f_classif 
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry'] 
df[categorical_columns] = df[categorical_columns].astype('category') 
df[categorical_columns]
```
<img width="811" height="658" alt="image" src="https://github.com/user-attachments/assets/055a0cb1-29ee-4f3e-97ca-95852c2c2b87" />

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes) 
df[categorical_columns]
```
<img width="747" height="477" alt="image" src="https://github.com/user-attachments/assets/9a54d34f-bbcf-48e4-a346-04a44a3ed93d" />
```
X = df.drop(columns=['SalStat']) 
y = df['SalStat'] 
k_chi2 = 6 
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2) 
X_chi2 = selector_chi2.fit_transform(X, y) 
selected_features_chi2 = X.columns[selector_chi2.get_support()] 
print("Selected features using chi-square test:") 
print(selected_features_chi2)
```
<img width="638" height="86" alt="image" src="https://github.com/user-attachments/assets/4d909237-c9f3-412f-bd57-ee034dd3e3df" />

```
import pandas as pd 
from sklearn.feature_selection import SelectKBest, chi2, f_classif 
from sklearn.model_selection import train_test_split # Importing the missing function 
from sklearn.ensemble import RandomForestClassifier 
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss', 'hoursperweek'] 
X = df[selected_features] 
y = df['SalStat'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 
rf = RandomForestClassifier(n_estimators=100, random_state=42) 
rf.fit(X_train, y_train)
```
<img width="356" height="102" alt="image" src="https://github.com/user-attachments/assets/72834d52-ae05-445c-8238-50972216a3c3" />

```
y_pred = rf.predict(X_test) 
from sklearn.metrics import accuracy_score 
accuracy = accuracy_score(y_test, y_pred) 
print(f"Model accuracy using selected features: {accuracy}")
```
<img width="518" height="36" alt="image" src="https://github.com/user-attachments/assets/b76b3253-253d-4adb-b06f-907964a0ac8d" />
```
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
categorical_columns = [ 'JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry' ]
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
<img width="735" height="477" alt="image" src="https://github.com/user-attachments/assets/230b2651-7e30-4489-85b4-dd01f366a7cb" />

```
X = df.drop(columns=['SalStat']) 
y = df['SalStat']
k_anova = 5
selector_anova = SelectKBest(score_func=f_classif,k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nSelected features using ANOVA:")
print(selected_features_anova)
```
<img width="746" height="56" alt="image" src="https://github.com/user-attachments/assets/fcf65036-9315-4dde-99e2-d9e54e78042c" />
```
import pandas as pd 
from sklearn.feature_selection import RFE 
from sklearn.linear_model import LogisticRegression 
df=pd.read_csv("income(1) (1).csv")
categorical_columns = [ 'JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry' ]
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
<img width="747" height="462" alt="image" src="https://github.com/user-attachments/assets/6db55d6f-e892-4fb8-aed0-3efc836236b6" />
```
X = df.drop(columns=['SalStat']) 
y = df['SalStat']
logreg = LogisticRegression()
n_features_to_select =6
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select) 
rfe.fit(X, y)
```
<img width="350" height="178" alt="image" src="https://github.com/user-attachments/assets/81bbbf69-45ac-4f10-823d-48bc2b4ac860" />

## RESULT:

Thus is the program is executed successfully





























