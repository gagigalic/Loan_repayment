import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_info = pd.read_csv("lending_club_info.csv", index_col="LoanStatNew")

def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])

feat_info("mort_acc")

df = pd.read_csv("lending_club_loan_two.csv")

sns.countplot(x="loan_status", data = df)
plt.savefig("loan_status.png")
plt.close()

plt.figure(figsize=(12,4))
sns.histplot(df["loan_amnt"], kde = False, bins = 40)
plt.savefig("loan_amnt.png")
plt.close()

plt.figure(figsize=(12,7))
sns.heatmap(df.corr(), annot = True, cmap = "viridis")
plt.ylim(10,0)
plt.savefig("heatmap.png")
plt.close()

#almost perfect correlation with the "installment"
feat_info("installment")
feat_info("loan_amnt")

sns.scatterplot(x = "installment", y="loan_amnt", data = df)
plt.savefig("plot.png")
plt.close()

#Boxplot showing the relationship between the loan_status and the Loan Amount
sns.boxplot(x = "loan_status", y = "loan_amnt", data = df)
plt.savefig("plot2.png")
plt.close()

#summary statistics for the loan amount, grouped by the loan_status
describe = df.groupby("loan_status")["loan_amnt"].describe()
print(describe)

print(df["grade"].unique())
print(df["sub_grade"].unique())

feat_info("sub_grade")

#Create a countplot per grade, hue to the loan_status label
sns.countplot(x="grade", data = df, hue = "loan_status")
plt.savefig("grade.png")
plt.close()

#reorrdered grade, countplot per sub grade
plt.figure(figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x="sub_grade", data = df, order=subgrade_order,
              palette = "coolwarm", hue= "loan_status")
plt.savefig("sub_grade.png")
plt.close()


#It looks like F and G subgrades don't get paid back that often

f_and_g = df[(df['grade']=='G') | (df['grade']=='F')]
plt.figure(figsize=(12,4))
subgrade_order = sorted(f_and_g['sub_grade'].unique())
sns.countplot(x='sub_grade',data=f_and_g,order = subgrade_order,hue='loan_status')
plt.savefig("plot4.png")
plt.close()

#Create a new column called 'load_repaid' .
df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0})
print(df[['loan_repaid','loan_status']])

#bar plot showing the correlation of the numeric features to the new loan_repaid column
df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')
plt.savefig("plot5.png")
plt.close()

#length of the dataframe
length = len(df)

#Series that displays the total count of missing values per column.
missing = df.isnull().sum()
print(missing)

#convert this Series to be in term of percentage of the total DataFrame
total = 100 * missing/len(df)
print(total)
#mort_acc 9.543469

feat_info("emp_title")
#The job title supplied by the Borrower when applying for the loan

#How many unique employment job titles are there?
df["emp_title"].nunique()
df['emp_title'].value_counts()
df.drop('emp_title',axis=1,inplace=True)

emp_length_order = [ '< 1 year',
                      '1 year',
                     '2 years',
                     '3 years',
                     '4 years',
                     '5 years',
                     '6 years',
                     '7 years',
                     '8 years',
                     '9 years',
                     '10+ years']

plt.figure(figsize=(12,4))
sns.countplot(x='emp_length',data=df,order=emp_length_order, hue = "loan_status")
plt.savefig("plot6.png")
plt.close()

emp_co = df[df['loan_status']=="Charged Off"].groupby("emp_length").count()['loan_status']
emp_fp = df[df['loan_status']=="Fully Paid"].groupby("emp_length").count()['loan_status']
emp_len = emp_co/(emp_co + emp_fp)

emp_len.plot(kind='bar')
plt.savefig("plot7.png")
plt.close()

df.drop('emp_length',axis=1,inplace=True)

#Revisit the DataFrame to see what feature columns still have missing data
print(df.isnull().sum())

df['purpose'].head(10)
df['title'].head(10)
df.drop('title',axis=1,inplace=True)

feat_info('mort_acc')
df['mort_acc'].value_counts()

print("Correlation with the mort_acc column")
df.corr()['mort_acc'].sort_values()
#total_acc 0.38 coor

print("Mean of mort_acc column per total_acc")
total_acc_avg = df.groupby('total_acc').mean()['mort_acc']


def fill_mort_acc(total_acc, mort_acc):

    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc

df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)

print(df.isnull().sum())
df.dropna(inplace=True)
print(df.isnull().sum())

#List all the columns that are currently non-numeric
df.select_dtypes(['object']).columns

#term feature
feat_info("term")
df['term'].value_counts()
df['term'] = df['term'].apply(lambda term: int(term[:3]))

#grade feature
df.drop('grade',axis=1, inplace=True)

#subgrade
subgrade_dummies = pd.get_dummies(df['sub_grade'],drop_first=True)
df = pd.concat([df.drop('sub_grade',axis=1),subgrade_dummies],axis=1)

print(df.columns)
print(df.select_dtypes(['object']).columns)

#verification_status, application_type,initial_list_status,purpose
dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose' ]],drop_first=True)
df = df.drop(['verification_status', 'application_type','initial_list_status','purpose'],axis=1)
df = pd.concat([df,dummies],axis=1)

#home_ownership
df['home_ownership'].value_counts()
df['home_ownership']=df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')

dummies = pd.get_dummies(df['home_ownership'],drop_first=True)
df = df.drop('home_ownership',axis=1)
df = pd.concat([df,dummies],axis=1)

#adress
df['zip_code'] = df['address'].apply(lambda address:address[-5:])

dummies = pd.get_dummies(df['zip_code'],drop_first=True)
df = df.drop(['zip_code','address'],axis=1)
df = pd.concat([df,dummies],axis=1)

#issue_d
df.drop('issue_d',axis=1, inplace=True)

#earliest_cr_line
df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda date:int(date[-4:]))
df = df.drop('earliest_cr_line',axis=1)

print(df.select_dtypes(['object']).columns)

#Train Test Split
from sklearn.model_selection import train_test_split

df = df.drop("loan_status", axis = 1)

X = df.drop("loan_repaid", axis = 1  ).values
y = df["loan_repaid"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.constraints import max_norm

model = Sequential()

X_train.shape #316175,78

model.add(Dense(78, activation = "relu"))
model.add(Dropout(0.2))

model.add(Dense(39, activation = "relu"))
model.add(Dropout(0.2))

model.add(Dense(19, activation = "relu"))
model.add(Dropout(0.2))

model.add(Dense(units=1, activation = "sigmoid"))

model.compile(loss = "binary_crossentropy", optimizer = "adam")

model.fit(x=X_train,
          y=y_train,
          epochs=25,
          batch_size=256,
          validation_data=(X_test, y_test),
          )

losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()
plt.savefig("plot8.png")
plt.close()

from sklearn.metrics import classification_report,confusion_matrix

predictions = model.predict(X_test)
binary_predictions = (predictions > 0.5).astype(int)

print(classification_report(y_test,binary_predictions))
print(confusion_matrix(y_test,binary_predictions))




