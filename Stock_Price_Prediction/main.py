import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('TSLA.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.head()
df.shape
df.describe()
df.info

# plt.figure(figsize=(15,5))
# plt.plot(df['Close'])
# plt.title('Tesla Stock Price', fontsize=15)
# plt.ylabel('Price in US Dollars')
# plt.show()

#print(df[df['Close'] == df['Adj Close']])
df = df.drop(['Adj Close'], axis=1) #Drop the column 'Adj Close'
#print(df.isnull().sum()) #Print total number of null value in each column

features = ['Open', 'High', 'Low', 'Close', 'Volume']
# plt.subplots(figsize = (20,10))
# for i, col in enumerate(features):
#     plt.subplot(2,3,i+1)
#     sb.displot(df[col])
# plt.show()

# plt.subplots(figsize = (20,10))
# for i, col in enumerate(features):
#     plt.subplot(2,3,i+1)
#     sb.boxplot(df[col])
# plt.show()


df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year


df['is_quarter_end'] = np.where(df['month']%3==0,1,0)
print(df.head())

# data_grouped = df.groupby('year').mean()
# plt.subplots(figsize=(40,20))
# for i,col in enumerate(['Open','High','Low','Close']):
#     plt.subplot(2,2,i+1)
#     data_grouped[col].plot.bar()
#     plt.title(f'Mean {col} over years')
#     plt.xlabel('Year')
#     plt.ylabel(f'Mean {col}')
#plt.show()

df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'],1,0)

# plt.pie(df['target'].value_counts().values,
#         labels=[0,1], autopct='%1.1f%%')
#plt.show()

#plt.figure(figsize=(10,10))
#sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)
#plt.show()

#machine learning with train_test_split of StandardScaler, random_state is the "pseudo number of random cases" in the test
features = df[['open-close','low-high','is_quarter_end']]
target = df['target']
scaler = StandardScaler()
features = scaler.fit_transform(features)
X_train, X_valid, Y_train, Y_valid = train_test_split(features, target, test_size=0.1, random_state=2022)
print(X_train.shape, X_valid.shape)

#machine learning with Model Development and Evaluation (logistic regression, Support vector Machine, XGBClassifier)
models = [LogisticRegression(), SVC(kernel='poly', probability=True), XGBClassifier()]
for i in range(3):
    models[i].fit(X_train, Y_train)
    print(f'{models[i]}: ')
    print('Training Accuracy: ', metrics.roc_auc_score(Y_train, models[i].predict_proba(X_train)[:,1]))
    print('Validation Accuracy: ', metrics.roc_auc_score(Y_valid, models[i].predict_proba(X_valid)[:,1]))
    print()

print(models[0])
y_pred = models[0].predict(X_valid)
cm = confusion_matrix(Y_valid, y_pred)

plt.figure(figsize=(8,6))
sb.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['True 0', 'True 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()