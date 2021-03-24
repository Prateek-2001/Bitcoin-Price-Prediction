#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# %%
df = pd.read_csv("data.csv")
# %%
df.head()
# %%
df.info()
# %%
df.isnull()

# %%
df.isnull().any()
# %%
df.isnull().sum()
# %%
nullval = df.isnull().any()
# %%
print(nullval)
# %%
nullvalue = []
for i in nullval:
    nullvalue.append(i)
# %%
nullvalues = {}
j = 0
for i in df:
    nullvalues[i] = str(nullvalue[j])
    j+=1
print(nullvalues)
# %%
for i in range(24):
    print(nullvalues[i])
    
# %%
for j in nullvalues:
    if nullvalues[j] == "True":
        df[j].fillna(int(df[j].mean()), inplace=True)


# %%
df.isnull().any()
# %%
df.btc_market_price[1023]
# %%
df.describe()
# %%
dates = []
for i in df.Date:
    i = i.split("/")
    newdate = int(i[0])*1000000+int(i[1])*10000+int(i[2])
    dates.append(newdate)
print(dates)
# %%
df.Date = dates
# %%
sns.jointplot(x="Date", y="btc_market_price", data=df, color="r", kind="scatter", sizes=(20,0))# %%

# %%
pearsoncorr = df.corr(method='pearson')
print(pearsoncorr)
# %%
pearsoncorr.head()

# %%
x = df.drop(['Date', 'btc_market_price'], axis =1)
y = df.btc_market_price
# %%
y = y.values
y.reshape(-1, 1)
# %%
from sklearn.model_selection import train_test_split
# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=9)

# %%
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

# %%
lr.fit(x_train, y_train)
# %%
pred = lr.predict(x_test)
# %%
from sklearn import metrics
# %%
print('Mean Squared Error:', metrics.mean_squared_error(y_test, pred))

# %%
