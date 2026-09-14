import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
df = pd.read_csv('forestfires.csv')
print(df.head())
df.info()
print(df.describe())
numeric_columns = list(df.select_dtypes([np.number]).columns)
X = df[numeric_columns].copy()
X.drop(columns=['area'], inplace=True)
y = np.log1p(df['area'])
#5% валідацію
X, X_holdout, y, y_holdout = train_test_split(X, y, test_size=0.05, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
linreg = LinearRegression()
linreg.fit(X_train_scaled, y_train)
score_scaled = linreg.score(X_test_scaled, y_test)
print("Testing results")
print(f"R**2 Score: {score_scaled}")
y_pred = linreg.predict(X_test_scaled)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse}")


