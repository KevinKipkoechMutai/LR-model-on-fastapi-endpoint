import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib

salary_data = pd.read_csv("data/ds_salaries.csv")

encoder = OrdinalEncoder()
salary_data['company_size_encoded'] = encoder.fit_transform(salary_data[['company_size']])

salary_data = pd.get_dummies(salary_data, columns=['employment_type', 'job_title'], drop_first=True, dtype=int)

salary_data = salary_data.drop(columns=['experience_level', 'company_size'])

#dependent and independent features
X = salary_data.drop(columns='salary_in_usd')
y = salary_data['salary_in_usd']

#train-test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=104, test_size=0.2, shuffle=True)

#LR model fitting
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

#make predictions
y_pred = regr.predict(X_test)

print("Coefficients: \n", regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

print("R2: %.2f" % r2_score(y_test, y_pred))

#save model
joblib.dump(regr, 'lin_regress.sav')