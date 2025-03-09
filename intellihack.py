# %% [markdown]
# # INTELLIHACK 5.0 Task - 01

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
from sklearn.model_selection import train_test_split

# %% [markdown]
# ## Exploratory Data Analysis

# %%
data = pd.read_csv('weather_data.csv')
data

# %%
data['date'] = pd.to_datetime(data['date'])

# %%
data['day'] = data['date'].dt.day
data['month'] = data['date'].dt.month

# %%
rain_or_not_map = {'Rain': 1, 'No Rain': 0}
data['rain_or_not'] = data['rain_or_not'].map(rain_or_not_map)
data

# %%
missing_values = data.isnull().sum()
missing_values_by_rain = data.groupby('rain_or_not').apply(lambda x: x.isnull().sum())
missing_values_by_rain

# %% [markdown]
# There are 6 `No Rain` missing values and 9 `Rain` missing values. Because of the limited no. of datapoints the na values will be filled 

# %%
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), cmap='coolwarm', annot=True)
plt.show()

# %% [markdown]
# We can observe some high co-linearity between avg_temperatures and humidity. But there are no high co-relations between the target variable and features.

# %% [markdown]
# We can the use the relationship between `pressure` to other features fill the missing values instead of forward filling or using other means like `mean` or `median`. When going through the dataset we can identify there are instances of `Rain` after `No Rain` and vice versa because of the feature.

# %%
from sklearn.linear_model import LinearRegression

cols_to_fill = ['avg_temperature', 'humidity', 'avg_wind_speed', 'cloud_cover']

for col in cols_to_fill:
    known = data[data[col].notna()]
    missing = data[data[col].isna()]

    model = LinearRegression()
    model.fit(known[['pressure']], known[col])

    data.loc[data[col].isna(), col] = model.predict(missing[['pressure']])

# %%
data

# %%
missing_values = data.isnull().sum()

# %%
sns.pairplot(data, vars=["avg_temperature", "humidity", "avg_wind_speed", "cloud_cover", "pressure"], hue="rain_or_not")
plt.show()

# %%
numeric_cols = ['avg_temperature', 'humidity', 'avg_wind_speed', 'cloud_cover', 'pressure']

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    sns.boxplot(y=data[col], ax=axes[i], hue=data['rain_or_not'])
    axes[i].set_title(col)

# Hide any unused subplots if needed
for j in range(len(numeric_cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# %% [markdown]
# Data is mostly clustured.

# %%
data['rain_or_not'].value_counts()

# %% [markdown]
# We can see a class imbalance

# %% [markdown]
# ## Model Training

# %%
X = data[['day', 'month', 'avg_temperature', 'humidity', 'avg_wind_speed', 'cloud_cover', 'pressure']]
y = data['rain_or_not']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# %%
# from imblearn.over_sampling import SMOTE

# sm = SMOTE(random_state=42)
# X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# X_train = X_train_res
# y_train = y_train_res

# %%
from imblearn.under_sampling import RandomUnderSampler

undersampler = RandomUnderSampler(random_state=42)
X_train_res, y_train_res = undersampler.fit_resample(X_train, y_train)

X_train = X_train_res
y_train = y_train_res

# %%
y_train.value_counts()

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# %%
# scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

# %%
models = [
    ("LogisticRegression", LogisticRegression()),
    ("DecisionTree", DecisionTreeClassifier()),
    ("RandomForest", RandomForestClassifier()),
    ("GradientBoosting", GradientBoostingClassifier()),
    ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
    ("LightGBM", LGBMClassifier())
]

base_results = []
for name, model in models:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    clr = classification_report(y_test, preds)
    base_results.append((name, clr))

best_base_model, best_base_score = max(base_results, key=lambda x: x[1])

# %%
for name, report in base_results:
    print(name)
    print(report)
    print('-' * 50)

# %%
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'num_leaves': [31, 50, 100],
    'boosting_type': ['gbdt', 'dart'],
    'max_depth': [-1, 10, 20],
    'min_child_samples': [20, 30, 40]
}

lgbm = LGBMClassifier()
grid_search = GridSearchCV(estimator=lgbm, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_lgbm_model = grid_search.best_estimator_
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation accuracy: ", grid_search.best_score_)

best_lgbm_preds = best_lgbm_model.predict(X_test)


# %%
print(classification_report(y_test, best_lgbm_preds))

# %% [markdown]
# ## Deep Learning

# %%
import tensorflow as tf

# %%
X_train_tensor = tf.convert_to_tensor(X_train.values, dtype=tf.float32)
y_train_tensor = tf.convert_to_tensor(y_train.values, dtype=tf.float32)

X_test_tensor = tf.convert_to_tensor(X_test.values, dtype=tf.float32)
y_test_tensor = tf.convert_to_tensor(y_test.values, dtype=tf.float32)

# %%
dl_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

dl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
dl_model.fit(X_train_tensor, y_train_tensor, epochs=10, batch_size=16, class_weight={0: 1, 1: 5})

loss, accuracy = dl_model.evaluate(X_test_tensor, y_test_tensor)
print("Test Accuracy:", accuracy)

# %%
data.tail()

# %% [markdown]
# ## Final Predictions

# %%
from statsmodels.tsa.statespace.sarimax import SARIMAX

forecast_days = 21
forecasts = {}

for feature in numeric_cols:
    model = SARIMAX(data[feature], order=(7, 1, 1), seasonal_order=(5, 1, 1, 12))
    model_fit = model.fit()
    forecast_values = model_fit.forecast(steps=forecast_days)
    forecasts[feature] = forecast_values.values

forecast_dates = pd.date_range(start=data['date'].max() + pd.Timedelta(days=1), periods=forecast_days)
forecast_df = pd.DataFrame(forecasts, index=forecast_dates)
forecast_df.index.name = 'date'

for feature in numeric_cols:
    plt.figure(figsize=(15,4))
    plt.plot(data['date'], data[feature], label='Historical')
    plt.plot(forecast_df.index, forecast_df[feature], label='Forecast')
    plt.title(f'{feature} Forecast')
    plt.xlabel('Date')
    plt.ylabel(feature)
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

forecast_df

# %%
forecast_df.reset_index(inplace=True)

# %%
def preprocess_and_predict_rain(forecast_df, model):
    forecast_df["day"] = forecast_df["date"].dt.day
    forecast_df["month"] = forecast_df["date"].dt.month

    X_forecast = forecast_df[["day", "month", "avg_temperature", "humidity", "avg_wind_speed", "cloud_cover", "pressure"]]

    X_forecast_tensor = tf.convert_to_tensor(X_forecast.values, dtype=tf.float32)

    predictions = model.predict(X_forecast_tensor).flatten()
    forecast_df["rain_or_not"] = (predictions >= 0.5).astype(int)

    return forecast_df

# %%
forecast_df_predicted = preprocess_and_predict_rain(forecast_df, dl_model)
forecast_df_predicted

# %%
forecast_df_predicted.to_csv('forecasted_weather.csv', index=False)