import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import math
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('modified.csv')

def convert_to_numeric(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            labels, uniques = pd.factorize(df[col])
            df[col] = labels + 1
    return df.dropna()

df = convert_to_numeric(df)

def handle_outliers(df):
    z_scores = (df - df.mean()) / df.std()
    outliers = (z_scores > 3) | (z_scores < -3)
    df[outliers] = np.nan
    df.fillna(df.median(), inplace=True)
    return df

df = handle_outliers(df)

def random_forest_model(data):
    data['Reason'] = data['Reason'].astype(str).str.split(';')
    exploded_data = data.explode('Reason')
    exploded_data['Reason'] = exploded_data['Reason'].str.strip()
    exploded_data['Reason'], uniques = pd.factorize(exploded_data['Reason'])
    reason_stats = exploded_data.groupby('Reason').agg({'CitationCount': 'sum', 'Duration': ['count', 'mean']}).reset_index()
    reason_stats.columns = ['Reason', 'Citation', 'Frequency', 'Mean_Duration']
    reason_stats_encoded = pd.get_dummies(reason_stats, columns=['Reason'], drop_first=True)
    y = reason_stats['Reason'].values
    X = reason_stats_encoded.drop(columns=[col for col in reason_stats_encoded.columns if 'Reason_' in col]).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    y_base = np.full(y_test.shape, np.mean(y_train))
    basline_mae = metrics.mean_absolute_error(y_test, y_base)
    basline_mse = metrics.mean_squared_error(y_test, y_base)
    basline_rmse = math.sqrt(basline_mse)
    print("---Baseline---")
    print("MAE: %.2f " % basline_mae)
    print("MSE: %.2f " % basline_mse)
    print("RMSE: %.2f " % basline_rmse)
    rf = RandomForestRegressor(n_estimators=300, max_depth=5, n_jobs=-1, random_state=18)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rfg_mae = metrics.mean_absolute_error(y_test, y_pred)
    rfg_mse = metrics.mean_squared_error(y_test, y_pred)
    rfg_rmse = math.sqrt(metrics.mean_squared_error(y_test, y_pred))
    print("---Random Forest Regressor---")
    print("MAE: %.2f " % rfg_mae)
    print("MSE: %.2f " % rfg_mse)
    print("RMSE: %.2f " % rfg_rmse)
    feature_importances = rf.feature_importances_
    feature_names = reason_stats_encoded.drop(columns=[col for col in reason_stats_encoded.columns if 'Reason_' in col]).columns
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    print("---Feature Importances---")
    print(feature_importance_df)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt'],
        'bootstrap': [True, False]
    }
    rf_regressor = RandomForestRegressor(random_state=18)
    grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)
    best_rf_regressor = RandomForestRegressor(**best_params, random_state=18)
    best_rf_regressor.fit(X_train, y_train)
    y_pred_tuned = best_rf_regressor.predict(X_test)
    mae_tuned = metrics.mean_absolute_error(y_test, y_pred_tuned)
    mse_tuned = metrics.mean_squared_error(y_test, y_pred_tuned)
    rmse_tuned = math.sqrt(mse_tuned)
    print("---Baseline---")
    print("MAE: %.2f " % basline_mae)
    print("MSE: %.2f " % basline_mse)
    print("RMSE: %.2f " % basline_rmse)
    print("---Random Forest Regressor---")
    print("MAE: %.2f " % rfg_mae)
    print("MSE: %.2f " % rfg_mse)
    print("RMSE: %.2f " % rfg_rmse)
    print("---Tuned Random Forest Regressor---")
    print("MAE: %.2f " % mae_tuned)
    print("MSE: %.2f " % mse_tuned)
    print("RMSE: %.2f " % rmse_tuned)
    feature_importances_tuned = best_rf_regressor.feature_importances_
    feature_importance_df_tuned = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances_tuned
    }).sort_values(by='Importance', ascending=False)
    print("---Feature Importances before---")
    print(feature_importance_df)
    print("---Feature Importances (Tuned Model)---")
    print(feature_importance_df_tuned)

random_forest_model(df)
