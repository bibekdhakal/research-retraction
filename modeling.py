import pandas as pd
from sklearn import datasets, metrics, preprocessing, svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from statistics import mean
import math
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import GridSearchCV


# df = pd.read_csv('../transformed_retractions_after_EDA.csv')
# df = pd.read_csv('../ready_for_modelling_reason.csv')
df = pd.read_csv('../modified.csv')
# print(df.columns)
# df.drop(columns=['RetractionDate', 'OriginalPaperDate', 'RetractionNature',], inplace=True)

def convert_to_numeric(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            labels, uniques = pd.factorize(df[col])
            df[col] = labels + 1  # Adding 1 to start labels from 1 instead of 0
    return df.dropna()



df = convert_to_numeric(df)

# Detect and handle outliers
def handle_outliers(df):
    # Identify outliers using z-score or IQR
    z_scores = (df - df.mean()) / df.std()
    outliers = (z_scores > 3) | (z_scores < -3)
    
    # Replace outliers with NaN
    df[outliers] = np.nan
    
    # Impute missing values
    df.fillna(df.median(), inplace=True)  # Impute with median or other suitable method
    
    return df

# Preprocess the data (including outlier handling)
df = handle_outliers(df)
# Determine the position of the 'Reason' column
# reason_col_index = df.columns.get_loc('Reason')

# Move the 'Reason' column to the last position
# df = pd.concat([df.iloc[:, :reason_col_index], df.iloc[:, reason_col_index+1:], df.iloc[:, reason_col_index]], axis=1)
# print(df)
# feature_cols = list(df.columns[:-1])
# target_col = df.columns[-1]
# features = df[feature_cols]
# target = df[target_col]


# def SVM_model(features, target):
#     # splitting data into 50% training and 50% test
#     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.5, random_state=0)

#     # SVM with no standardisation
#     clf = SVC(kernel='linear').fit(X_train, y_train)
#     y_hat = clf.predict(X_test)
#     acc = metrics.accuracy_score(y_test, y_hat)

#     # create SVM with standardisation via standardisation --> ≠classification pipeline
#     pipe = make_pipeline(StandardScaler(), SVC(kernel="linear"))
#     pipe.fit(X_train, y_train)
#     acc_std = pipe.score(X_test, y_test)

#     # compare accuracy scores
#     print("Accuracy comparison:")
#     print("SVM --> %.3f%%" % (acc * 100))
#     print("SVM with standardisation --> %.3f%%" % (acc_std * 100))

def SVM_kfolds(df):
    # B. Prepate the dataset
    X = df.drop(columns=['Reason'])  # Features
    y = df['Reason']  # Target
    # articleType = df['ArticleType'].unique()
    # print(articleType)
    # print(X)
    # print(y)
    # C. Prepare Cross Validator and the scoring schemes
    scoring = ['accuracy', 'recall_macro', 'precision_macro', 'f1_macro']

    # D. Build a cross validation pipeline
    pipe = make_pipeline(preprocessing.StandardScaler(), svm.SVC(kernel='linear'))

    # E. Evaluate the model
    scores = cross_validate(pipe, X, y, cv=10, scoring=scoring)

    # accuracy
    print("Mean accuracy: %.3f%%" % (mean(scores['test_accuracy'])*100))

    # precision
    print("Mean precision: %.3f " % (mean(scores['test_precision_macro'])))

    # recall
    print("Mean recall: %.3f" % (mean(scores['test_recall_macro'])))

    # # F1 (F-Measure)
    print("Mean F1: %.3f" % (mean(scores['test_f1_macro'])))


# SVM_model(features, target)
# SVM_kfolds(df)



def random_forest_model(data):
   # Ensure 'Reason' column is of string type and split it into individual reasons
    # data['Reason'] = data['Reason'].astype(str).str.split(';')

    # # Explode the 'Reason' column to have one reason per row
    # exploded_data = data.explode('Reason')

    # # Remove leading and trailing whitespace from 'Reason'
    # exploded_data['Reason'] = exploded_data['Reason'].str.strip()

    # # Group by Reason and aggregate
    # reason_stats = exploded_data.groupby('Reason').agg({'CitationCount': 'sum', 'Duration': ['count', 'mean']}).reset_index()
    # # print("Resons", reason_stats)
    # # Flatten the multi-level columns
    # reason_stats.columns = ['Reason', 'Citation', 'Frequency', 'Mean_Duration']

    # # One-hot encode the reasons
    # reason_stats_encoded = pd.get_dummies(reason_stats, columns=['Reason'], drop_first=True)

    # # Get the response variable as ndarray
    # y = reason_stats_encoded["Mean_Duration"].values

    # # Drop the response variable so what remains are the explanatory variables
    # X = reason_stats_encoded.drop(columns=["Mean_Duration", "Frequency", "Citation"]).values

    # # Split data into training and test sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

    # # Build, apply and evaluate the baseline predictor using the existing avg. retraction time values
    # # y_base = X_test[:, reason_stats_encoded.columns.get_loc("Mean_Duration") - 3]  # Adjust index after dropping columns : Initial
    # y_base = np.full(y_test.shape, np.mean(y_train))


    # mae = metrics.mean_absolute_error(y_test, y_base)
    # mse = metrics.mean_squared_error(y_test, y_base)
    # rmse = math.sqrt(mse)

    # print("---Baseline---")
    # print("MAE: %.2f " % mae)
    # print("MSE: %.2f " % mse)
    # print("RMSE: %.2f " % rmse)

    # # Build, apply and evaluate RF Regressor 
    # rf = RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1, random_state=0)
    # rf.fit(X_train, y_train)
    # y_pred = rf.predict(X_test)

    # mae = metrics.mean_absolute_error(y_test, y_pred)
    # mse = metrics.mean_squared_error(y_test, y_pred)
    # rmse = math.sqrt(metrics.mean_squared_error(y_test, y_pred))

    # print("---Random Forest Regressor---")
    # print("MAE: %.2f " % mae)
    # print("MSE: %.2f " % mse)
    # print("RMSE: %.2f " % rmse)

    # # Evaluate feature importance
    # feature_importances = rf.feature_importances_
    # feature_names = reason_stats_encoded.drop(columns=["Mean_Duration", "Frequency", "Citation"]).columns
    # feature_importance_df = pd.DataFrame({
    #     'Feature': feature_names,
    #     'Importance': feature_importances
    # }).sort_values(by='Importance', ascending=False)

    # print("---Feature Importances---")
    # print(feature_importance_df)

        # Ensure 'Reason' column is of string type and split it into individual reasons
    data['Reason'] = data['Reason'].astype(str).str.split(';')
    # Explode the 'Reason' column to have one reason per row
    exploded_data = data.explode('Reason')
    # Remove leading and trailing whitespace from 'Reason'
    exploded_data['Reason'] = exploded_data['Reason'].str.strip()

    # Factorize the 'Reason' column to get numeric labels
    exploded_data['Reason'], uniques = pd.factorize(exploded_data['Reason'])

    # Group by Reason and aggregate
    reason_stats = exploded_data.groupby('Reason').agg({'CitationCount': 'sum', 'Duration': ['count', 'mean']}).reset_index()
    reason_stats.columns = ['Reason', 'Citation', 'Frequency', 'Mean_Duration']
    # print("Reason stats:\n", reason_stats)

    # One-hot encode the reasons
    reason_stats_encoded = pd.get_dummies(reason_stats, columns=['Reason'], drop_first=True)
    # print("Encoded Reason stats:\n", reason_stats_encoded.head())

    # Extract features and target
    y = reason_stats['Reason'].values
    X = reason_stats_encoded.drop(columns=[col for col in reason_stats_encoded.columns if 'Reason_' in col]).values
    # print("X shape:", X.shape)
    # print("y shape:", y.shape)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Baseline predictor using the existing avg. retraction time values
    y_base = np.full(y_test.shape, np.mean(y_train))
    basline_mae = metrics.mean_absolute_error(y_test, y_base)
    basline_mse = metrics.mean_squared_error(y_test, y_base)
    basline_rmse = math.sqrt(basline_mae)
    

    print("---Baseline---")
    print("MAE: %.2f " % basline_mae)
    print("MSE: %.2f " % basline_mse)
    print("RMSE: %.2f " % basline_rmse)

    # Random Forest Regressor
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

    # Evaluate feature importance
    feature_importances = rf.feature_importances_
    feature_names = reason_stats_encoded.drop(columns=[col for col in reason_stats_encoded.columns if 'Reason_' in col]).columns
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    print("---Feature Importances---")
    print(feature_importance_df)


    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt'],
        'bootstrap': [True, False]
    }

    # Instantiate the Random Forest Regressor
    rf_regressor = RandomForestRegressor(random_state=18)

    # Instantiate GridSearchCV
    grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)

    # Perform Grid Search Cross-Validation
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    # Train the Random Forest Regressor with the best parameters
    best_rf_regressor = RandomForestRegressor(**best_params, random_state=18)
    best_rf_regressor.fit(X_train, y_train)

    # Make predictions on the test data using the tuned model
    y_pred_tuned = best_rf_regressor.predict(X_test)

    # Evaluate the tuned model's performance
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

    # Evaluate feature importance of the tuned model
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
