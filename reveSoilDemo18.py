import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import joblib  # For saving and loading models

# Load Transformed Dataset
df = pd.read_csv("transformed_soildataset.csv")
df.columns = df.columns.str.strip()

# Identify feature and target columns
wavelength_cols = [col for col in df.columns if col.replace(".", "").isdigit()]
sensor_features = ["Capacitity Moist", "Temp", "Moist", "EC (u/10 gram)", "Ph", "Nitro (mg/10 g)", "Posh Nitro (mg/10 g)", "Pota Nitro (mg/10 g)"]

df_numeric = df[wavelength_cols + sensor_features].apply(pd.to_numeric, errors='coerce')

# Standardizing wavelength data
scaler = StandardScaler()
wavelength_data = df_numeric[wavelength_cols]
wavelength_scaled = scaler.fit_transform(wavelength_data.fillna(0))

# Feature Selection using VIF
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

X_vif = pd.DataFrame(wavelength_scaled, columns=wavelength_cols)
vif_scores = calculate_vif(X_vif)
vif_filtered_cols = X_vif.columns[vif_scores['VIF'] < 10]  # Keep features with VIF < 10
X_filtered = X_vif[vif_filtered_cols]

# Ensure at least one feature is selected
if X_filtered.shape[1] == 0:
    print("Warning: No features selected after VIF filtering. Using original wavelength data.")
    X_filtered = X_vif

# Compute correlation matrix for wavelength vs sensor features
corr_df = pd.DataFrame(
    {wavelength: df_numeric[wavelength].corr(df_numeric[sensor]) for wavelength in wavelength_cols} 
    for sensor in sensor_features if sensor in df_numeric.columns
).T
corr_df.index = wavelength_cols  # Wavelengths on x-axis
corr_df.columns = [sensor for sensor in sensor_features if sensor in df_numeric.columns]  # Sensor features on y-axis

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_df.T, annot=True, cmap="coolwarm", linewidths=0.5)
plt.xlabel("Wavelength Values")
plt.ylabel("Sensor Features")
plt.title("Correlation Heatmap: Wavelength vs Sensor Features")
plt.show()

# Train and evaluate models to predict sensor data from wavelength data
best_models = {}  # Dictionary to store best models for each feature

# Ensure model save directory exists
os.makedirs("models", exist_ok=True)

for feature in sensor_features:
    if feature not in df_numeric.columns:
        continue
    
    Y = df_numeric[[feature]].dropna()
    valid_rows = df_numeric[wavelength_cols].notna().all(axis=1) & Y.notna().all(axis=1)
    X_valid, Y_valid = X_filtered.loc[valid_rows], Y.loc[valid_rows]
    
    if X_valid.shape[0] == 0:
        print(f"Skipping {feature} due to insufficient valid data.")
        continue
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_valid, Y_valid, test_size=0.2, random_state=42)
    
    # Fix missing values and ensure correct types
    X_train = X_train.fillna(0).astype(np.float64)
    Y_train = Y_train.fillna(0).astype(np.float64)
    X_test = X_test.fillna(0).astype(np.float64)
    Y_test = Y_test.fillna(0).astype(np.float64)
    
    print(f"Training {feature}: X_train shape {X_train.shape}, Y_train shape {Y_train.shape}")
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "SVR": SVR()
    }
    
    best_r2, best_model, best_model_name = -np.inf, None, ""
    for name, model in models.items():
        model.fit(X_train, Y_train.values.ravel())
        
        Y_pred = model.predict(X_test)
        r2 = r2_score(Y_test, Y_pred)
        mae = mean_absolute_error(Y_test, Y_pred)
        print(f"{name} for {feature} - R²: {r2:.4f}, MAE: {mae:.4f}")
        
        if r2 > best_r2:
            best_r2, best_model, best_model_name = r2, model, name
    
    print(f"Best Model for {feature}: {best_model_name}\n")
    
    # Save the best model
    best_models[feature] = best_model
    sanitized_feature = feature.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    joblib.dump(best_model, f"models/best_model_{sanitized_feature}.pkl")
    
    # Plot actual vs predicted values
    Y_pred_best = best_model.predict(X_test)
    plt.figure(figsize=(6, 4))
    plt.scatter(Y_test, Y_pred_best, alpha=0.5, label='Predicted vs Actual')
    plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', label='Ideal Fit')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Actual vs Predicted for {feature}")
    plt.legend()
    plt.show()

# Function to predict sensor values from new spectrometer data
def predict_sensor_values(new_spectrometer_data_list):
    new_spectrometer_data = pd.DataFrame([new_spectrometer_data_list], columns=wavelength_cols[:len(new_spectrometer_data_list)])  # Ensure correct column names
    new_spectrometer_scaled = scaler.transform(new_spectrometer_data.fillna(0))  # Standardize data
    
    # Convert back to DataFrame with column names after transformation
    new_spectrometer_scaled_df = pd.DataFrame(new_spectrometer_scaled, columns=wavelength_cols[:new_spectrometer_scaled.shape[1]])
    
    predictions = {feature: float(model.predict(new_spectrometer_scaled_df)[0]) for feature, model in best_models.items()}
    
    print("\nPredicted Sensor Values:")
    print("=================================")
    for key, value in predictions.items():
        print(f"{key:<25}: {value:.4f}")
    print("=================================")
    
    return predictions


# Example: Predicting with new spectrometer data (list format)
# print("Enter The SPectrometer Readings : ")
new_spectrometer_data_list = eval(input("Enter The SPectrometer Readings : "))  # Adjust length dynamically
predictions = predict_sensor_values(new_spectrometer_data_list)
