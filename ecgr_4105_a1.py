import pandas as pd
from google.colab import drive
#drive.mount('/content/drive')

file_path = '/content/drive/My Drive/Colab Notebooks/Motor_Vehicle_Collisions.csv'

try:
  df = pd.read_csv(file_path)
  print(df.head())
except FileNotFoundError:
  print(f"Error: File not found at {file_path}")
except pd.errors.EmptyDataError:
  print(f"Error: The CSV file at {file_path} is empty.")
except pd.errors.ParserError:
  print(f"Error: Could not parse the CSV file at {file_path}. Check the file format.")
except Exception as e:
  print(f"An unexpected error occurred: {e}")

    #Isolate columns for CRASH_DATE, CRASH_TIME, PED_LOCATION, PED_ACTION, PERSON_TYPE, and PERSON_INJURY

selected_columns = ['CRASH_DATE', 'CRASH_TIME', 'PED_LOCATION', 'PED_ACTION', 'PERSON_TYPE', 'PERSON_INJURY']
try:
  new_df = df[selected_columns].copy() # Added .copy()
  print(new_df.head())
except KeyError as e:
  print(f"Error: Column(s) not found in the DataFrame: {e}")
except Exception as e:
  print(f"An unexpected error occurred: {e}")

    #Keep only Pedestrian values in PERSON_TYPE

try:
  pedestrian_df = new_df[new_df['PERSON_TYPE'] == 'Pedestrian'].copy() # Added .copy()
  print(pedestrian_df.head())
except KeyError as e:
  print(f"Error: Column 'PERSON_TYPE' not found in the DataFrame: {e}")
except Exception as e:
  print(f"An unexpected error occurred: {e}")

import datetime
# Convert 'CRASH_DATE' to datetime objects
pedestrian_df['CRASH_DATE'] = pd.to_datetime(pedestrian_df['CRASH_DATE'], errors='coerce')

# Create the 'CRASH_DAY' column
pedestrian_df['CRASH_DAY'] = pedestrian_df['CRASH_DATE'].dt.day_name()

print(pedestrian_df.head())

def get_daypart(time):
    """Categorizes a datetime object into a day part."""
    hour = time.hour
    if hour <= 6:
      return 'Early Morning'
    elif 6 <= hour < 10:
        return 'Morning Rush'
    elif 10 <= hour < 16:
        return 'Daytime'
    elif 16 <= hour < 19:
        return 'Evening Rush'
    else:
        return 'Nighttime'

# Create the 'CRASH_DAYPART' column using .loc
pedestrian_df.loc[:, 'CRASH_DAYPART'] = pd.to_datetime(pedestrian_df['CRASH_TIME'], format='%H:%M').apply(get_daypart)

print(pedestrian_df[['CRASH_TIME', 'CRASH_DAYPART']].head(20))

values_to_remove = ['Unknown', 'nan', 'Does Not Apply']
pedestrian_df = pedestrian_df[~pedestrian_df['PED_LOCATION'].isin(values_to_remove)]
pedestrian_df = pedestrian_df[~pedestrian_df['PED_LOCATION'].isna()]
pedestrian_df = pedestrian_df[~pedestrian_df['PED_ACTION'].isin(values_to_remove)]
pedestrian_df = pedestrian_df[~pedestrian_df['PED_ACTION'].isna()]
pedestrian_df = pedestrian_df[~pedestrian_df['PERSON_INJURY'].isin(values_to_remove)]

# Replace 'Injured' or 'Killed' with 'Injured', otherwise 0
pedestrian_df['PERSON_INJURY'] = pedestrian_df['PERSON_INJURY'].apply(lambda x: 0 if x == 'Unspecified' else x)
pedestrian_df['PERSON_INJURY'] = pedestrian_df['PERSON_INJURY'].apply(lambda x: 1 if x == 'Injured' else x)
pedestrian_df['PERSON_INJURY'] = pedestrian_df['PERSON_INJURY'].apply(lambda x: 2 if x == 'Killed' else x)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# Exploratory Data Analysis (EDA)

# Descriptive Statistics for Categorical Columns
print("\n--- Descriptive Statistics for Categorical Columns ---")
print(pedestrian_df['CRASH_DAY'].value_counts())
print(pedestrian_df['CRASH_DAYPART'].value_counts())
print(pedestrian_df['PED_LOCATION'].value_counts())
print(pedestrian_df['PED_ACTION'].value_counts())
print(pedestrian_df['PERSON_INJURY'].value_counts())

# Visualizations
print("\n--- Visualizations ---")

# Injury Severity by Day of Week
plt.figure(figsize=(10, 6))
sns.countplot(x='CRASH_DAY', hue='PERSON_INJURY', data=pedestrian_df,
 order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title('Injury Severity by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Number of Pedestrians')
plt.show()

# Injury Severity by Day Part
plt.figure(figsize=(10, 6))
sns.countplot(x='CRASH_DAYPART', hue='PERSON_INJURY', data=pedestrian_df,
 order=['Early Morning', 'Morning Rush', 'Daytime', 'Evening Rush', 'Nighttime'])
plt.title('Injury Severity by Day Part')
plt.xlabel('Day Part')
plt.ylabel('Number of Pedestrians')
plt.show()

# Pedestrian Location at time of Crash
plt.figure(figsize=(10, 6))
sns.countplot(x='PED_LOCATION', hue='PERSON_INJURY', data=pedestrian_df)
plt.title('Pedestrian Location at Time of Crash')
plt.xlabel('Pedestrian Location')
plt.ylabel('Number of Pedestrians')
plt.xticks(rotation=45, ha="right")
plt.show()

# Pedestrian Action at time of Crash
plt.figure(figsize=(10, 6))
sns.countplot(x='PED_ACTION', hue='PERSON_INJURY', data=pedestrian_df)
plt.title('Pedestrian Action at Time of Crash')
plt.xlabel('Pedestrian Action')
plt.ylabel('Number of Pedestrians')
plt.xticks(rotation=45, ha="right")
plt.show()

# Feature Engineering
# One-Hot Encoding and Data Preparation for Modeling
categorical_features = ['CRASH_DAY', 'CRASH_DAYPART', 'PED_LOCATION', 'PED_ACTION']

# Separate features (X) and target variable (y)
X = pedestrian_df.drop(['CRASH_DATE', 'CRASH_TIME', 'PERSON_TYPE', 'PERSON_INJURY'], axis=1)
y = pedestrian_df['PERSON_INJURY'].astype(int)  # Ensure y is integer type

# Column Transformer for One-Hot Encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) #Added stratify


# Model Training and Evaluation

def evaluate_model(model, X_train, y_train, X_test, y_test, cv=10):
    """
    Trains, evaluates, and performs cross-validation on a model.

    Args:
        model: The machine learning model.
        X_train: Training features.
        y_train: Training target.
        X_test: Testing features.
        y_test: Testing target.
        cv: Number of cross-validation folds.

    Returns:
        A dictionary containing model performance metrics.
    """

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)]) # Include the preprocessor

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1_weighted') # Use pipeline in CV

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'cv_mean_f1': np.mean(cv_scores),
        'cv_std_f1': np.std(cv_scores)
    }

# Initialize Models
logistic_regression = LogisticRegression(random_state=42, solver='liblinear')  # Removed multi_class
random_forest = RandomForestClassifier(random_state=42)
xgboost = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')  # Disable the deprecated warning


# Evaluate Models
print("\n--- Model Evaluation ---")
results = {}
results['Logistic Regression'] = evaluate_model(logistic_regression, X_train, y_train, X_test, y_test)
results['Random Forest'] = evaluate_model(random_forest, X_train, y_train, X_test, y_test)
results['XGBoost'] = evaluate_model(xgboost, X_train, y_train, X_test, y_test)

# Print Results
for model_name, metrics in results.items():
    print(f"\n--- {model_name} ---")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1']:.4f}")
    print(f"  CV Mean F1: {metrics['cv_mean_f1']:.4f}")
    print(f"  CV Std F1: {metrics['cv_std_f1']:.4f}")
    print("  Confusion Matrix:\n", metrics['confusion_matrix'])

import matplotlib.pyplot as plt
import seaborn as sns

# Print Results with Confusion Matrices
for model_name, metrics in results.items():
    print(f"\n--- {model_name} ---")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1']:.4f}")
    print(f"  CV Mean F1: {metrics['cv_mean_f1']:.4f}")
    print(f"  CV Std F1: {metrics['cv_std_f1']:.4f}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Uninjured', 'Injured', 'Killed'],
                yticklabels=['Uninjured', 'Injured', 'Killed'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Separate features (X) and target variable (y)
X = pedestrian_df.drop(['CRASH_DATE', 'CRASH_TIME', 'PERSON_TYPE', 'PERSON_INJURY'], axis=1)
y = pedestrian_df['PERSON_INJURY'].astype(int)

# One-hot encode categorical features (replace with your actual categorical features)
categorical_features = ['CRASH_DAY', 'CRASH_DAYPART', 'PED_LOCATION', 'PED_ACTION']
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier (you can use other models too)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Get feature importances
importances = rf_model.feature_importances_
feature_names = X.columns

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

importance_df

# Generate a larger synthetic dataset for testing
from sklearn.utils import resample

# Upsample the minority classes to balance the dataset (optional but recommended)
# This is crucial for models like RandomForest to perform optimally when dealing with class imbalances.
df_majority = pedestrian_df[pedestrian_df.PERSON_INJURY==1]
df_minority1 = pedestrian_df[pedestrian_df.PERSON_INJURY==0]
df_minority2 = pedestrian_df[pedestrian_df.PERSON_INJURY==2]


df_minority1_upsampled = resample(df_minority1,
                                 replace=True,     # sample with replacement
                                 n_samples=len(df_majority),    # to match majority class
                                 random_state=123) # reproducible results

df_minority2_upsampled = resample(df_minority2,
                                 replace=True,     # sample with replacement
                                 n_samples=len(df_majority),    # to match majority class
                                 random_state=123) # reproducible results


# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority1_upsampled, df_minority2_upsampled])

# Separate features and target variable
X_large = df_upsampled.drop(['CRASH_DATE', 'CRASH_TIME', 'PERSON_TYPE', 'PERSON_INJURY'], axis=1)
y_large = df_upsampled['PERSON_INJURY'].astype(int)


# One-hot encode the categorical features in the large dataset
X_large = pd.get_dummies(X_large, columns=categorical_features, drop_first=True)

# Align columns with the training data (Important for one-hot encoded features)
X_large = X_large.reindex(columns=X_train.columns, fill_value=0)

# Train the RandomForest model (or any other model) on the upsampled dataset
rf_model_large = RandomForestClassifier(random_state=42)
rf_model_large.fit(X_large, y_large)


# Evaluate the model on the test set
y_pred_large = rf_model_large.predict(X_test)  # Use the same X_test


# Evaluate the model's performance
print("Evaluation metrics on the large synthetic dataset test set:")
accuracy_large = accuracy_score(y_test, y_pred_large)
print(f'Accuracy: {accuracy_large}')

# Other evaluation metrics
precision_large = precision_score(y_test, y_pred_large, average='weighted', zero_division=0)
recall_large = recall_score(y_test, y_pred_large, average='weighted', zero_division=0)
f1_large = f1_score(y_test, y_pred_large, average='weighted', zero_division=0)


print(f"Precision: {precision_large}")
print(f"Recall: {recall_large}")
print(f"F1-Score: {f1_large}")

import matplotlib.pyplot as plt
# Calculate the correlation matrix
correlation_matrix = X.corr()

# Create the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.show()
