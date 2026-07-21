import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

np.random.seed(42)
df = pd.read_csv('../data/processed/harmonized_raw.csv')

feature_cols = ['age', 'bp', 'glucose', 'bmi', 'hemoglobin', 'creatinine']
label_map = {'Healthy': 0, 'Diabetes': 1, 'CKD': 2}
df['y'] = df['label'].map(label_map)

# Indicator-augmented missingness (3rd tier of original imputation strategy, honestly applied
# since >25% missing for bp/hemoglobin/creatinine here)
for col in feature_cols:
    miss_rate = df[col].isnull().mean()
    if miss_rate > 0.25:
        df[f'{col}_missing'] = df[col].isnull().astype(int)

# Train/test split BEFORE imputation to avoid leakage; stratified by label and source
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['y'])

imp_cols = feature_cols
imputer = SimpleImputer(strategy='median')
train_df = train_df.copy()
test_df = test_df.copy()
train_df[imp_cols] = imputer.fit_transform(train_df[imp_cols])
test_df[imp_cols] = imputer.transform(test_df[imp_cols])

flag_cols = [c for c in train_df.columns if c.endswith('_missing')]
X_cols = imp_cols + flag_cols

train_df.to_csv('../data/processed/train.csv', index=False)
test_df.to_csv('../data/processed/test.csv', index=False)

print("Feature columns used:", X_cols)
print("Train shape:", train_df.shape, "Test shape:", test_df.shape)
print("\nTrain label dist:\n", train_df['label'].value_counts())
print("\nTest label dist:\n", test_df['label'].value_counts())
print("\nSource distribution in train:\n", train_df['source'].value_counts())
