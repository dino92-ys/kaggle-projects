import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# 데이터 로드
train = pd.read_csv('/Users/19mac/Library/Mobile Documents/com~apple~CloudDocs/Study/AI/Kaggle/kaggle-projects/house-price/Data/train.csv')

# 타겟 분리
target = train['SalePrice']
train_data = train.drop(['SalePrice', 'Id'], axis=1)

# 숫자형 특징만 선택 (간단한 전처리)
numeric_features = train_data.select_dtypes(include=['int64', 'float64']).columns
X = train_data[numeric_features]

# 결측치 중앙값으로 채우기
X = X.fillna(X.median())

# Train/Validation 분리
X_train, X_val, y_train, y_val = train_test_split(X, target, test_size=0.2, random_state=42)

# XGBoost 모델 학습
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

print("모델 학습 중...")
model.fit(X_train, y_train)

# 검증 세트 예측
y_pred = model.predict(X_val)

# 평가 지표 계산
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

# 로그 변환 RMSE (Kaggle 평가 방식)
log_rmse = np.sqrt(mean_squared_error(np.log(y_val), np.log(y_pred)))

print("\n=== House Price 검증 점수 ===")
print(f"RMSE: ${rmse:,.2f}")
print(f"MAE: ${mae:,.2f}")
print(f"R² Score: {r2:.4f}")
print(f"Log RMSE (Kaggle 방식): {log_rmse:.5f}")
