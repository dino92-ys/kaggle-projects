#!/usr/bin/env python3
"""
House Price 프로젝트의 주요 시각화 생성 스크립트
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 경로 설정
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'Data'
IMAGES_DIR = BASE_DIR / 'images'
IMAGES_DIR.mkdir(exist_ok=True)

# 데이터 로드
train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# 1. 상관관계 히트맵 (수치형 변수만)
print("\n1. 상관관계 히트맵 생성 중...")
plt.figure(figsize=(16, 12))
numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = train[numeric_cols].corr()

# SalePrice와의 상관관계가 높은 상위 15개 변수 선택
top_corr = corr_matrix['SalePrice'].abs().sort_values(ascending=False).head(16).index
top_corr_matrix = train[top_corr].corr()

sns.heatmap(top_corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('주요 변수들의 상관관계 히트맵 (SalePrice 기준)', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig(IMAGES_DIR / '상관관계히트맵.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   저장: {IMAGES_DIR / '상관관계히트맵.png'}")

# 2. SalePrice와 주요 변수들의 관계 (상위 6개)
print("\n2. 주요 변수와 SalePrice 관계 시각화 중...")
top_features = corr_matrix['SalePrice'].abs().sort_values(ascending=False).iloc[1:7]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (feature, corr_value) in enumerate(top_features.items()):
    ax = axes[idx]
    ax.scatter(train[feature], train['SalePrice'], alpha=0.5, s=10)
    ax.set_xlabel(feature, fontsize=11)
    ax.set_ylabel('SalePrice', fontsize=11)
    ax.set_title(f'{feature} vs SalePrice\n(상관계수: {corr_value:.3f})', fontsize=12)

    # 추세선 추가
    z = np.polyfit(train[feature].fillna(0), train['SalePrice'], 1)
    p = np.poly1d(z)
    ax.plot(train[feature], p(train[feature].fillna(0)), "r--", alpha=0.8, linewidth=2)

plt.suptitle('주요 변수와 판매 가격의 관계', fontsize=16, y=1.00)
plt.tight_layout()
plt.savefig(IMAGES_DIR / '주요변수관계.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   저장: {IMAGES_DIR / '주요변수관계.png'}")

# 3. SalePrice 분포
print("\n3. SalePrice 분포 시각화 중...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 원본 분포
axes[0].hist(train['SalePrice'], bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('SalePrice', fontsize=12)
axes[0].set_ylabel('빈도', fontsize=12)
axes[0].set_title('판매 가격 분포 (원본)', fontsize=14)
axes[0].axvline(train['SalePrice'].mean(), color='red', linestyle='--',
                linewidth=2, label=f'평균: ${train["SalePrice"].mean():,.0f}')
axes[0].axvline(train['SalePrice'].median(), color='green', linestyle='--',
                linewidth=2, label=f'중앙값: ${train["SalePrice"].median():,.0f}')
axes[0].legend()

# 로그 변환 분포
log_price = np.log1p(train['SalePrice'])
axes[1].hist(log_price, bins=50, edgecolor='black', alpha=0.7, color='orange')
axes[1].set_xlabel('log(SalePrice)', fontsize=12)
axes[1].set_ylabel('빈도', fontsize=12)
axes[1].set_title('판매 가격 분포 (로그 변환)', fontsize=14)
axes[1].axvline(log_price.mean(), color='red', linestyle='--',
                linewidth=2, label=f'평균: {log_price.mean():.2f}')
axes[1].axvline(log_price.median(), color='green', linestyle='--',
                linewidth=2, label=f'중앙값: {log_price.median():.2f}')
axes[1].legend()

plt.suptitle('판매 가격 분포 분석', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(IMAGES_DIR / '가격분포.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   저장: {IMAGES_DIR / '가격분포.png'}")

# 4. 범주형 변수와 SalePrice 관계 (주요 변수)
print("\n4. 범주형 변수와 SalePrice 관계 시각화 중...")
categorical_features = ['OverallQual', 'Neighborhood', 'GarageCars', 'FullBath']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, feature in enumerate(categorical_features):
    ax = axes[idx]

    # 각 카테고리별 평균 SalePrice 계산
    avg_price = train.groupby(feature)['SalePrice'].mean().sort_values(ascending=False)

    # 막대 그래프
    avg_price.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
    ax.set_xlabel(feature, fontsize=12)
    ax.set_ylabel('평균 판매 가격 ($)', fontsize=12)
    ax.set_title(f'{feature}별 평균 판매 가격', fontsize=13)
    ax.tick_params(axis='x', rotation=45)

    # 값 표시
    for i, v in enumerate(avg_price):
        ax.text(i, v, f'${v:,.0f}', ha='center', va='bottom', fontsize=9)

plt.suptitle('범주형 변수별 평균 판매 가격', fontsize=16, y=1.00)
plt.tight_layout()
plt.savefig(IMAGES_DIR / '범주형변수관계.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   저장: {IMAGES_DIR / '범주형변수관계.png'}")

# 5. 결측치 비율 시각화
print("\n5. 결측치 분석 시각화 중...")
missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
missing_percent = (missing / len(train)) * 100

fig, ax = plt.subplots(figsize=(12, 8))
missing_percent.plot(kind='barh', ax=ax, color='coral', edgecolor='black')
ax.set_xlabel('결측치 비율 (%)', fontsize=12)
ax.set_ylabel('변수명', fontsize=12)
ax.set_title('변수별 결측치 비율 (상위 항목)', fontsize=14)

# 값 표시
for i, v in enumerate(missing_percent):
    ax.text(v, i, f' {v:.1f}%', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(IMAGES_DIR / '결측치분석.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   저장: {IMAGES_DIR / '결측치분석.png'}")

# 6. 주요 통계 정보 출력
print("\n" + "="*60)
print("주요 통계 정보")
print("="*60)
print(f"\n판매 가격 통계:")
print(f"  평균: ${train['SalePrice'].mean():,.0f}")
print(f"  중앙값: ${train['SalePrice'].median():,.0f}")
print(f"  최소: ${train['SalePrice'].min():,.0f}")
print(f"  최대: ${train['SalePrice'].max():,.0f}")
print(f"  표준편차: ${train['SalePrice'].std():,.0f}")

print(f"\nSalePrice와 상관관계가 높은 상위 10개 변수:")
top_10_corr = corr_matrix['SalePrice'].abs().sort_values(ascending=False).iloc[1:11]
for i, (feature, corr_val) in enumerate(top_10_corr.items(), 1):
    print(f"  {i:2d}. {feature:20s}: {corr_val:.4f}")

print("\n시각화 생성 완료!")
print(f"모든 이미지가 {IMAGES_DIR}/ 폴더에 저장되었습니다.")
