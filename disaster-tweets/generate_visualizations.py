#!/usr/bin/env python3
"""
Disaster Tweets 프로젝트의 주요 시각화 생성 스크립트
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import re

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 경로 설정
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
IMAGES_DIR = BASE_DIR / 'images'
IMAGES_DIR.mkdir(exist_ok=True)

# 데이터 로드
train = pd.read_csv(DATA_DIR / 'train.csv')

print(f"Train shape: {train.shape}")
print(f"Target distribution:\n{train['target'].value_counts()}")

# 1. 타겟 분포
print("\n1. 타겟 분포 시각화 중...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 카운트 플롯
target_counts = train['target'].value_counts()
axes[0].bar(['일반 트윗 (0)', '재난 트윗 (1)'], target_counts.values,
            color=['skyblue', 'coral'], edgecolor='black')
axes[0].set_ylabel('개수', fontsize=12)
axes[0].set_title('트윗 분류 분포', fontsize=14)
for i, v in enumerate(target_counts.values):
    axes[0].text(i, v, f'{v}\n({v/len(train)*100:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

# 파이 차트
axes[1].pie(target_counts.values, labels=['일반 트윗', '재난 트윗'],
           autopct='%1.1f%%', startangle=90, colors=['skyblue', 'coral'],
           explode=(0.05, 0.05))
axes[1].set_title('트윗 분류 비율', fontsize=14)

plt.suptitle('재난 트윗 데이터셋 타겟 분포', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(IMAGES_DIR / '타겟분포.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   저장: {IMAGES_DIR / '타겟분포.png'}")

# 2. 트윗 길이 분석
print("\n2. 트윗 길이 분석 시각화 중...")
train['text_length'] = train['text'].apply(lambda x: len(str(x).split()))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 전체 분포
axes[0].hist(train['text_length'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].set_xlabel('트윗 길이 (단어 수)', fontsize=12)
axes[0].set_ylabel('빈도', fontsize=12)
axes[0].set_title('트윗 길이 분포 (전체)', fontsize=14)
axes[0].axvline(train['text_length'].mean(), color='red', linestyle='--',
                linewidth=2, label=f'평균: {train["text_length"].mean():.1f}')
axes[0].axvline(train['text_length'].median(), color='green', linestyle='--',
                linewidth=2, label=f'중앙값: {train["text_length"].median():.1f}')
axes[0].legend()

# 재난/일반 비교
disaster_lengths = train[train['target'] == 1]['text_length']
normal_lengths = train[train['target'] == 0]['text_length']

axes[1].hist([normal_lengths, disaster_lengths], bins=30,
            label=['일반 트윗', '재난 트윗'],
            color=['skyblue', 'coral'], edgecolor='black', alpha=0.7)
axes[1].set_xlabel('트윗 길이 (단어 수)', fontsize=12)
axes[1].set_ylabel('빈도', fontsize=12)
axes[1].set_title('트윗 길이 분포 (타겟별)', fontsize=14)
axes[1].legend()

plt.suptitle('트윗 길이 분석', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(IMAGES_DIR / '트윗길이분석.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   저장: {IMAGES_DIR / '트윗길이분석.png'}")

# 3. 키워드 분석
print("\n3. 키워드 분석 시각화 중...")
# 결측치 제거
train_with_keyword = train[train['keyword'].notna()]

# 상위 키워드
top_keywords = train_with_keyword['keyword'].value_counts().head(20)

fig, ax = plt.subplots(figsize=(12, 8))
top_keywords.plot(kind='barh', ax=ax, color='teal', edgecolor='black')
ax.set_xlabel('빈도', fontsize=12)
ax.set_ylabel('키워드', fontsize=12)
ax.set_title('상위 20개 키워드', fontsize=14)
ax.invert_yaxis()

# 값 표시
for i, v in enumerate(top_keywords.values):
    ax.text(v, i, f' {v}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(IMAGES_DIR / '키워드분석.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   저장: {IMAGES_DIR / '키워드분석.png'}")

# 4. 타겟별 키워드 비교
print("\n4. 타겟별 키워드 비교 시각화 중...")
disaster_keywords = train_with_keyword[train_with_keyword['target'] == 1]['keyword'].value_counts().head(10)
normal_keywords = train_with_keyword[train_with_keyword['target'] == 0]['keyword'].value_counts().head(10)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 재난 트윗 키워드
axes[0].barh(range(len(disaster_keywords)), disaster_keywords.values, color='coral', edgecolor='black')
axes[0].set_yticks(range(len(disaster_keywords)))
axes[0].set_yticklabels(disaster_keywords.index)
axes[0].set_xlabel('빈도', fontsize=12)
axes[0].set_title('재난 트윗 상위 10개 키워드', fontsize=13)
axes[0].invert_yaxis()
for i, v in enumerate(disaster_keywords.values):
    axes[0].text(v, i, f' {v}', va='center', fontsize=9)

# 일반 트윗 키워드
axes[1].barh(range(len(normal_keywords)), normal_keywords.values, color='skyblue', edgecolor='black')
axes[1].set_yticks(range(len(normal_keywords)))
axes[1].set_yticklabels(normal_keywords.index)
axes[1].set_xlabel('빈도', fontsize=12)
axes[1].set_title('일반 트윗 상위 10개 키워드', fontsize=13)
axes[1].invert_yaxis()
for i, v in enumerate(normal_keywords.values):
    axes[1].text(v, i, f' {v}', va='center', fontsize=9)

plt.suptitle('타겟별 키워드 비교', fontsize=16, y=1.00)
plt.tight_layout()
plt.savefig(IMAGES_DIR / '타겟별키워드.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   저장: {IMAGES_DIR / '타겟별키워드.png'}")

# 5. 단어 빈도 분석 (간단한 버전)
print("\n5. 단어 빈도 분석 시각화 중...")

def get_top_words(texts, n=15):
    """텍스트에서 상위 N개 단어 추출"""
    all_words = []
    for text in texts:
        # 간단한 전처리: 소문자 변환, 특수문자 제거
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        # 간단한 불용어 제거
        stopwords = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and',
                     'is', 'it', 'this', 'that', 'with', 'as', 'by', 'from', 'or',
                     'be', 'are', 'was', 'were', 'been', 'have', 'has', 'had'}
        words = [w for w in words if w not in stopwords and len(w) > 2]
        all_words.extend(words)
    return Counter(all_words).most_common(n)

disaster_texts = train[train['target'] == 1]['text']
normal_texts = train[train['target'] == 0]['text']

disaster_top = get_top_words(disaster_texts, 15)
normal_top = get_top_words(normal_texts, 15)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 재난 트윗 단어
words, counts = zip(*disaster_top)
axes[0].barh(range(len(words)), counts, color='coral', edgecolor='black')
axes[0].set_yticks(range(len(words)))
axes[0].set_yticklabels(words)
axes[0].set_xlabel('빈도', fontsize=12)
axes[0].set_title('재난 트윗 상위 15개 단어', fontsize=13)
axes[0].invert_yaxis()

# 일반 트윗 단어
words, counts = zip(*normal_top)
axes[1].barh(range(len(words)), counts, color='skyblue', edgecolor='black')
axes[1].set_yticks(range(len(words)))
axes[1].set_yticklabels(words)
axes[1].set_xlabel('빈도', fontsize=12)
axes[1].set_title('일반 트윗 상위 15개 단어', fontsize=13)
axes[1].invert_yaxis()

plt.suptitle('타겟별 상위 단어 빈도', fontsize=16, y=1.00)
plt.tight_layout()
plt.savefig(IMAGES_DIR / '단어빈도분석.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   저장: {IMAGES_DIR / '단어빈도분석.png'}")

# 6. 결측치 분석
print("\n6. 결측치 분석 시각화 중...")
missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
missing_percent = (missing / len(train)) * 100

if len(missing) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    missing_percent.plot(kind='barh', ax=ax, color='orange', edgecolor='black')
    ax.set_xlabel('결측치 비율 (%)', fontsize=12)
    ax.set_ylabel('변수명', fontsize=12)
    ax.set_title('변수별 결측치 비율', fontsize=14)

    for i, v in enumerate(missing_percent):
        ax.text(v, i, f' {v:.1f}% ({missing.values[i]}개)', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / '결측치분석.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   저장: {IMAGES_DIR / '결측치분석.png'}")
else:
    print("   결측치가 없습니다 (keyword는 0.8%만 결측)")

# 7. 주요 통계 정보 출력
print("\n" + "="*60)
print("주요 통계 정보")
print("="*60)
print(f"\n타겟 분포:")
print(f"  일반 트윗 (0): {(train['target'] == 0).sum()}개 ({(train['target'] == 0).sum()/len(train)*100:.1f}%)")
print(f"  재난 트윗 (1): {(train['target'] == 1).sum()}개 ({(train['target'] == 1).sum()/len(train)*100:.1f}%)")

print(f"\n트윗 길이 통계:")
print(f"  평균: {train['text_length'].mean():.1f} 단어")
print(f"  중앙값: {train['text_length'].median():.0f} 단어")
print(f"  최소: {train['text_length'].min()} 단어")
print(f"  최대: {train['text_length'].max()} 단어")

print(f"\n키워드 통계:")
print(f"  고유 키워드 수: {train['keyword'].nunique()}개")
print(f"  키워드 결측: {train['keyword'].isnull().sum()}개 ({train['keyword'].isnull().sum()/len(train)*100:.1f}%)")

print(f"\n재난 트윗 vs 일반 트윗 평균 길이:")
print(f"  재난 트윗: {disaster_lengths.mean():.1f} 단어")
print(f"  일반 트윗: {normal_lengths.mean():.1f} 단어")

print("\n시각화 생성 완료!")
print(f"모든 이미지가 {IMAGES_DIR}/ 폴더에 저장되었습니다.")
