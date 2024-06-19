import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import joblib
import utils
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from datetime import datetime

# 그래프에 한글 표시하기 위함
font_path = 'C:/Windows/Fonts/malgun.ttf'
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = 'data/'

csv_file_list = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]

features = set()
df_list = []
for csv_file in csv_file_list:
    df = pd.read_csv(f'{DATA_DIR}{csv_file}', encoding='cp949')    
    features.update(df.columns)
    df_list.append(df)
    print(csv_file)
df = pd.concat(df_list, axis=0, ignore_index=True)
df = df.reindex(columns=sorted(features))
df.replace('%', '', regex=True, inplace=True)
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors='ignore')

print(f'all features: {df.columns}')

# 일부 피처는 데이터가 너무 적어 뺌
exclude_var_list = ['score', '거리 (km)', '가로채기', '골킥', '예상 득점 (xG)', '크로스', '클리어런스', '패스 성공률']
mod_exclude_var_list = [v + '_home' for v in exclude_var_list] + [v + '_away' for v in exclude_var_list] 
df.drop(columns=mod_exclude_var_list, inplace=True)

print(f'used features: {df.columns}')

# 결측치를 평균값으로 채움
df.fillna(df.drop(columns=['result']).mean(), inplace=True)

print(df.count())

# homeground여부를 반영하기위해 가상의 away team 데이터 생성
df_away_as_home = df.copy()
df_away_as_home['result'] = df['result'].map({'WIN':'LOSS', 'LOSS':'WIN'}).fillna(df['result'])
for c in df.columns:
    if not c.endswith('_home'):
        continue
    away_c = c.replace('_home', '_away')
    df_away_as_home[c], df_away_as_home[away_c] = df[away_c], df[c]

df['homeground'] = 1
df_away_as_home['homeground'] = 0
df = pd.concat([df, df_away_as_home], ignore_index=True)

# 상관행렬 시각화
corr_matrix = df.copy()
corr_matrix['result'] = corr_matrix['result'].map({'DRAW': 0, 'LOSS': 1, 'WIN': 2})
corr_matrix = corr_matrix.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

X = df.drop(columns=['result'])
y = df['result']

# 문자 label을 숫자로 변환
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 데이터 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)

# 스케일링으로 데이터 늘리기
scale_factors = np.arange(0.8, 1.4, 0.1)
X_scaled_augmented = np.concatenate([X_scaled * factor for factor in scale_factors], axis=0)
y_augmented = np.concatenate([y for _ in scale_factors], axis=0)

# train set, test set 분할
#X_train, X_test, y_train, y_test = train_test_split(X_scaled_augmented, y_augmented, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_scaled_augmented, y_augmented, test_size=0.2, random_state=42)

print(len(X_train))
print(len(X_test))

# model_list = {
#     "Logistic Regression": LogisticRegression(penalty='l2', solver='lbfgs', class_weight='balanced', max_iter=10000, C=1.0),
#     "Decision Tree": DecisionTreeClassifier(),
#     "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
#     "Support Vector Machine": SVC(),
#     "Neural Network": MLPClassifier(max_iter=500),
#     "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
#     "Gradient Boosting": GradientBoostingClassifier(random_state=42)
# }

# solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

# 로지스틱 회귀 모델 생성 및 훈련
# 승, 패, 비김으로 2가지 이상의 경우가 있기때문에 다중 클래스 분류 사용
start_time = time.time()
model = LogisticRegression(penalty='l2',
                           solver='lbfgs',
                           class_weight='balanced',
                           max_iter=10000,
                           C=1.0)
model.fit(X_train, y_train)

print(f'Elapsed: {time.time()-start_time}')

# rf_model = RandomForestClassifier(n_estimators=100)
# rf_model.fit(X_train, y_train)

# 교차 검증
print(f'Cross Validation Score: {cross_val_score(model, X_train, y_train, cv=5)}')

model_dict = {'model': model, 'mean': X.mean().to_dict(), 'scaler': scaler}
# 모델 저장
now = datetime.now()
formatted_now = now.strftime('%Y%m%d%H%M%S')
model_name = f"model_{formatted_now}.pkl"
utils.create_dir_if_not_exists('model')
joblib.dump(model_dict, f'model/{model_name}', compress=3)

# 승패 예측
y_pred = model.predict(X_test)

# 예측에 대한 확률
y_pred_proba = model.predict_proba(X_test)

# 예측된 숫자형 레이블을 다시 문자열 레이블로 변환
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_test_labels = label_encoder.inverse_transform(y_test)

# rf_y_pred_labels = label_encoder.inverse_transform(rf_y_pred)

# 독립변수가 종속변수에 미치는 영향 (회귀계수)출력
feature_names = X.columns
for i, class_label in enumerate(label_encoder.classes_):
    print(f"[{class_label}] coefficients:")
    for coef, feature in zip(model.coef_[i], feature_names):
        print(f"  {feature}: {coef:.4f}")

loop_cnt = 0
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
for m_coef in model.coef_:
    coef_df = pd.DataFrame({'Features': X.columns, 'Coefficients': m_coef})
    print(coef_df.head())

    # 계수 절대값 기준으로 정렬
    coef_df['Abs_Coefficients'] = coef_df['Coefficients'].abs()
    coef_df = coef_df.sort_values(by='Abs_Coefficients', ascending=False)
    # 상위 n개 선택해서 표시
    drop_top = 0 #4
    show_top = 12 #7
    coef_df = coef_df.iloc[drop_top:drop_top+show_top]

    sns.barplot(x='Features', y='Coefficients', data=coef_df, ax=axs[loop_cnt])
    print(y_pred_labels[loop_cnt])
    axs[loop_cnt].set_title(f'Feature Importance({label_encoder.classes_[loop_cnt]})')
    axs[loop_cnt].set_xlabel('Features')
    axs[loop_cnt].set_ylabel('Coefficient')

    axs[loop_cnt].set_xticklabels(axs[loop_cnt].get_xticklabels(), rotation=90, ha='right', fontsize=10)

    loop_cnt+=1

plt.tight_layout()
plt.show()

# 모델 평가
accuracy = accuracy_score(y_test_labels, y_pred_labels)
conf_mat = confusion_matrix(y_test_labels, y_pred_labels)
cls_report = classification_report(y_test_labels, y_pred_labels, target_names=label_encoder.classes_)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_mat}')
print(f'Classification Report:\n{cls_report}')

utils.create_dir_if_not_exists('log')
with open(f'log/{formatted_now}.log', 'w') as f:
    f.write(f'ModelName:\n{model_name}\n')
    f.write(f'Accuracy:\n{accuracy}\n')
    f.write(f'Classification Report:\n{cls_report}')
