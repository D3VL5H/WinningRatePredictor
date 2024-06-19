import os
import argparse
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

argparser = argparse.ArgumentParser()
argparser.add_argument('--filename', type=str, default='k_data.csv', help='data file name(.csv). default: %(default)s')
argparser.add_argument('--model', type=str, default='model_0618.pkl', help='model file name(.pkl). default: %(default)s')
args = argparser.parse_args()

if not args.filename.endswith('.csv'):
    args.filename += '.csv'

if not os.path.exists(args.filename):
    raise Exception(f'{args.filename} does not exist')

csv_file = args.filename

model_dict = joblib.load(f'model/{args.model}')
model = model_dict['model']
mean_dict = model_dict['mean']
scaler = model_dict['scaler']

df = pd.read_csv(csv_file, encoding='cp949')
df.replace('%', '', regex=True, inplace=True)
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors='ignore')
df['homeground'] = 1

# 모델엔 있으나 데이터엔 없는 피쳐가 있을 시 모델 학습 당시의 평균값으로 채움
for n in mean_dict.keys():
    if n not in df.columns:
        print(f'{n} is not in columns, fill to default value: {mean_dict[n]}')
        df[n] = mean_dict[n]

# 결측치가 존재한다면 해당 피쳐의 평균값으로 채움
df.fillna(df.drop(columns=['result']).mean(), inplace=True)

X = df.drop(columns=['result'])[mean_dict.keys()]
y = df['result']

X_scaled = scaler.transform(X)

# label encoding (DRAW=0, LOSS=1, WIN=2)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

y_pred = model.predict(X_scaled)
y_pred_proba = model.predict_proba(X_scaled)

# label decoding (0=DRAW, 1=LOSS, 2=WIN)
y_pred_labels = label_encoder.inverse_transform(y_pred)

print(f'predicted value: {y_pred}')
print(f'proba: {y_pred_proba}')

# # 디테일한 결과출력도 옵션으로 넣을것..
# result_list = ['DRAW', 'LOSS', 'WIN']

# print(f'predicted value: {y_pred_labels[y_pred[0]]}')
# print(f'proba:')
# for i, pb in enumerate(y_pred_proba[0]):
#     print(f'{y_pred_labels[i]}: {pb * 100.0:.3f}%')

accuracy = accuracy_score(y, y_pred)
conf_mat = confusion_matrix(y, y_pred)
cls_report = classification_report(y, y_pred, target_names=label_encoder.classes_)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_mat}')
print(f'Classification Report:\n{cls_report}')
