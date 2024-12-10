import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, train_test_split
from sklearn.metrics import make_scorer, confusion_matrix, matthews_corrcoef, roc_auc_score,accuracy_score
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization

# 定义模型
RF = RandomForestClassifier()
XGB = XGBClassifier()
LR = LogisticRegression()

# 定义评估指标
def sensitivity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn)

def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

scoring = {
    'acc': make_scorer(accuracy_score),
    'mcc': make_scorer(matthews_corrcoef),
    'auc': 'roc_auc',
    'sen': make_scorer(sensitivity),
    'spe': make_scorer(specificity)
}

def lightGBM(x,y):

    # 训练LightGBM模型并获取特征重要性
    model = lgb.LGBMClassifier()
    model.fit(x, y)
    feature_importance = model.feature_importances_

    # 根据特征重要性对特征进行排序
    features = sorted(range(len(feature_importance)), key=lambda k: feature_importance[k], reverse=True)

    return features[:50]





def read_feature(dataset,i):

        # 重新初始化字典
        df_pos_features = {1: 'AAC_pos', 2: 'descriptor_rdkitpos', 3: 'DPC_pos', 4: 'NT_CT_5_pos', 5: 'ESM1-b_pos',
                           6: 'fingerprints_pos', 7: 'Atom_count_pos'}
        df_neg_features = {1: 'AAC_neg', 2: 'descriptor_rdkitneg', 3: 'DPC_neg', 4: 'NT_CT_5_neg', 5: 'ESM1-b_neg',
                           6: 'fingerprints_neg', 7: 'Atom_count_neg'}


        print(f'特征{i}')

        # 特征路径
        pos_feature_path = f'./feature_data/{df_pos_features[i]}_{dataset}.csv'
        neg_feature_path = f'./feature_data/{df_neg_features[i]}_{dataset}.csv'

        # 提取特征的正样本
        df_pos_features[i] = pd.read_csv(pos_feature_path)
        df_pos_features[i]['label'] = 1


        # 提取特征的负样本
        df_neg_features[i] = pd.read_csv(neg_feature_path)
        df_neg_features[i]['label'] = 0

        # 合并正负样本的数据
        df = pd.concat([df_pos_features[i], df_neg_features[i]], ignore_index=True)

        if i in [3, 7]:

            # 中位数填充
            df.fillna(df.median(), inplace=True)


        if i in [2, 6] :

            # 查找非数字值并替换
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if pd.api.types.is_numeric_dtype(df[col]):
                    # 计算缺失值、0值、inf值，如果>85%就删除此列
                    nan_percent = (df[col].isnull().sum() / len(df[col])) * 100
                    zero_percent = ((df[col] == 0).sum(axis=0) / len(df[col])) * 100

                if nan_percent > 85.0 or zero_percent > 85.0 :
                    df.drop(col, axis=1, inplace=True)
                    continue

            # 查找无穷值
            df.replace([float('inf'), float('-inf')], 0, inplace=True)

            # 中位数填充
            df.fillna(df.median(), inplace=True)

        # 打乱数据顺序
        df = df.sample(frac=1, random_state=1).reset_index(drop=True)

        # 提取特征和标签
        if i in [2, 4, 5]:
            X = df.iloc[:, 0:-1]
        else:
            X = df.iloc[:, 1:-1]
        y = df.iloc[:, -1]


        if i in [2, 3, 5]:

            # 特征选择
            bestsubset_lightGBM = lightGBM(X, y)
            X_lightGBM = X.iloc[:, bestsubset_lightGBM]

        return X,y



def Stacking_self(X_train, y_train, j):


    # 使用五折交叉验证
    kf = KFold(n_splits=5)

    # 初始化存储结果的数组
    probs_train = np.zeros((len(X_train), 1))
    probs_test = np.zeros((len(X_test), 5))

    i = 0
    for train_index, val_index in kf.split(X_train):

        # 将数据分成训练集和验证集
        X_CV_train, X_val = X_train.iloc[train_index, :], X_train.iloc[val_index, :]
        y_CV_train, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

        if j in [1, 3, 4, 5]:

            # 使用RF训练分类器
            RF.fit(X_CV_train, y_CV_train)

            # 获取验证集上的概率预测
            probs_train[val_index] = RF.predict_proba(X_val)[:, 1].reshape(-1, 1)

            # 获取测试集上的概率预测
            probs_test[:, i] = RF.predict_proba(X_test)[:, 1]
            i = i + 1

        else:

            # 使用XGB训练分类器
            XGB.fit(X_CV_train, y_CV_train)

            # 获取验证集上的概率预测
            probs_train[val_index] = XGB.predict_proba(X_val)[:, 1].reshape(-1, 1)

            # 获取测试集上的概率预测
            probs_test[:, i] = XGB.predict_proba(X_test)[:, 1]
            i = i + 1


    probs_test = np.mean(probs_test, axis=1).reshape(-1,1)

    return probs_train, probs_test


# 主程序入口
if __name__ == "__main__":

    probs_train_combine = np.empty((1503, 0))
    probs_test_combine = np.empty((376, 0))

    for i in range(1, 8):

        X_train, y_train = read_feature('train',i)


        # 重新初始化字典
        df_pos_features = {1: 'AAC_pos', 2: 'descriptor_rdkitpos', 3: 'DPC_pos', 4: 'NT_CT_5_pos',
                           6: 'fingerprints_pos', 7: 'Atom_count_pos'}
        df_neg_features = {1: 'AAC_neg', 2: 'descriptor_rdkitneg', 3: 'DPC_neg', 4: 'NT_CT_5_neg',
                           6: 'fingerprints_neg', 7: 'Atom_count_neg'}

        # 读取测试集数据
        X_test, y_test = read_feature('test',i)

        probs_train, probs_test = Stacking_self(X_train, y_train, i)
        probs_train_combine = np.concatenate([probs_train_combine, probs_train], axis=1)
        probs_test_combine = np.concatenate([probs_test_combine, probs_test], axis=1)

    for score_name, score_func in scoring.items():
        cv_scores = cross_val_score(LR, probs_train_combine, y_train, cv=5, scoring = score_func)
        print(f"CV{score_name}:{np.mean(cv_scores)}")




    # # 定义超参数的范围
    # pbounds = {
    #     'n_estimators': (10, 200),
    #     'max_depth': (5, 50),
    #     'min_samples_split': (2, 20),
    #     'learning_rate': (0.01, 0.5),
    #     'n_estimators_xgb': (10, 200),
    #     'max_depth_xgb': (5, 50),
    #     'subsample': (0.5, 1.0),
    #     'colsample_bytree': (0.5, 1.0)
    # }
    #
    # # 初始化贝叶斯优化器
    # optimizer = BayesianOptimization(f=evaluate_model, pbounds=pbounds, random_state=1)
    #
    # # 运行优化过程，这里以30次迭代为例
    # optimizer.maximize(init_points=10, n_iter=30)
    #
    # # 输出最优参数
    # print("Best parameters found: ", optimizer.max)


    # 模型在测试集上的表现
    LR.fit(probs_train_combine, y_train)
    y_pred = LR.predict(probs_test_combine)
    y_proba = LR.predict_proba(probs_test_combine)[:, 1]
    print(f'Test sen: {sensitivity(y_test, y_pred):.3f}')
    print(f'Test spe: {specificity(y_test, y_pred):.3f}')
    print(f'Test acc: {accuracy_score(y_test, y_pred):.3f}')
    print(f'Test mcc: {matthews_corrcoef(y_test, y_pred):.3f}')
    print(f'Test auc: {roc_auc_score(y_test, y_proba):.3f}')


