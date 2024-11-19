import os

import numpy as np
import pandas as pd
import shap as shap
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from boruta import BorutaPy
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, train_test_split
from sklearn.metrics import make_scorer, confusion_matrix, matthews_corrcoef, roc_auc_score,accuracy_score
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from bayes_opt import BayesianOptimization

# 定义模型
RF = RandomForestClassifier()
XGB = XGBClassifier()

def F_Score(X,y):

    columns_df = X.columns
    X = X.values

    # 创建选择器对象，选择所有特征
    selector = SelectKBest(score_func=f_classif,k = 'all')

    # 使用选择器对象对特征进行选择
    X_new = selector.fit_transform(X, y)

    # 获取特征的F-Score
    f_scores = selector.scores_

    # 创建一个DataFrame来存储特征名和对应的F-Score
    feature_scores = pd.DataFrame({'Features': columns_df, 'F-Score': f_scores})

    # 按照F-Score从大到小排列
    feature_scores_sorted = feature_scores.sort_values(by='F-Score', ascending=False)
    feature_scores_50 = feature_scores_sorted["Features"][:50]
    return feature_scores_50

    # 将排序后的特征输出到csv文件中
    # feature_scores_sorted.to_csv('fingerprints_F_Score.csv',index = False)

    # # 绘制特征的F-Score图
    # plt.bar(range(len(f_scores)), f_scores)
    # plt.xlabel('Feature Index')
    # plt.ylabel('F-Score')
    # plt.title('F-Score of Features')
    # plt.show()

def boruta(X,y):

    columns_df = X.columns
    X = X.values

    rf = RandomForestClassifier(n_jobs=-1, max_depth=5)

    # define Boruta feature selection method
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

    # find all relevant features - 5 features should be selected
    feat_selector.fit(X, y)

    # check ranking of features
    features_importance = feat_selector.ranking_

    # call transform() on X to filter it down to selected features
    # X_filtered = feat_selector.transform(X)
    # print(X_filtered)

    df_features_importance = pd.DataFrame({'Features':columns_df,'importance':features_importance}).sort_values('importance',ascending = False)
    return df_features_importance["Features"][:50]

def lightGBM_IFS(X,y):

    # 训练LightGBM模型并获取特征重要性
    model = lgb.LGBMClassifier()
    model.fit(X, y)
    feature_importance = model.feature_importances_

    # 多标签：训练LightGBM模型并获取特征重要性
    # model = lgb.LGBMClassifier()
    # mutilabel_clf_classifier = OneVsRestClassifier(model)
    # mutilabel_clf_classifier.fit(X, y)
    # # 获取特征重要性
    # feature_importance = mutilabel_clf_classifier.estimators_[0].feature_importances_

    # 根据特征重要性对特征进行排序
    features = sorted(range(len(feature_importance)), key=lambda k: feature_importance[k], reverse=True)
    # feature_100 = features[:100]
    #
    # # IFS 特征选择
    # best_score = 0
    # best_subset = []
    # auc_scores = []
    # for i in range(10,101,10):
    #     subset = features[:i]
    #     cv_scores = cross_val_score(mutilabel_clf_classifier, X.values[:, subset], y, scoring = 'roc_auc',cv=5)
    #     score = cv_scores.mean()
    #     auc_scores.append(score)
    #
    #     if score > best_score:
    #         best_score = score
    #         best_subset = subset
    #     print(f'Number of features: {i}, AUC为{score}, CV (+/- {np.std(cv_scores):.3f})')

    # # 画出AUC随特征数量变化的折线图
    # plt.plot(range(10,101,10), auc_scores, marker='o')
    # plt.xticks(np.arange(10,201,step = 10))
    # plt.yticks(np.arange(0.9200,1.00,step = 0.01))
    # plt.xlabel('Number of features')
    # plt.ylabel('AUC')

    # # 标记最大值点
    # best_num_features = len(best_subset)
    # plt.scatter(best_num_features, best_score, color='red', s = 100)
    # plt.annotate(text=f'Best Score: ({best_num_features:.3f},{best_score:.3f})', xy=(best_num_features, best_score),xytext=(best_num_features - 20.0, best_score + 0.0005))
    # plt.vlines(x = best_num_features,ymin = 0.9200,ymax = best_score,linestyle = 'dashed',color = 'blue')
    # plt.show()
    #
    # return best_subset
    return features[:50]

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







probs_best = np.empty((1503, 0))
df_pos_features = {1: 'AAC_pos', 2: 'descriptor_rdkitpos', 3: 'DPC_pos', 4: 'NT_CT_5_pos' ,7: 'fingerprints_pos',8:'diatom_pos',9:'Atom_count_pos'}
df_neg_features = {1: 'AAC_neg', 2: 'descriptor_rdkitneg', 3: 'DPC_neg', 4: 'NT_CT_5_neg' ,7: 'fingerprints_neg',8:'diatom_neg' ,9:'Atom_count_neg'}


def read_feature(dataset,i):

        # 重新初始化字典
        df_pos_features = {1: 'AAC_pos', 2: 'descriptor_rdkitpos', 3: 'DPC_pos', 4: 'NT_CT_5_pos',
                           7: 'fingerprints_pos',
                           8: 'diatom_pos', 9: 'Atom_count_pos'}
        df_neg_features = {1: 'AAC_neg', 2: 'descriptor_rdkitneg', 3: 'DPC_neg', 4: 'NT_CT_5_neg',
                           7: 'fingerprints_neg',
                           8: 'diatom_neg', 9: 'Atom_count_neg'}


        print(f'特征{i}')

        # 读取深度学习特征
        if i == 6:
            # 读取数据
            df = pd.read_csv(f'{dataset}_data.csv', header=None, index_col=None)

            # 读取标签
            df_label = pd.read_csv(f'{dataset}_label.csv', header=None, index_col=None).T

            # 打乱数据顺序
            df['label'] = df_label
            df = df.sample(frac=1).reset_index(drop=True)

        elif i == 5:
            pos_filenames = os.listdir(f'./esm1b/pos_{dataset}_1_seq')
            neg_filenames = os.listdir(f'./esm1b/neg_{dataset}_1_seq')
            df_pos = pd.DataFrame()
            df_neg = pd.DataFrame()
            df = pd.DataFrame()

            # 读取正样本
            for pos_filename in pos_filenames:
                inputfile = os.path.join(f'./esm1b/pos_{dataset}_1_seq', pos_filename)

                with open(inputfile, 'r') as f:
                    lines = f.readlines()
                    pos_data = [line.strip().split() for line in lines]
                    df_pos = df_pos.append(pos_data)
            df_pos['label'] = 1

            # 读取负样本
            for neg_filename in neg_filenames:
                inputfile = os.path.join(f'./esm1b/neg_{dataset}_1_seq', neg_filename)

                with open(inputfile, 'r') as f:
                    lines = f.readlines()
                    neg_data = [line.strip().split() for line in lines]
                    df_neg = df_neg.append(neg_data)
            df_neg['label'] = 0

            # 合并正负样本
            df = pd.concat([df_pos, df_neg], ignore_index=True)

            # 转换数据类型
            df = df.apply(pd.to_numeric, errors='coerce')
            df.to_csv("esm1-b.csv", index = False)

            # 读取样本和标签
            # Esm1_b_feature = pd.read_csv('Esm1-b.csv', header=None)
            # Esm1_b_label = pd.read_csv('Esm1-b-label.csv')
            # df = pd.concat([Esm1_b_feature, Esm1_b_label], ignore_index=True, axis = 1)



        else:
            # 提取特征的正样本
            df_pos_features[i] = pd.read_csv(f'{df_pos_features[i]}_{dataset}.csv')
            df_pos_features[i]['label'] = 1
            # labels = pd.read_csv(f'pos_{dataset}_seq_label.csv')
            # labels = labels.iloc[:, 3:].fillna(0)


            # 提取特征的负样本
            df_neg_features[i] = pd.read_csv(f'{df_neg_features[i]}_{dataset}.csv')
            df_neg_features[i]['label'] = 0

            # 合并正负样本的数据
            df = pd.concat([df_pos_features[i], df_neg_features[i]], ignore_index=True)
            # 合并样本和标签
            # df = pd.concat([df_pos_features[i], labels], axis=1)

        if i == 3 or i == 9:
            # 中位数填充
            df.fillna(df.median(), inplace=True)


        if i == 2 or i == 7 or i == 6 or i == 8 :
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
                # # 计算该列的均值
                # mean = df[col].mean()
                # # 将NaN替换为均值
                # df[col] = df[col].fillna(mean)

            # 查找无穷值
            df.replace([float('inf'), float('-inf')], 0, inplace=True)

            # 中位数填充
            df.fillna(df.median(), inplace=True)

            # 特征归一化
            # for col in df.iloc[:, : -1].columns:
            #     df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min()) * 100

            print("111")

        # 打乱数据顺序
        df = df.sample(frac=1, random_state=1).reset_index(drop=True)

        # k近邻填充
        # imputer = KNNImputer(n_neighbors=5)
        # X = imputer.fit_transform(df)
        # df = pd.DataFrame(X)



        # 提取特征和标签
        if i == 4 or i == 5 or i == 6 or i == 2:
            X = df.iloc[:, 0:-1]
            y = df.iloc[:, -1]

        else:
            X = df.iloc[:, 1:-1]
            y = df.iloc[:, -1]


        if i in [2, 3, 5]:

            # 特征选择
            bestsubset_lightGBM_IFS = lightGBM_IFS(X, y)
            X_lightGBM = X.iloc[:, bestsubset_lightGBM_IFS]

            X_boruta_subset = boruta(X, y)
            X_boruta = X.loc[:, X_boruta_subset]

            X_F_score_subset = F_Score(X, y)
            X_F_score = X.loc[:, X_F_score_subset]

            method_outputs = [X_lightGBM, X_boruta, X_F_score]

            for method_output in method_outputs:
                for score_name, score_func in scoring.items():
                    cv_scores = cross_val_score(XGB, method_output, y, cv=5, scoring=score_func)
                    print(f"CV{score_name}:{np.mean(cv_scores)}")

            print("----------")


        # return X,y



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



# def evaluate_model(n_estimators, max_depth, min_samples_split, learning_rate, n_estimators_xgb, max_depth_xgb, subsample, colsample_bytree):

probs_train_combine = np.empty((1503, 0))
probs_test_combine = np.empty((376, 0))
LR = LogisticRegression()


for i in range(2, 10):

    if i in [6, 8]:
        continue


    X_train, y_train = read_feature('train',i)
    # df_feature_train = pd.concat([df_feature_train, X_train], axis=1)

    # 重新初始化字典
    df_pos_features = {1: 'AAC_pos', 2: 'descriptor_rdkitpos', 3: 'DPC_pos', 4: 'NT_CT_5_pos',
                       7: 'fingerprints_pos',
                       8: 'diatom_pos', 9: 'Atom_count_pos'}
    df_neg_features = {1: 'AAC_neg', 2: 'descriptor_rdkitneg', 3: 'DPC_neg', 4: 'NT_CT_5_neg',
                       7: 'fingerprints_neg',
                       8: 'diatom_neg', 9: 'Atom_count_neg'}

    # 读取测试集数据
    X_test, y_test = read_feature('test',i)
    # df_feature_test = pd.concat([df_feature_test, X_test], axis=1)

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



LR.fit(probs_train_combine, y_train)
y_pred = LR.predict(probs_test_combine)
y_proba = LR.predict_proba(probs_test_combine)[:, 1]
print(f'Test sen: {sensitivity(y_test, y_pred):.3f}')
print(f'Test spe: {specificity(y_test, y_pred):.3f}')
print(f'Test acc: {accuracy_score(y_test, y_pred):.3f}')
print(f'Test mcc: {matthews_corrcoef(y_test, y_pred):.3f}')
print(f'Test auc: {roc_auc_score(y_test, y_proba):.3f}')


# df_feature_train = pd.DataFrame()
# df_feature_test = pd.DataFrame()
#
# for i in range(1,4):
#
#     X_train,y_train = read_feature('train',i)
#     df_feature_train = pd.concat([df_feature_train,X_train],axis = 1)
#
# df_feature_train.fillna(df_feature_train.median(), inplace=True)
#
# # 重新初始化字典
# df_pos_features = {1: 'AAC_pos', 2: 'descriptor_rdkitpos', 3: 'DPC_pos', 5: 'NT_CT_5_pos', 7: 'fingerprints_pos',8:'diatom_pos',9:'Atom_count_pos'}
# df_neg_features = {1: 'AAC_neg', 2: 'descriptor_rdkitneg', 3: 'DPC_neg', 5: 'NT_CT_5_neg', 7: 'fingerprints_neg',8:'diatom_neg' ,9:'Atom_count_neg'}
#
# for j in range(1,4):
#     X_test,y_test = read_feature('test',j)
#     df_feature_test = pd.concat([df_feature_test,X_test],axis = 1)
#
# df_feature_test.fillna(df_feature_test.median(), inplace=True)
#
# # 训练模型
# model = RandomForestClassifier()
# model.fit(df_feature_train, y_train)
#
# # 创建SHAP解释器
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(df_feature_test,check_additivity=False)
#
# # 生成摘要图
# shap.summary_plot(shap_values[0], df_feature_test, max_display = 20)
# # # 创建SHAP摘要图
# # shap.summary_plot(shap_values[0], df_feature)
# #
# # 取每个特征的SHAP值的绝对值的平均值作为该特征的重要性
# shap.summary_plot(shap_values[0], df_feature_test, plot_type="bar")

# 创建两个子图
# fig, (ax1, ax2) = plt.subplots(1, 2)
#
# # 在第一个子图中绘制摘要图
# shap.summary_plot(shap_values[0], df_feature, plot_type="bar", show=False, plot_size=(10, 5))
#
# # 在第二个子图中绘制条形图
# shap.summary_plot(shap_values[0], df_feature, show=False, plot_size=(10, 5))
#
# # 调整子图的位置
# fig.tight_layout()


    # models = [
    #      KNeighborsClassifier(),
    #      LogisticRegression(),
    #      MLPClassifier(),
    #      RandomForestClassifier(),
    #      SVC(probability=True),
    #      XGBClassifier()
    # ]
    # for model in models:
    #     print(model)
    #     for score_name, score_func in scoring.items():
    #         cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=score_func)
    #         print(f'CV {score_name}: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores):.3f})')


    # 使用五折交叉验证
#     kf = KFold(n_splits=5)
#
#     #  定义模型
#     RF = RandomForestClassifier()

#     #  初始化存储结果的数组
#     probs = np.zeros((len(X_train), 1))
#
#     for train_index, val_index in kf.split(X_train):
#
#         # 将数据分成训练集和验证集
#         X_CV_train, X_val = X_train[train_index],X_train[val_index]
#         y_CV_train, y_val = y_train[train_index], y_train[val_index]
#
# # #         #if i == 1:
# # #             # # 训练XGBoost分类器
# # #             # xgb_clf = XGBClassifier()
# # #             # xgb_clf.fit(X_train, y_train)
# # #             # # 获取验证集上的概率预测
# # #             # probs[val_index] = xgb_clf.predict_proba(X_val)[:,1].reshape(-1, 1)
# # #
#         # 使用RF训练分类器
#         RF.fit(X_CV_train, y_CV_train)
#         # 获取验证集上的概率预测
#         # probs_AAC[val_index] = RF.predict_proba(X_val)[:, 1].reshape(-1, 1)
# # #             # # 训练RF分类器
# # #             # rf_clf = RandomForestClassifier()
# # #             # rf_clf.fit(X_train, y_train)
#         # 获取验证集上的概率预测
#         probs[val_index] = RF.predict_proba(X_val)[:,1].reshape(-1, 1)
#
# # #         if i == 4:
# # #
# # #             XGB.fit(X_CV_train, y_CV_train)
# # #             probs[val_index] = XGB.predict_proba(X_val)[:,1].reshape(-1, 1)
# # #
# # #         else:
# # #             RF.fit(X_CV_train, y_CV_train)
# # #             probs[val_index] = RF.predict_proba(X_val)[:,1].reshape(-1, 1)
# # #
#         probs_combine = np.concatenate([probs_combine, probs], axis=1)
#         X2_train = probs_combine
#         y2_train = y_train
#
# model = LogisticRegression()
# y_pred = model.predict(X_train)
# y_proba = model.predict_proba(X_train)[:, 1]
# print(model)
# print(f'Test sen: {sensitivity(y_train, y_pred):.3f}')
# print(f'Test spe: {specificity(y_train, y_pred):.3f}')
# print(f'Test acc: {accuracy_score(y_train, y_pred):.3f}')
# print(f'Test mcc: {matthews_corrcoef(y_train, y_pred):.3f}')
# print(f'Test auc: {roc_auc_score(y_train, y_proba):.3f}')

# # 第二层LR交叉验证
# for score_name, score_func in scoring.items():
#     cv_scores = cross_val_score(model, X2_train, y2_train, cv=5, scoring=score_func)
#     print(f'CV {score_name}: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores):.3f})')

# columns = ["AAC_RF","descriptor_RF","DPC_RF"]
# X2_train = pd.DataFrame(X2_train,columns = columns)
# model.fit(X2_train, y2_train)
#
# # 创建解释器对象
# explainer = shap.LinearExplainer(model, X2_train)
# shap_values = explainer(X2_train)
# # shap.summary_plot(shap_values, X2_train)
# shap.plots.bar(shap_values)



# np.save('probs_label.npy',probs_combine)
# np.save('y.npy',y_train)
# print('--------')

# probs_combine = probs_best
# print(f'特征{i}的最优算法：{best_model},AUC最佳值为{best_auc}')
    # if i == 8:
    #     np.save('probs_label.npy',probs_combine)
    #     np.save('y.npy',y)


    # 合并所有特征的验证概率
#     all_probs = np.concatenate([all_probs, probs], axis=1)
# # 保存为.npy文件
# # np.save('all_probs.npy', all_probs)
# #
# # # 读取第二层模型的X和标签
# # X2_train = np.load('all_probs.npy')
# X2_train = all_probs
# y2_train = y
#
# 使用pearson相关系数计算特征相关性
# Classifier_pcc = pd.DataFrame(columns = ['Classifier_AAC','Classifier_descriptor','Classifier_DPC','Classifier_NT_CT_5','Classifier_fingerprints','Classifier_Atom_count','Classifier_diatom','Classifier_DL'],index = ['Classifier_AAC','Classifier_descriptor','Classifier_DPC','Classifier_NT_CT_5','Classifier_fingerprints','Classifier_Atom_count','Classifier_diatom','Classifier_DL'])
#
# for i in range(0,8):
#     for j in range(0,8):
#       pcc = pearsonr( probs_combine[:,i], probs_combine[:,j])
#       Classifier_pcc.iloc[i,j] = pcc[0]
# Classifier_pcc.to_csv('Classifier_pcc1.csv',index = True)

    # # 使用算法训练分类器进行比较
    # models = [
    #      KNeighborsClassifier(),
    #      LogisticRegression(),
    #      MLPClassifier(),
    #      RandomForestClassifier(),
    #      SVC(probability=True),
    #      XGBClassifier()
    # ]
    #
    #
    # # 使用五折交叉验证评估模型
    # for model in models:
    #     print(model)
    #     for score_name, score_func in scoring.items():
    #         cv_scores = cross_val_score(model, X, y, cv=5, scoring=score_func)
    #         print(f'CV {score_name}: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores):.3f})')

# 在测试集上测试模型
# for model in models:
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     y_proba = model.predict_proba(X_test)[:, 1]
#     print(model)
#     print(f'Test sen: {sensitivity(y_test, y_pred):.3f}')
#     print(f'Test spe: {specificity(y_test, y_pred):.3f}')
#     print(f'Test acc: {accuracy_score(y_test, y_pred):.3f}')
#     print(f'Test mcc: {matthews_corrcoef(y_test, y_pred):.3f}')
#     print(f'Test auc: {roc_auc_score(y_test, y_proba):.3f}')



    # 读取正样本的特征文件
    # df_positive_descriptor = pd.read_csv("descriptor_rdkitpos_{}.csv".format(sample))
    # df_positive_fingerprints = pd.read_csv("fingerprints_pos_{}.csv".format(sample))
    # df_positive_fingerprints['ID'] = df_positive_fingerprints['ID'].str.replace(r'\.mdl$', '', regex=True)
    # df_positive_Atom_percentage = pd.read_csv("Atom_count_pos_{}.csv".format(sample))
    # df_positive_DPC = pd.read_csv("DPC_pos_{}.csv".format(sample))
    # df_positive_descriptor ['label'] = 1

    # 读取正样本末端特征
    # df_positive_CT = {}
    # df_negative_CT = {}
    # for i in range(2, 6):
    #     df_positive_CT[i] = pd.read_csv(f"./CT-k NT-k/AAC_CT{i}_pos_{sample}.csv")
    #     df_positive_CT[i]['label'] = 1

    # 读取负样本的特征文件
    # df_negative_descriptor = pd.read_csv("descriptor_rdkitneg_{}.csv".format(sample))
    # # df_negative_DPC = pd.read_csv("DPC_neg_{}.csv".format(sample))
    # # df_negative_fingerprints['ID'] = df_negative_fingerprints['ID'].str.replace(r'\.mdl$', '', regex=True)
    # # df_negative_Atom_percentage = pd.read_csv("Atom_count_neg_{}.csv".format(sample))
    # df_negative_descriptor['label'] = 0

    # for i in range(2, 6):
    #     df_negative_CT[i] = pd.read_csv(f"./CT-k NT-k/AAC_CT{i}_neg_{sample}.csv")
    #     df_negative_CT[i]['label'] = 0
    #合并正样本
    # df_positive_defi = df_positive_descriptor.merge(df_positive_fingerprints, on='ID', how="inner")
    # df_pos = pd.merge(df_positive_defi,df_positive_Atom_percentage,on='ID', how="inner" )
    # df_positive_fingerprints ["label"] = 1

    #合并负样本
    # df_negative_defi = df_negative_descriptor.merge(df_negative_fingerprints, on='ID', how="inner")
    # df_neg = pd.merge(df_negative_defi,df_negative_Atom_percentage,on='ID', how="inner")
    # df_negative_fingerprints["label"] = 0

    # # 合并正负样本的数据
    # df = pd.concat([df_positive_descriptor,df_negative_descriptor], ignore_index=True)

    # #查找非数字值并替换
    # for col in df.columns:
    #     df[col] = pd.to_numeric(df[col], errors='coerce')
    #     if pd.api.types.is_numeric_dtype(df[col]):
    #         #计算缺失值、0值、inf值，如果>85%就删除此列
    #         nan_percent = (df[col].isnull().sum() / len(df[col])) * 100
    #         zero_percent = ((df[col] == 0).sum(axis=0) / len(df[col]))* 100
    #         # one_percent = ((df[col] == 1).sum(axis=0) / len(df[col]))* 100
    #
    #         if nan_percent > 85.0 or zero_percent > 85.0 :
    #             df.drop(col, axis=1, inplace=True)
    #             continue
    #         # # 计算该列的均值
    #         # mean = df[col].mean()
    #         # # 将NaN替换为均值
    #         # df[col] = df[col].fillna(mean)
    #
    # #中位数填充
    # df.fillna(df.median(), inplace = True)
    #
    # #k近邻填充
    # # imputer = KNNImputer(n_neighbors=5)
    # # X = imputer.fit_transform(df)
    # # df = pd.DataFrame(X)
    #
    # # 打乱数据顺序
    # df = df.sample(frac=1).reset_index(drop=True)
    #
    #
    # # #查找无穷值
    # df_inf = np.isinf(df)
    # df[df_inf] = 0
    #
    # # 提取特征和标签
    # X = df.iloc[:,1:-1].values
    # y = df.iloc[:, -1].values
    # return X,y,df
    #

def F_Score(X,y):

    columns_df = X.columns
    X = X.values

    # 创建选择器对象，选择所有特征
    selector = SelectKBest(score_func=f_classif,k = 'all')

    # 使用选择器对象对特征进行选择
    X_new = selector.fit_transform(X, y)

    # 获取特征的F-Score
    f_scores = selector.scores_

    # 创建一个DataFrame来存储特征名和对应的F-Score
    feature_scores = pd.DataFrame({'Features': columns_df, 'F-Score': f_scores})

    # 按照F-Score从大到小排列
    feature_scores_sorted = feature_scores.sort_values(by='F-Score', ascending=False)
    feature_scores_50 = feature_scores_sorted[:50]
    return feature_scores_50

    # 将排序后的特征输出到csv文件中
    # feature_scores_sorted.to_csv('fingerprints_F_Score.csv',index = False)

    # # 绘制特征的F-Score图
    # plt.bar(range(len(f_scores)), f_scores)
    # plt.xlabel('Feature Index')
    # plt.ylabel('F-Score')
    # plt.title('F-Score of Features')
    # plt.show()

def boruta(X,y):

    columns_df = X.columns
    X = X.values

    rf = RandomForestClassifier(n_jobs=-1, max_depth=5)

    # define Boruta feature selection method
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

    # find all relevant features - 5 features should be selected
    feat_selector.fit(X, y)

    # check ranking of features
    features_importance = feat_selector.ranking_

    # call transform() on X to filter it down to selected features
    # X_filtered = feat_selector.transform(X)
    # print(X_filtered)

    df_features_importance = pd.DataFrame({'Features':columns_df,'importance':features_importance}).sort_values('importance',ascending = False)
    return df_features_importance.iloc[:, :50]
    # df_features_importance.to_csv('fingerprints_boruta.csv',index = False)

# def sfs():
#
#     clf = SVC(kernel='rbf', probability=True, gamma="auto")
#
#     sfs = SFS(clf,
#               k_features= 'best',
#               forward=True,
#               floating=False,
#               verbose=1,
#               scoring='roc_auc',
#               cv=5,
#               n_jobs=-1
#               )
#     sfs = sfs.fit(X_train, y_train)
#
#     # 获取最佳特征子集
#     selected_features = list(sfs.k_feature_idx_)
#
#     # 画出AUC随特征数量变化的图
#     plot_sfs(sfs.get_metric_dict(), kind='std_dev', figsize=(14, 8))
#     plt.title('top_SFS(std)', fontsize=25)
#     plt.xlabel("Number of features", fontsize=15)
#     plt.ylabel("Performance", fontsize=15)
#     plt.grid()
#     plt.savefig('feature_selection.png', dpi=110, format='png', bbox_inches='tight')


#读取训练集和测试集数据
# X_train,y_train = reading('train')
# X_test,y_test,df1 = reading('test')
# X_train = df.drop(labels = ['label','ID'], axis=1, inplace = False).values

# y = df['label']
# X = df.drop(labels = ['label','ID'], axis=1, inplace = False)
#
# # 使用F_Score对特征重要性进行排序
# F_Score(X,y)
# df_F_Score = pd.read_csv('fingerprints_F_Score.csv')
# df_F_Score_100 = df_F_Score.iloc[:100,:]
# columns_name = df_F_Score_100['Features'].tolist()
#
# # IFS特征选择
# best_score = 0
# best_subset = []
# auc_scores = []
# for i in range(10,len(columns_name) + 1, 10):
#     subset = columns_name[:i]
#     cv_scores = cross_val_score(XGBClassifier(), df[subset].values, y_train, scoring='roc_auc', cv=5)
#     score = cv_scores.mean()
#     auc_scores.append(score)
#
#     if score > best_score:
#         best_score = score
#         best_subset = subset
#     print(f'Number of features: {i},CV (+/- {np.std(cv_scores):.3f})')
#
# # 画出AUC随特征数量变化的折线图
# plt.plot(range(10, 101, 10), auc_scores, marker='o')
# plt.xticks(np.arange(10, 101, step=10))
# plt.yticks(np.arange(0.800, 1.000, step=0.05))
# plt.xlabel('Number of features')
# plt.ylabel('AUC')
#
# # 标记最大值点
# best_num_features = len(best_subset)
# plt.scatter(best_num_features, best_score, color='red', s=100)
# plt.annotate(text=f'Best Score: ({best_num_features:.3f},{best_score:.3f})', xy=(best_num_features, best_score),
#              xytext=(best_num_features - 30.0, best_score + 0.010))
# plt.vlines(x=best_num_features, ymin=0.800, ymax=best_score, linestyle='dashed', color='blue')
# plt.show()

# X_test = df1[columns_name].values



# Feature_pcc[column] = pcc1
# # features = sorted(Feature_pcc.items(),key=lambda x: x[1], reverse=True)
# # df_pearson = pd.DataFrame(features,columns = ['Features','pcc'])
# 使用boruta对特征进行排序
# boruta(X,y)
# df_boruta = pd.read_csv('descriptor_boruta.csv')
# df_boruta_100 = df_boruta.iloc[:100,:]
# columns_name = df_boruta_100['Features'].tolist()
# # X_test = df[columns_name]
# #
# # # IFS特征选择
# best_score = 0
# best_subset = []
# auc_scores = []
# for i in range(10,len(columns_name) + 1, 10):
#     subset = columns_name[:i]
#     cv_scores = cross_val_score(XGBClassifier(), df[subset].values, y_train, scoring='roc_auc', cv=5)
#     score = cv_scores.mean()
#     auc_scores.append(score)
#
#     if score > best_score:
#         best_score = score
#         best_subset = subset
#     print(f'Number of features: {i},CV (+/- {np.std(cv_scores):.3f})')
#
# # # 画出AUC随特征数量变化的折线图
# plt.plot(range(10, 101, 10), auc_scores, marker='o')
# plt.xticks(np.arange(10, 101, step=10))
# plt.yticks(np.arange(0.550, 1.000, step=0.05))
# plt.xlabel('Number of features')
# plt.ylabel('AUC')
# #
# # # 标记最大值点
# best_num_features = len(best_subset)
# plt.scatter(best_num_features, best_score, color='red', s=100)
# plt.annotate(text=f'Best Score: ({best_num_features:.3f},{best_score:.3f})', xy=(best_num_features, best_score),
#              xytext=(best_num_features - 30.0, best_score + 0.010))
# plt.vlines(x=best_num_features, ymin=0.550, ymax=best_score, linestyle='dashed', color='blue')
# plt.show()

# X_train = df[columns_name].values
# X_test = df1[columns_name].values

# 使用SFS做特征选择
# sfs()

# LightGBM + IFS
# bestsubset_lightGBM_IFS = lightGBM_IFS()
# # np.save('bestsubset_lightGBM+IFS', bestsubset_lightGBM_IFS)
# # bestsubset_lightGBM_IFS = np.load('bestsubset_lightGBM+IFS.npy')
# # bestsubset_lightGBM_IFS = bestsubset_lightGBM_IFS.tolist()
# #
# X_train = X_train[:, bestsubset_lightGBM_IFS]
# X_test = X_test[:, bestsubset_lightGBM_IFS]


# df_positive_NT_CT_ = {}
# df_negative_NT_CT_ = {}
# for i in range(4, 6):

    # 读取C末端
    # df_positive_CT[i] = pd.read_csv(f"./CT-k NT-k/AAC_CT{i}_pos_train.csv")
    # df_positive_CT[i]['label'] = 1
    # df_negative_CT[i] = pd.read_csv(f"./CT-k NT-k/AAC_CT{i}_neg_train.csv")
    # df_negative_CT[i]['label'] = 0
    # print(f'CT{i}')

    # 读取N末端
    # df_positive_NT[i] = pd.read_csv(f"./CT-k NT-k/AAC_NT{i}_pos_train.csv")
    # df_positive_NT[i]['label'] = 1
    # df_negative_NT[i] = pd.read_csv(f"./CT-k NT-k/AAC_NT{i}_neg_train.csv")
    # df_negative_NT[i]['label'] = 0
    # print(f'NT{i}')

# 读取NT-CT正负样本
# for i in range(2,6):
#     print(f'NT_CT_{i}')
# df_positive_NT_CT_5 = pd.read_csv(f'pos_train_binary_NT_CT_5.csv')
# df_positive_NT_CT_5['label'] = 1
# df_negative_NT_CT_5 = pd.read_csv(f'neg_train_binary_NT_CT_5.csv')
# df_negative_NT_CT_5['label'] = 0
#
# # 合并正负样本
# df = pd.concat([df_positive_NT_CT_5,df_negative_NT_CT_5], ignore_index=True)
# # df = df.drop('ID',axis=1, inplace = False)
#
# # 打乱样本
# df = df.sample(frac=1).reset_index(drop=True)
#
# # 提取特征和标签
# X_train = df.iloc[:,0:-1].values
# y_train = df.iloc[:, -1].values

# 定义模型
# models = [
#     KNeighborsClassifier(),
#     LogisticRegression(),
#     MLPClassifier(),
#     RandomForestClassifier(),
#     SVC(probability=True),
#     XGBClassifier()
# ]
#
# # 定义评估指标
# def sensitivity(y_true, y_pred):
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#     return tp / (tp + fn)
#
# def specificity(y_true, y_pred):
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#     return tn / (tn + fp)
#
# scoring = {
#     'acc': make_scorer(accuracy_score),
#     'mcc': make_scorer(matthews_corrcoef),
#     'auc': 'roc_auc',
#     'sen': make_scorer(sensitivity),
#     'spe': make_scorer(specificity)
# }
#
# # 使用五折交叉验证评估模型
# for model in models:
#     print(model)
#     for score_name, score_func in scoring.items():
#         cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=score_func)
#         print(f'CV {score_name}: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores):.3f})')

# 在测试集上测试模型
# for model in models:
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     y_proba = model.predict_proba(X_test)[:, 1]
#     print(model)
#     print(f'Test sen: {sensitivity(y_test, y_pred):.3f}')
#     print(f'Test spe: {specificity(y_test, y_pred):.3f}')
#     print(f'Test acc: {accuracy_score(y_test, y_pred):.3f}')
#     print(f'Test mcc: {matthews_corrcoef(y_test, y_pred):.3f}')
#     print(f'Test auc: {roc_auc_score(y_test, y_proba):.3f}')


