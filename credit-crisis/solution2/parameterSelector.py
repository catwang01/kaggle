# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 16:30:03 2018

@author: jia.liu
"""

import time
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import cross_val_score

pd.set_option('display.max_columns', 20)

pd.set_option('display.width', 60)


class AdjustParam(object):
    '''
    model：最优模型
    ort_res: 正交实验结果
    param_res：各因素水平结果
    '''

    def __init__(self, X, y, scoring=None, cv=5):
        self.X = X
        self.y = y
        self.scoring = scoring
        self.cv = cv
        self.model = None
        self.ort_res = None
        self.param_res = None

    def run_ort(self, model, df_params):
        model_str = str(model)
        k = 0
        n = len(df_params.index)
        df_params['ort_res'] = 0
        df_params['ort_res_std'] = 0
        df_params['ort_train_time'] = 0
        for i in df_params.index:
            k += 1
            params_li = list(map(lambda x: x + '=' + repr(df_params.loc[i, x]), df_params.columns[:-3]))
            param_str = ', '.join(params_li)
            print(param_str)
            for re_p in params_li:
                p_val = re_p.split('=')[0]
                model_str = re.sub('%s=[^,)]*' % p_val, re_p, model_str)
            model_ = eval(model_str)
            t1 = time.time()
            cv_score = cross_val_score(model_, self.X, self.y, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
            err = np.sqrt(-cv_score)
            res = np.mean(err)
            res_std = np.std(cv_score)
            t2 = int(time.time() - t1)
            print('res: %f, time: %d, num: %d/%d' % (res, t2, k, n))
            df_params.loc[i, 'ort_res'] = res
            df_params.loc[i, 'ort_res_std'] = res_std
            df_params.loc[i, 'ort_train_time'] = t2
        self.ort_res = df_params
        # 筛选最优值
        res_li = list(map(lambda x: df_params.groupby(x).ort_res.mean().argmin(), df_params.columns[:-3]))
        param_li = df_params.columns[:-3]
        param_opt = list(map(lambda x: '='.join(x), zip(param_li, map(repr, res_li))))
        # 最优RF模型
        for re_p in param_opt:
            p_val = re_p.split('=')[0]
            model_str = re.sub('%s=[^,)]*' % p_val, re_p, model_str)
        self.model = eval(model_str)

        temp_li = []
        for i in df_params.columns[:-3]:
            temp = df_params.groupby(i).mean()[['ort_res', 'ort_res_std', 'ort_train_time']].sort_values(
                ['ort_res', 'ort_res_std', 'ort_train_time'])
            temp = temp.rename(index=lambda x: str(i) + '=' + repr(x))
            temp_li.append(temp)
        self.param_res = pd.concat(temp_li)


# %%
if __name__ == '__main__':
    from xgboost.sklearn import XGBRegressor
    from sklearn.model_selection import train_test_split
    from my_ort import ORT


    def rmsle_cv(model, n_folds=5):
        rmse = np.sqrt(-cross_val_score(model, X.values, y, scoring="neg_mean_squared_error", cv=n_folds))
        return rmse


    X = pd.read_csv('X.csv')
    y = pd.read_csv('y.csv')
    training_X, validation_X, training_y, validation_y = train_test_split(X, y, test_size=0.2, random_state=17)
    xgb_model = XGBRegressor(n_jobs=-1)
    # 构建正交实验表
    ort = ORT()
    # 输入参数数量，返回可选的正交表
    ort.seeSets(4, see_num=10)

    # 设计参数
    params = {'max_depth': [3],
              'gamma': [0.3, 0.4, 0.5],
              'colsample_bytree': [0.2],
              'n_estimators': [750],
              'subsample': [0.5, 0.7, 0.9],
              'reg_lambda': [1, 5, 10],
              'learning_rate': [0.01, 0.1, 0.3]}

    # 获取参数正交表
    param_df = ort.genSets(params, mode=2)
    # 训练模型
    a = AdjustParam(X, y, 'neg_mean_squared_error')
    a.run_ort(model=xgb_model, df_params=param_df)

    # 查看正交搜索自动选出的模型
    a.model
    # 查看所有的结果
    a.ort_res
    # 查看极差分析结果（可以根据正交实验的结果和训练时间，手动调整超参数）
    a.param_res

    # 正交分析选出的模型
    score = rmsle_cv(a.model)
    print("Averaged models error: {:.4f} ({:.4f})".format(score.mean(), score.std()))
    # rmse: 0.1384 (0.0148)
