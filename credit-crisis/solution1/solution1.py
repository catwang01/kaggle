#!/usr/bin/env python
# coding: utf-8


# ![image.png](attachment:image.png)

# In[98]:

import re
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve
import json
import argparse

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
import xgboost as xgb
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-u", "--update", type=int, default=True, help="display a square of a given number")
parser.add_argument("-v", "--verbose", type=bool, default=False , help="whether verbose or not")
args = parser.parse_args()

#### utils

def plotAuc(y, yhat, xlabel=None):
    fpr, tpr, thresholds = roc_curve(y, yhat, pos_label=1)
    plt.plot(fpr, tpr)
    plt.title("auc: {}".format(auc(fpr, tpr)))
    if xlabel: plt.xlabel(xlabel)
    plt.show()

def getLatestVer(pattern):
    for file in os.listdir():
        ret = re.search(pattern, file)
        if ret is not None:
            i = int(ret.group(1))
            return i
    return 0

def getNextVer(pattern):
    return getLatestVer(pattern) + 1

def getFreq(df):
    tmp = df.value_counts()
    return tmp / tmp.sum()

def fillnaInplace(df, val):
    df.fillna(val, inplace=True)


def dropInplace(df, condition):
    df.drop(df.index[condition], inplace=True)

def printCount(df):
    if args.verbose:
        print(df.value_counts(dropna=False))


root = os.path.dirname(__file__)
processed_train_feature_path = os.path.join(root, 'processed_train_feature.pkl')
processed_test_feature_path = os.path.join(root, 'processed_test_feature.pkl')
jsonPath = os.path.join(root, "colNames.json")

tagtrainPath = '/Users/ed/kaggle/credit-crisis/train/训练数据集_tag.csv'
behtrainPath = '/Users/ed/kaggle/credit-crisis/train/训练数据集_beh.csv'
trdtrainPath = '/Users/ed/kaggle/credit-crisis/train/训练数据集_trd.csv'

tagtestPath = '/Users/ed/kaggle/credit-crisis/test/评分数据集_tag.csv'
behtestPath = '/Users/ed/kaggle/credit-crisis/test/评分数据集_beh.csv'
trdtestPath = '/Users/ed/kaggle/credit-crisis/test/评分数据集_trd.csv'


def load_data():
    if os.path.exists(processed_test_feature_path):
        train = pd.read_pickle(processed_train_feature_path)
        test = pd.read_pickle(processed_test_feature_path)
        with open(jsonPath) as f:
            features, target = json.load(f)
    else:
        train, test, features, target = _load_data()
    return train, test, features, target

def _load_data():

    data = {}
    types = {'atdd_type': str}

    data['train'] = {
        'beh': pd.read_csv(behtrainPath, dtype=types),
        'tag': pd.read_csv(tagtrainPath, dtype=types),
        'trd': pd.read_csv(trdtrainPath, dtype=types)
    }

    data['test'] = {
        'beh': pd.read_csv(behtestPath, dtype=types),
        'tag': pd.read_csv(tagtestPath, dtype=types),
        'trd': pd.read_csv(trdtestPath, dtype=types)
    }


    features = []
    target = ['flag']
    mappings = {
        'encode': [],
        'int16': [],
        'float16': []
    }

    def processingBeh(data, features, mappings):
        pass
        # for stage in ['train', 'test']:
        #     df = data[stage]['beh']
        #     data[stage]['newbeh'] = df.groupby('id')['flag'].agg(np.sum)
        #     data[stage]['newbeh'].reset_index()
        # features.append('count')
        # mappings['int16'].append('count')

    def processingTrd(data, features, mappings):
        for stage in ['train', 'test']:
            df = data[stage]['trd']

            # # 说明 trd 中的一个id只有一个flag
            # tmp1 = data['train']['trd'][['id', 'flag']].drop_duplicates()
            # assert tmp1.shape[0]== data['train']['trd']['id'].unique().shape[0]
            # # 说明 trd 中的 id 和 tag 中对应的 id 对应的flag 是相同的。
            # merged = pd.merge(data['train']['tag'][['id', 'flag']], tmp1, on='id')
            # assert (merged.flag_x != merged.flag_y).sum() == 0
            # # 结论： trd 中的flag和tag中的tag是相同的，可以删除

            if stage == 'train':
                df.drop(columns=['flag'], inplace=True)
            # 先暴力求个平均
            data[stage]['newtrd'] = df.groupby('id')['cny_trx_amt'].mean()
            data[stage]['newtrd'] = data[stage]['newtrd'].reset_index()
            features.append('cny_trx_amt')

            # printCount(df["Dat_Flg1_Cd"])
            # mappings['encode'].append('Dat_Flg1_Cd')
            # features.append('Dat_Flg1_Cd')
            #
            # printCount(df["Dat_Flg3_Cd"])
            # mappings['encode'].append('Dat_Flg3_Cd')
            # features.append('Dat_Flg3_Cd')
            #
            # printCount(df['Trx_Cod1_Cd'])
            # mappings['encode'].append("Trx_Cod1_Cd")
            # features.append("Trx_Cod1_Cd")

            # 收支二级分类代码
            # 这个 test 比 train 大。先 drop 掉
            # printCount(df['Trx_Cod2_Cd'])
            # features.append("Trx_Cod2_Cd")
            # mappings['encode'].append("Trx_Cod2_Cd")

            # trx_tm 交易时间现不算

            # 交易金额
            # printCount(df['cny_trx_amt'])
            # mappings['float16'].append("cny_trx_amt")
            # features.append("cny_trx_amt")

    def processingTag(data, features, mappings):
        for stage in ['train', 'test']:
            df = data[stage]['tag']

            if stage == "train":
                printCount(df['flag'])
                mappings['int16'].append("flag")

            # 持有招行借记卡张数
            printCount(df['cur_debit_cnt'])
            features.append('cur_debit_cnt')
            mappings['int16'].append("cur_debit_cnt")

            # 持有招行信用卡张数
            printCount(df['cur_credit_cnt'])
            features.append("cur_credit_cnt")
            mappings['int16'].append('cur_credit_cnt')

            # 持有招行借记卡天数
            printCount(df["cur_debit_min_opn_dt_cnt"])
            features.append("cur_debit_min_opn_dt_cnt")
            mappings['int16'].append("cur_debit_min_opn_dt_cnt")

            # 持有招行信用卡天数
            printCount(df["cur_credit_min_opn_dt_cnt"])
            features.append("cur_credit_min_opn_dt_cnt")
            mappings['int16'].append("cur_credit_min_opn_dt_cnt")

            # 招行借记卡持卡最高等级代码
            printCount(df["cur_debit_crd_lvl"])
            features.append("cur_debit_crd_lvl")
            mappings['int16'].append("cur_debit_crd_lvl")

            # 招行信用卡持卡最高等级代码
            # 出现 \N，当作一个新的类，不作处理
            printCount(df["hld_crd_card_grd_cd"])
            # dropInplace(df, df['hld_crd_card_grd_cd']==r'\N')
            features.append("hld_crd_card_grd_cd")
            mappings['encode'].append("hld_crd_card_grd_cd")

            # 信用卡活跃标识
            printCount(df['crd_card_act_ind'])
            # 出现 \N，当作一个新的类，不作处理
            # dropInplace(df, df['crd_card_act_ind']==r'\N')
            features.append('crd_card_act_ind')
            mappings['encode'].append('crd_card_act_ind')

            # 最近一年信用卡消费金额分层
            # dropInplace(df, df['l1y_crd_card_csm_amt_dlm_cd']==r'\N')
            printCount(df['l1y_crd_card_csm_amt_dlm_cd'])
            features.append('l1y_crd_card_csm_amt_dlm_cd')
            mappings['encode'].append('l1y_crd_card_csm_amt_dlm_cd')

            # 信用卡还款方式
            printCount(df['atdd_type'])
            df['atdd_type'].replace({'atdd_type': 2}, inplace=True)
            fillnaInplace(df['atdd_type'], '#')

            # df['atdd_type'] = df['atdd_type'].astype('str')
            # df.loc[df["atdd_type"] == r'\N', "atdd_type"] = 2
            # df['atdd_type'] = df['atdd_type'].astype('float')
            features.append('atdd_type')
            mappings['encode'].append('atdd_type')

            # 信用卡永久信用额度分层
            printCount(df['perm_crd_lmt_cd'])
            features.append('perm_crd_lmt_cd')
            mappings['encode'].append('perm_crd_lmt_cd')

            # 年龄
            printCount(df['age'])
            features.append('age')
            mappings['int16'].append('age')

            # 性别
            printCount(df['gdr_cd'])
            # 出现 \N 不做处理
            # dropInplace(df, df['gdr_cd']==r'\N')
            features.append('gdr_cd')
            mappings['encode'].append('gdr_cd')

            # 婚姻
            printCount(df['mrg_situ_cd'])
            features.append('mrg_situ_cd')
            mappings['encode'].append('mrg_situ_cd')

            # 教育程度
            printCount(df['edu_deg_cd'])
            features.append('edu_deg_cd')
            mappings['encode'].append('edu_deg_cd')
            # NaN 如果处理？ 这里变成另一类
            fillnaInplace(df['edu_deg_cd'], '#')

            # 学历
            printCount(df['acdm_deg_cd'])
            fillnaInplace(df['acdm_deg_cd'], '#')
            features.append('acdm_deg_cd')
            mappings['encode'].append('acdm_deg_cd')
            # dropInplace(df, df['acdm_deg_cd'].isna())

            # 学位
            printCount(df['deg_cd'])
            features.append("deg_cd")
            mappings['encode'].append("deg_cd")
            fillnaInplace(df['deg_cd'], '#')

            # 工作年限
            printCount(df['job_year'])
            features.append('job_year')
            mappings['encode'].append('job_year')

            # 工商标识
            printCount(df['ic_ind'])
            features.append('ic_ind')
            mappings['encode'].append('ic_ind')

            # 法人或股东标识
            printCount(df['fr_or_sh_ind'])
            features.append('fr_or_sh_ind')
            mappings['encode'].append('fr_or_sh_ind')

            # 下载并登录招行APP标识
            printCount(df['dnl_mbl_bnk_ind'])
            features.append('dnl_mbl_bnk_ind')
            mappings['encode'].append('dnl_mbl_bnk_ind')

            # 下载并绑定掌上生活标识
            printCount(df['dnl_bind_cmb_lif_ind'])
            features.append("dnl_bind_cmb_lif_ind")
            mappings['encode'].append("dnl_bind_cmb_lif_ind")

            # 有车一族标识
            printCount(df['hav_car_grp_ind'])
            features.append('hav_car_grp_ind')
            mappings['encode'].append('hav_car_grp_ind')

            # 有房一族标识
            printCount(df['hav_hou_grp_ind'])
            features.append('hav_hou_grp_ind')
            mappings['encode'].append('hav_hou_grp_ind')

            # 近6个月代发工资标识
            printCount(df['l6mon_agn_ind'])
            features.append('l6mon_agn_ind')
            mappings['encode'].append('l6mon_agn_ind')

            # 首次代发工资距今天数
            printCount(df['frs_agn_dt_cnt'])
            df['frs_agn_dt_cnt'].replace({r'\N': 0}, inplace=True)
            features.append('frs_agn_dt_cnt')
            mappings['int16'].append('frs_agn_dt_cnt')

            # 有效投资风险评估标识
            printCount(df['vld_rsk_ases_ind'])
            df['vld_rsk_ases_ind'].replace({r'\N': 0}, inplace=True)
            features.append('vld_rsk_ases_ind')
            mappings['int16'].append('vld_rsk_ases_ind')

            # 用户理财风险承受能力等级代码
            printCount(df['fin_rsk_ases_grd_cd'])
            features.append('fin_rsk_ases_grd_cd')
            mappings['encode'].append('fin_rsk_ases_grd_cd')

            # 投资强风评等级类型代码
            printCount(df['confirm_rsk_ases_lvl_typ_cd'])
            features.append('confirm_rsk_ases_lvl_typ_cd')
            mappings['encode'].append('confirm_rsk_ases_lvl_typ_cd')

            # 用户投资风险承受级别
            printCount(df['cust_inv_rsk_endu_lvl_cd'])
            features.append('cust_inv_rsk_endu_lvl_cd')
            mappings['encode'].append('cust_inv_rsk_endu_lvl_cd')

            # 近6个月月日均AUM分层
            printCount(df['l6mon_daim_aum_cd'])
            features.append('l6mon_daim_aum_cd')
            mappings['encode'].append('l6mon_daim_aum_cd')

            # 总资产级别代码
            printCount(df['tot_ast_lvl_cd'])
            features.append('tot_ast_lvl_cd')
            mappings['encode'].append('tot_ast_lvl_cd')

            # 潜力资产等级代码
            printCount(df['pot_ast_lvl_cd'])
            features.append('pot_ast_lvl_cd')
            mappings['encode'].append('pot_ast_lvl_cd')

            # 本年月均代发金额分层
            printCount(df['bk1_cur_year_mon_avg_agn_amt_cd'])
            features.append('bk1_cur_year_mon_avg_agn_amt_cd')
            mappings['encode'].append('bk1_cur_year_mon_avg_agn_amt_cd')

            # 近12个月理财产品购买次数
            printCount(df['l12mon_buy_fin_mng_whl_tms'])
            df['l12mon_buy_fin_mng_whl_tms'].replace({r'\N': '0'}, inplace=True)
            features.append('l12mon_buy_fin_mng_whl_tms')
            mappings['int16'].append('l12mon_buy_fin_mng_whl_tms')

            # 近12个月基金购买次数
            printCount(df['l12_mon_fnd_buy_whl_tms'])
            df['l12_mon_fnd_buy_whl_tms'].replace({r'\N': '0'}, inplace=True)
            features.append('l12_mon_fnd_buy_whl_tms')
            mappings['int16'].append('l12_mon_fnd_buy_whl_tms')

            # 近12个月保险购买次数
            printCount(df['l12_mon_insu_buy_whl_tms'])
            df['l12_mon_insu_buy_whl_tms'].replace({r'\N': '0'}, inplace=True)
            features.append('l12_mon_insu_buy_whl_tms')
            mappings['int16'].append('l12_mon_insu_buy_whl_tms')

            # 近12个月黄金购买次数
            printCount(df['l12_mon_gld_buy_whl_tms'])
            df['l12_mon_gld_buy_whl_tms'].replace({r'\N': '0'}, inplace=True)
            features.append('l12_mon_gld_buy_whl_tms')
            mappings['int16'].append('l12_mon_gld_buy_whl_tms')

            # 贷款用户标识
            printCount(df['loan_act_ind'])
            features.append('loan_act_ind')
            mappings['encode'].append('loan_act_ind')

            # 个贷授信总额度分层
            printCount(df['pl_crd_lmt_cd'])
            features.append('pl_crd_lmt_cd')
            mappings['encode'].append('pl_crd_lmt_cd')

            # 30天以上逾期贷款的总笔数
            printCount(df['ovd_30d_loan_tot_cnt'])
            df['ovd_30d_loan_tot_cnt'].replace({r'\N': '0'}, inplace=True)
            features.append('ovd_30d_loan_tot_cnt')
            mappings['int16'].append('ovd_30d_loan_tot_cnt')

            # 历史贷款最长逾期天数
            printCount(df['his_lng_ovd_day'])
            df['his_lng_ovd_day'].replace({r'\N': '0'}, inplace=True)
            features.append('his_lng_ovd_day')
            mappings['int16'].append('his_lng_ovd_day')

    def mergeById(data):
        train = pd.merge(data['train']['tag'], data['train']['newtrd'], how='left', on='id')
        test = pd.merge(data['test']['tag'], data['test']['newtrd'], how='left', on='id')
        fillnaInplace(train['cny_trx_amt'], 0)
        fillnaInplace(test['cny_trx_amt'], 0)
        return train, test

    def castType(train, test, mappings):
        for dtype in mappings:
            for col in mappings[dtype]:
                if col == 'flag': continue
                if dtype == 'encode':
                    try:
                        encoder = LabelEncoder()
                        train[col] = encoder.fit_transform(train[col])
                        test[col] = encoder.transform(test[col])
                    except:
                        print("haha: encode " + col)
                else:
                    try:
                        train[col] = train[col].astype(dtype)
                        test[col] = test[col].astype(dtype)
                    except:
                        print("haha: " + col)

    processors = [processingBeh, processingTag, processingTrd]
    for processor in processors:
        processor(data, features, mappings)

    features = list(set(features))
    train, test = mergeById(data)
    castType(train, test, mappings)
    train.to_pickle(processed_train_feature_path)
    test.to_pickle(processed_test_feature_path)
    with open(jsonPath, 'w') as f:
        json.dump([features, target], f)
    return train, test, features, target

def _loadModel(versionNo):
    bst = xgb.Booster()
    bst.load_model('xgb-{}.model'.format(versionNo))
    return bst

def getModel(X_train, y_train, X_val, y_val):
    for file in os.listdir():
        searched = re.search('xgb-(\d).model', file)
        if searched is not None:
            return _loadModel(searched.group(1))
    if args.update == False:
        return _loadModel(searched.group(1))

    return _trainModel(X_train, y_train, X_val, y_val)

def _trainModel(X_train, y_train, X_val, y_val):
    params = {
        'objective': 'binary:logistic',
        "booster": "gbtree",
        "eta": 0.01,  # shrinkage
        "max_depth": 10,
        "subsample": 0.7,
        "colsample_bytree": 0.6,
        "silent": 1,
        "seed": 1301
    }
    num_boost_round = 1000

    print("Train a XGBoost model")
    dtrain = xgb.DMatrix(X_train, y_train)
    dval = xgb.DMatrix(X_val, y_val)

    watchlist = [(dtrain, 'train'), (dval, 'val')]
    bst = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=100, verbose_eval=True)


    y_train_hat = bst.predict(dtrain)

    plotAuc(y_train, y_train_hat, xlabel='train')

    y_val_hat = bst.predict(dval)
    plotAuc(y_val, y_val_hat, xlabel='validation')

    bst.save_model('xgb-{}.model'.format(getNextVer("xgb-(\d).model")))

    return bst

######### train

train, test, features, target = load_data()
X_train, X_val, y_train, y_val = train_test_split(train[features], train[target], test_size=0.05, random_state=10)
bst = getModel(X_train, y_train, X_val, y_val)

dtest = xgb.DMatrix(test[features])
yhat = bst.predict(dtest)

output = pd.DataFrame({'id': test.id, 'flag': yhat})
output = output[['id','flag']]
output.to_csv('output-{}.txt'.format(getNextVer('output-(\d).txt')), index=False, columns=None, header=False, sep='\t')
