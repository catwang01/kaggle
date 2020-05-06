#!/usr/bin/env python

# coding: utf-8
import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from utils import plotAuc, getFreq, fillnaInplace, dropInplace
from path import *
import json
import numpy as np
import argparse

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", type=bool, default=False, help="whether verbose or not")
args = parser.parse_args()

beh = pd.read_csv(behtestPath)
trd = pd.read_csv(trdtestPath)
tag = pd.read_csv(tagtestPath)

# 1000+的人没有交易记录
assert trd.id.unique().shape[0] == 4787
assert tag.id.unique().shape[0] == 6000

def printCount(df):
    if args.verbose:
        print(df.value_counts(dropna=False))

def processingBeh(data, features, mappings):
    for stage in ['train', 'test']:
        df = data[stage]['beh']
        if stage=='train': df.drop(columns=['flag'], inplace=True)
        data[stage]['newbeh'] = df.groupby(['id', 'page_no'], as_index=False).count()
        data[stage]['newbeh'] = data[stage]['newbeh'].pivot(values='page_tm', columns='page_no', index='id').fillna(0)

    # 29 个features
    features.extend((data[stage]['newbeh'].columns.values))
    mappings['float32'].extend(data[stage]['newbeh'].columns.values)

def processingTrd(data, features, mappings):
    for stage in ['train', 'test']:
        df = data[stage]['trd']

        # 说明 trd 中的一个id只有一个flag
        # tmp1 = data['train']['trd'][['id', 'flag']].drop_duplicates()
        # assert tmp1.shape[0]== data['train']['trd']['id'].unique().shape[0]
        # # 说明 trd 中的 id 和 tag 中对应的 id 对应的flag 是相同的。
        # merged = pd.merge(data['train']['tag'][['id', 'flag']], tmp1, on='id')
        # assert (merged.flag_x != merged.flag_y).sum() == 0
        # # 结论： trd 中的flag和tag中的tag是相同的，可以删除

        if stage == 'train':
            df.drop(columns=['flag'], inplace=True)

        # # 先暴力求个平均
        # data[stage]['newtrd'] = df.groupby('id')['cny_trx_amt'].mean()
        # data[stage]['newtrd'] = data[stage]['newtrd'].reset_index()
        # features.append('cny_trx_amt')

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
        labels = []
        for idx in ['Dat_Flg1_Cd', 'Dat_Flg3_Cd', 'Trx_Cod1_Cd']:
            labels.append(set(df[idx]))


        allCombinations =  set()
        for x in labels[0]:
            for y in labels[1]:
                for z in labels[2]:
                    allCombinations.add(x + y + str(z))


        data[stage]['newtrd'] = df.groupby(['id', 'Dat_Flg1_Cd', 'Dat_Flg3_Cd', 'Trx_Cod1_Cd'], as_index=False).agg(np.mean)
        data[stage]['newtrd']['newcol'] = data[stage]['newtrd']['Dat_Flg1_Cd'].str.cat(data[stage]['newtrd']['Dat_Flg3_Cd']).str.cat(data[stage]['newtrd']['Trx_Cod1_Cd'].apply(str))
        data[stage]['newtrd'] = data[stage]['newtrd'].pivot(columns='newcol', values='cny_trx_amt', index='id').fillna(0)

        for idx in allCombinations - set(data[stage]['newtrd']) - {'id'}:
            data[stage]['newtrd'][idx] = np.nan

        data[stage]['newtrd'] = data[stage]['newtrd'].fillna(0)

        # 交易金额
        features.extend(data[stage]['newtrd'].columns.values)
        mappings['float32'].extend(data[stage]['newtrd'].columns.values)

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
        df['atdd_type'].replace({r'\N': '2'}, inplace=True)
        fillnaInplace(df['atdd_type'], '#')
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


def castType(train, test, mappings):
    for dtype in mappings:
        for col in train.columns:
            if col in mappings[dtype]:
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

def process_data():
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
    index = np.array([True] * data['train']['tag'].shape[0])
    target = ['flag']
    mappings = {
        'encode': [],
        'int16': [],
        'float32': []
    }

    processors = [processingBeh, processingTag, processingTrd]
    for processor in processors:
        processor(data, features, mappings)

    features = list(set(features))
    for k in mappings:
        mappings[k] = list(set(mappings[k]))

    for table in ['tag', 'trd']:
        castType(data['train'][table], data['test'][table], mappings)

    train = pd.merge(data['train']['tag'], data['train']['newtrd'], on='id', how='left')
    train = pd.merge(train, data['train']['newbeh'], on='id', how='left')
    train.fillna(0, inplace=True)
    encodedFeatures = train[mappings['encode']]
    y = train['flag']
    train.drop(columns=mappings['encode'] + ['id', 'flag'], inplace=True)
    onehot = OneHotEncoder()
    trainarray = np.c_[train.values, onehot.fit_transform(encodedFeatures).toarray()]


    test = pd.merge(data['test']['tag'], data['test']['newtrd'], on='id', how='left')
    test = pd.merge(test, data['test']['newbeh'], on='id', how='left')
    test.fillna(0, inplace=True)
    encodedFeatures = test[mappings['encode']]
    test.drop(columns=mappings['encode'] + ['id'], inplace=True)
    testarray = np.c_[test.values, onehot.transform(encodedFeatures).toarray()]


    X_train, X_val, y_train, y_val = train_test_split(trainarray, y)
    data = {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "X_test": testarray,
    }
    np.savez_compressed(processedDataPath, **data)
    # train.to_pickle(processed_train_tag_feature_path)
    # test.to_pickle(processed_test_tag_feature_path)
    #
    # with open(jsonPath, 'w') as f:
    #     json.dump([features, target], f)
    #

if __name__ == '__main__':
    process_data()
