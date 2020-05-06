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
        print("dtype:", df.dtype)

def processingBeh(data, features, mappings):
    for stage in ['train', 'test']:
        df = data[stage]['beh']
        if stage=='train': df.drop(columns=['flag'], inplace=True)
        data[stage]['newbeh'] = df.groupby(['id', 'page_no'], as_index=False).count()
        data[stage]['newbeh'] = data[stage]['newbeh'].pivot(values='page_tm', columns='page_no', index='id').fillna(0)

    # 29 个features
    features.extend((data[stage]['newbeh'].columns.values))
    data[stage]['newbeh'] = data[stage]['newbeh'].apply(lambda x: x.astype('float32'))

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

        ###### 收支二级分类代码
        # 这个 test 比 train 大。先 drop 掉
        # printCount(df['Trx_Cod2_Cd'])
        # features.append("Trx_Cod2_Cd")
        # mappings['encode'].append("Trx_Cod2_Cd")

        ####### trx_tm 交易时间先不算

        labels = []
        for idx in ['Dat_Flg1_Cd', 'Dat_Flg3_Cd']:
            labels.append(set(df[idx]))

        allCombinations =  set()
        for x in labels[0]:
            for y in labels[1]:
                allCombinations.add(x + y)

        data[stage]['trd']['cny_trx_amt'] = data[stage]['trd']['cny_trx_amt'].astype('float32')
        data[stage]['newtrd'] = df.groupby(['id', 'Dat_Flg1_Cd', 'Dat_Flg3_Cd'], as_index=False)['cny_trx_amt'].agg(np.mean)
        data[stage]['newtrd']['newcol'] = data[stage]['newtrd']['Dat_Flg1_Cd'].str.cat(data[stage]['newtrd']['Dat_Flg3_Cd'])
        data[stage]['newtrd'] = data[stage]['newtrd'].pivot(columns='newcol', values='cny_trx_amt', index='id').fillna(0)

        for idx in allCombinations - set(data[stage]['newtrd']) - {'id'}:
            data[stage]['newtrd'][idx] = np.nan

        data[stage]['newtrd'] = data[stage]['newtrd'].fillna(0)

        # 交易金额
        features.extend(data[stage]['newtrd'].columns.values)

def processingTag(data, features, mappings):
    for stage in ['test', 'train']:
        df = data[stage]['tag']

        if stage == "train":
            printCount(df['flag'])
            df['flag'] = df['flag'].astype('int16')

        # 持有招行借记卡张数
        printCount(df['cur_debit_cnt'])
        df['cur_debit_cnt'] = df['cur_debit_cnt'].astype("int8")
        features.append('cur_debit_cnt')

        # 持有招行信用卡张数
        printCount(df['cur_credit_cnt'])
        df['cur_credit_cnt'] = df['cur_credit_cnt'].astype("int8")
        features.append("cur_credit_cnt")

        # 添加特征： 没有借记卡却有信用卡的人
        df['has_credit_but_not_debit'] = (df['cur_debit_cnt'] == 0) & (df['cur_credit_cnt'] != 0)
        df['has_credit_but_not_debit'] = df['has_credit_but_not_debit'].astype('int8')
        features.append('has_credit_but_not_debit')
        mappings['encode'].append('has_credit_but_not_debit')

        # 持有招行借记卡天数
        printCount(df["cur_debit_min_opn_dt_cnt"])
        df["cur_debit_min_opn_dt_cnt"] = df["cur_debit_min_opn_dt_cnt"].astype("int16")
        # 分别代表半年以下、半年到1年、1年以上
        df["cur_debit_min_opn_dt_cnt"] = pd.cut(df["cur_debit_min_opn_dt_cnt"], bins=[-2, 183, 365, 5000], labels=False)
        features.append("cur_debit_min_opn_dt_cnt")

        # 持有招行信用卡天数
        printCount(df["cur_credit_min_opn_dt_cnt"])
        df["cur_credit_min_opn_dt_cnt"] = df["cur_credit_min_opn_dt_cnt"].astype("int16")
        # 分别代表半年以下、半年到1年、1年以上
        df["cur_credit_min_opn_dt_cnt"] = pd.cut(df["cur_credit_min_opn_dt_cnt"], bins=[-2, 183, 365, 5000], labels=False)
        features.append("cur_credit_min_opn_dt_cnt")

        # 招行借记卡持卡最高等级代码
        printCount(df["cur_debit_crd_lvl"])
        features.append("cur_debit_crd_lvl")
        mappings['encode'].append("cur_debit_crd_lvl")

        # 招行信用卡持卡最高等级代码
        # 出现 \N，在 train 中的drop掉；test中的填充为 -1
        printCount(df["hld_crd_card_grd_cd"])
        if stage == 'train':
            dropInplace(df, df['hld_crd_card_grd_cd']==r'\N')
        else:
            df['hld_crd_card_grd_cd'].replace({r'\N': '-1'}, inplace=True)
        # 转化为整数才可以cut
        df['hld_crd_card_grd_cd'] = df['hld_crd_card_grd_cd'].astype('int8')
        df['hld_crd_card_grd_cd'] = pd.cut(df['hld_crd_card_grd_cd'], bins=[-1, 10, 20, 100], include_lowest=True, labels=False)
        features.append("hld_crd_card_grd_cd")
        mappings['encode'].append("hld_crd_card_grd_cd")

        # 信用卡活跃标识
        printCount(df['crd_card_act_ind'])
        if stage=='train':
            dropInplace(df, df['crd_card_act_ind']==r'\N')
        else:
            df['crd_card_act_ind'].replace({r'\N': '0'}, inplace=True)
        features.append('crd_card_act_ind')
        mappings['encode'].append('crd_card_act_ind')

        # 最近一年信用卡消费金额分层
        printCount(df['l1y_crd_card_csm_amt_dlm_cd'])
        if stage == 'train':
            dropInplace(df, df['l1y_crd_card_csm_amt_dlm_cd']==r'\N')
        else:
            df['l1y_crd_card_csm_amt_dlm_cd'].replace({r'\N': '0'}, inplace=True)

        features.append('l1y_crd_card_csm_amt_dlm_cd')
        mappings['encode'].append('l1y_crd_card_csm_amt_dlm_cd')

        # 信用卡还款方式
        printCount(df['atdd_type'])
        df['atdd_type'].replace({r'\N': '2'}, inplace=True)
        fillnaInplace(df['atdd_type'], '2')
        features.append('atdd_type')
        mappings['encode'].append('atdd_type')

        # 信用卡永久信用额度分层
        printCount(df['perm_crd_lmt_cd'])
        if stage == 'train':
            dropInplace(df, df['perm_crd_lmt_cd']=='-1')
        else:
            df['perm_crd_lmt_cd'].replace({'-1':'0'}, inplace=True)
        features.append('perm_crd_lmt_cd')
        mappings['encode'].append('perm_crd_lmt_cd')

        ############### 年龄
        # 对年龄分桶
        printCount(df['age'])
        df['age'] = df['age'].astype('int8')
        df['age'].hist(); plt.show()
        df['newage'] = pd.cut(df['age'], bins=[0,20,30,40,60,100], labels=False)
        features.append('newage')
        mappings['encode'].append('newage')
        #################

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

        ########### 教育程度
        ### 低于500人的分成一类 newedu
        printCount(df['edu_deg_cd'])
        # NaN 如果处理？ 这里变成另一类
        df['edu_deg_cd'].replace({r"\N": '#'}, inplace=True)
        fillnaInplace(df['edu_deg_cd'], '#')

        tmp = df['edu_deg_cd'].value_counts()
        df['edu_deg_cd'].replace({idx: 'newedu' for idx in tmp.index[tmp <= 500]}, inplace=True)
        features.append('edu_deg_cd')
        mappings['encode'].append('edu_deg_cd')
        ###########

        # 学历
        printCount(df['acdm_deg_cd'])
        fillnaInplace(df['acdm_deg_cd'], 'G')
        features.append('acdm_deg_cd')
        mappings['encode'].append('acdm_deg_cd')

        # 学位
        printCount(df['deg_cd'])
        features.append("deg_cd")
        mappings['encode'].append("deg_cd")
        fillnaInplace(df['deg_cd'], '#')
        df['deg_cd'].replace({r'\N': '#', "D": '#'}, inplace=True)

        # 工作年限
        printCount(df['job_year'])
        df['job_year'].replace({r'\N': '0'}, inplace=True)
        df['job_year'].value_counts().hist(); plt.show()
        df['job_year'] = df['job_year'].astype(int)
        df['new_job_year'] = pd.cut(df['job_year'], bins=[-1, 10, 30, 100])
        features.append('new_job_year')
        mappings['encode'].append('new_job_year')

        # 工商标识
        printCount(df['ic_ind'])
        df['ic_ind'].replace({r"\N": '0'}, inplace=True)
        features.append('ic_ind')
        mappings['encode'].append('ic_ind')

        # 法人或股东标识
        printCount(df['fr_or_sh_ind'])
        df['fr_or_sh_ind'].replace({r"\N": '1'}, inplace=True)
        features.append('fr_or_sh_ind')
        mappings['encode'].append('fr_or_sh_ind')

        # 下载并登录招行APP标识 没有的话认为是0
        printCount(df['dnl_mbl_bnk_ind'])
        df['dnl_mbl_bnk_ind'].replace({r"\N": '0'}, inplace=True)
        features.append('dnl_mbl_bnk_ind')
        mappings['encode'].append('dnl_mbl_bnk_ind')

        # 下载并绑定掌上生活标识 没有的话认为是0
        printCount(df['dnl_bind_cmb_lif_ind'])
        df['dnl_bind_cmb_lif_ind'].replace({r"\N": '0'}, inplace=True)
        features.append("dnl_bind_cmb_lif_ind")
        mappings['encode'].append("dnl_bind_cmb_lif_ind")

        # 有车一族标识
        printCount(df['hav_car_grp_ind'])
        df['hav_car_grp_ind'] = df['hav_car_grp_ind'].astype(str)
        if stage == 'train':
            dropInplace(df, df['hav_car_grp_ind']==r'\N')
        else:
            df['hav_car_grp_ind'].replace({r'\N': '0'}, inplace=True)

        features.append('hav_car_grp_ind')
        mappings['encode'].append('hav_car_grp_ind')

        # 有房一族标识
        printCount(df['hav_hou_grp_ind'])
        df['hav_hou_grp_ind'] = df['hav_hou_grp_ind'].astype(str)
        if stage == 'train':
            dropInplace(df, df['hav_hou_grp_ind']==r'\N')
        else:
            df['hav_hou_grp_ind'].replace({r'\N': '0'}, inplace=True)

        features.append('hav_hou_grp_ind')
        mappings['encode'].append('hav_hou_grp_ind')

        # 近6个月代发工资标识
        printCount(df['l6mon_agn_ind'])
        if stage == 'train':
            dropInplace(df, df['l6mon_agn_ind']==r'\N')
        else:
            df['l6mon_agn_ind'].replace({r'\N': '0'}, inplace=True)

        features.append('l6mon_agn_ind')
        mappings['encode'].append('l6mon_agn_ind')

        ######### 首次代发工资距今天数
        # drop掉
        df.drop(columns=['frs_agn_dt_cnt'], inplace=True)

        # printCount(df['frs_agn_dt_cnt'])
        # df['frs_agn_dt_cnt'].replace({r'\N': 0}, inplace=True)
        # features.append('frs_agn_dt_cnt')
        # mappings['int16'].append('frs_agn_dt_cnt')

        ############

        ########## 有效投资风险评估标识
        printCount(df['vld_rsk_ases_ind'])
        if stage == 'train':
            dropInplace(df, df['vld_rsk_ases_ind']==r'\N')
        else:
            df['vld_rsk_ases_ind'].replace({r'\N': '0'}, inplace=True)
        features.append('vld_rsk_ases_ind')
        mappings['encode'].append('vld_rsk_ases_ind')

        ########## 用户理财风险承受能力等级代码
        printCount(df['fin_rsk_ases_grd_cd'])
        printCount(data['test']['tag']['fin_rsk_ases_grd_cd'])
        if stage == 'train':
            dropInplace(df, df['fin_rsk_ases_grd_cd']==r'\N')
        else:
            df['fin_rsk_ases_grd_cd'].replace({r'\N': '-1'}, inplace=True)
        df['fin_rsk_ases_grd_cd'] = df['fin_rsk_ases_grd_cd'].astype("int8")
        df['fin_rsk_ases_grd_cd'] = pd.cut(df['fin_rsk_ases_grd_cd'], bins=[-1, 0, 2, 3, 4, 20], include_lowest=True, labels=False)
        features.append('fin_rsk_ases_grd_cd')
        mappings['encode'].append('fin_rsk_ases_grd_cd')

        ##############

        # 投资强风评等级类型代码
        printCount(df['confirm_rsk_ases_lvl_typ_cd'])
        if stage == 'train':
            dropInplace(df, df['confirm_rsk_ases_lvl_typ_cd']==r'\N')
        else:
            df['confirm_rsk_ases_lvl_typ_cd'].replace({r'\N': '-1'}, inplace=True)
        features.append('confirm_rsk_ases_lvl_typ_cd')
        mappings['encode'].append('confirm_rsk_ases_lvl_typ_cd')

        # 用户投资风险承受级别
        printCount(df['cust_inv_rsk_endu_lvl_cd'])
        if stage=='train':
            dropInplace(df, df['cust_inv_rsk_endu_lvl_cd']=='9')
            dropInplace(df, df['cust_inv_rsk_endu_lvl_cd']==r'\N')
        else:
            df['cust_inv_rsk_endu_lvl_cd'].replace({r'\N': '1', '9': '8'}, inplace=True)
        features.append('cust_inv_rsk_endu_lvl_cd')
        mappings['encode'].append('cust_inv_rsk_endu_lvl_cd')

        # 近6个月月日均AUM分层
        printCount(df['l6mon_daim_aum_cd'])
        if stage == 'train':
            dropInplace(df, df['l6mon_daim_aum_cd']==r'\N')
        else:
            df['l6mon_daim_aum_cd'].replace({r'\N': '-1'}, inplace=True)
        df['l6mon_daim_aum_cd'].replace({'9': '8'}, inplace=True)
        df['l6mon_daim_aum_cd'] = df['l6mon_daim_aum_cd'].astype('int8')
        df['l6mon_daim_aum_cd'] = pd.cut(df['l6mon_daim_aum_cd'], bins=[-1, 6, 8], include_lowest=True)
        features.append('l6mon_daim_aum_cd')
        mappings['encode'].append('l6mon_daim_aum_cd')

        # 总资产级别代码
        printCount(df['tot_ast_lvl_cd'])
        if stage == 'train':
            dropInplace(df, df['tot_ast_lvl_cd']==r'\N')
        else:
            df['tot_ast_lvl_cd'].replace({r'\N': '-1'}, inplace=True)
        df['tot_ast_lvl_cd'] = df['tot_ast_lvl_cd'].astype('int8')
        df['tot_ast_lvl_cd'] = pd.cut(df['tot_ast_lvl_cd'], bins=[-1, 0, 2, 12], include_lowest=True)
        features.append('tot_ast_lvl_cd')
        mappings['encode'].append('tot_ast_lvl_cd')

        ########## 潜力资产等级代码

        printCount(df['pot_ast_lvl_cd'])
        if stage == 'train':
            dropInplace(df, df['pot_ast_lvl_cd']==r'\N')
        else:
            df['pot_ast_lvl_cd'].replace({"1": '-1', r'\N': '-1'}, inplace=True)
        features.append('pot_ast_lvl_cd')
        mappings['encode'].append('pot_ast_lvl_cd')

        ##############

        ######## 本年月均代发金额分层

        printCount(df['bk1_cur_year_mon_avg_agn_amt_cd'])
        df['bk1_cur_year_mon_avg_agn_amt_cd'] = df['bk1_cur_year_mon_avg_agn_amt_cd'].astype('int8')
        df['bk1_cur_year_mon_avg_agn_amt_cd'] = pd.cut(df['bk1_cur_year_mon_avg_agn_amt_cd'], bins=[-1, 1, 2, 10], include_lowest=True)
        features.append('bk1_cur_year_mon_avg_agn_amt_cd')
        mappings['encode'].append('bk1_cur_year_mon_avg_agn_amt_cd')

        ################

        ################## 合为一个
        # # 近12个月理财产品购买次数
        printCount(df['l12mon_buy_fin_mng_whl_tms'])
        df['l12mon_buy_fin_mng_whl_tms'].replace({r'\N': '0'}, inplace=True)
        df['l12mon_buy_fin_mng_whl_tms'] = df['l12mon_buy_fin_mng_whl_tms'].astype("int16")

        # 近12个月基金购买次数
        printCount(df['l12_mon_fnd_buy_whl_tms'])
        df['l12_mon_fnd_buy_whl_tms'].replace({r'\N': '0'}, inplace=True)
        df['l12_mon_fnd_buy_whl_tms'] = df['l12_mon_fnd_buy_whl_tms'].astype('int16')
        # features.append('l12_mon_fnd_buy_whl_tms')
        # mappings['int16'].append('l12_mon_fnd_buy_whl_tms')

        # 近12个月保险购买次数
        printCount(df['l12_mon_insu_buy_whl_tms'])
        df['l12_mon_insu_buy_whl_tms'].replace({r'\N': '0'}, inplace=True)
        df['l12_mon_insu_buy_whl_tms'] = df['l12_mon_insu_buy_whl_tms'].astype("int16")
        # features.append('l12_mon_insu_buy_whl_tms')
        # mappings['int16'].append('l12_mon_insu_buy_whl_tms')

        # 近12个月黄金购买次数
        printCount(df['l12_mon_gld_buy_whl_tms'])
        df['l12_mon_gld_buy_whl_tms'].replace({r'\N': '0'}, inplace=True)
        df['l12_mon_gld_buy_whl_tms'] = df['l12_mon_gld_buy_whl_tms'].astype("int16")
        # features.append('l12_mon_gld_buy_whl_tms')
        # mappings['int16'].append('l12_mon_gld_buy_whl_tms')

        df['l12_mon_tms'] = df[['l12mon_buy_fin_mng_whl_tms', 'l12_mon_fnd_buy_whl_tms', 'l12_mon_insu_buy_whl_tms', 'l12_mon_gld_buy_whl_tms']].sum(axis=1)
        df['l12_mon_tms'] = df['l12_mon_tms'].astype("int16")
        if args.verbose: df['l12_mon_tms'].hist(); plt.show()
        # 分桶
        df['l12_mon_tms'] = pd.cut(df['l12_mon_tms'], bins=[-1, 10, 10000], labels=False)
        features.append('l12_mon_tms')

        #################################

        ################## 贷款用户标识

        printCount(df['loan_act_ind'])
        if stage == 'train':
            dropInplace(df, df['loan_act_ind']==r'\N')
        else:
            df['loan_act_ind'].replace({r'\N': '0'}, inplace=True)
        features.append('loan_act_ind')
        mappings['encode'].append('loan_act_ind')

        ########## 个贷授信总额度分层

        printCount(df['pl_crd_lmt_cd'])
        df['pl_crd_lmt_cd'] = df['pl_crd_lmt_cd'].astype('int8')
        df['pl_crd_lmt_cd'] = pd.cut(df['pl_crd_lmt_cd'], bins=[-1,0, 10], include_lowest=True)
        features.append('pl_crd_lmt_cd')
        mappings['encode'].append('pl_crd_lmt_cd')

        ##################

        ################## 30天以上逾期贷款的总笔数
        # drop
        df.drop(columns=['ovd_30d_loan_tot_cnt'], inplace=True)
        # df['ovd_30d_loan_tot_cnt'] = df['ovd_30d_loan_tot_cnt'].astype(str)
        # printCount(df['ovd_30d_loan_tot_cnt'])
        # df['ovd_30d_loan_tot_cnt'].replace({r'\N': '0'}, inplace=True)
        # df['ovd_30d_loan_tot_cnt'] = df['ovd_30d_loan_tot_cnt'].astype(int)
        # ( pd.cut(df['ovd_30d_loan_tot_cnt'], bins=[-1, 2, 1000], labels=False) == 1).sum()
        # plt.show()
        # df['ovd_30d_loan_tot_cnt'].describe()
        # features.append('ovd_30d_loan_tot_cnt')

        #####################

        ############# 历史贷款最长逾期天数

        # printCount(df['his_lng_ovd_day'])
        # df['his_lng_ovd_day'].replace({r'\N': '0'}, inplace=True)
        # # 187 有 187 个人的flag是0，但是有历史贷款最长逾期天数？
        # df['his_lng_ovd_day'] = df['his_lng_ovd_day'].astype(int)
        # assert df[df['his_lng_ovd_day'] != '0']['flag'].sum() == 0
        # df['his_lng_ovd_day'][df['his_lng_ovd_day'] != 0].describe()
        # #
        # df['his_lng_ovd_day'][df['his_lng_ovd_day'] != 0].hist()
        # plt.show()
        df.drop(columns=['his_lng_ovd_day'], inplace=True)
        # ###### 这个不知道如何用，先drop了



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

    data['train'] = {
        'beh': pd.read_csv(behtrainPath, dtype=str),
        'tag': pd.read_csv(tagtrainPath, dtype=str),
        'trd': pd.read_csv(trdtrainPath, dtype=str)
    }

    data['test'] = {
        'beh': pd.read_csv(behtestPath, dtype=str),
        'tag': pd.read_csv(tagtestPath, dtype=str),
        'trd': pd.read_csv(trdtestPath, dtype=str)
    }

    features = []
    mappings = {
        'encode': [],
    }

    processors = [processingBeh, processingTag, processingTrd]
    for processor in processors:
        processor(data, features, mappings)

    for k in mappings:
        mappings[k] = list(set(mappings[k]))

    for table in ['tag', 'trd']:
        castType(data['train'][table], data['test'][table], mappings)

    train = pd.merge(data['train']['tag'], data['train']['newtrd'], on='id', how='left')
    train = pd.merge(train, data['train']['newbeh'], on='id', how='left')
    train.fillna(0, inplace=True)
    encodedFeatures = train[mappings['encode']]
    y = train['flag'].values.ravel()
    train.drop(columns=mappings['encode'] + ['id', 'flag'], inplace=True)
    onehot = OneHotEncoder()
    trainarray = np.c_[train.values, onehot.fit_transform(encodedFeatures).toarray()]

    test_id = data['test']['tag']['id'].values
    test = pd.merge(data['test']['tag'], data['test']['newtrd'], on='id', how='left')
    test = pd.merge(test, data['test']['newbeh'], on='id', how='left')
    test.fillna(0, inplace=True)
    encodedFeatures = test[mappings['encode']]
    test.drop(columns=mappings['encode'] + ['id'], inplace=True)
    testarray = np.c_[test.values, onehot.transform(encodedFeatures).toarray()]

    data = {
        "X": trainarray,
        "y": y,
        "X_test": testarray,
        'id': test_id
    }
    np.savez_compressed(processedDataPath, **data)
    print('Export new files {}!'.format(processedDataPath))

if __name__ == '__main__':
    process_data()
