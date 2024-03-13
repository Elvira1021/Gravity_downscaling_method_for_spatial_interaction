import numpy as np
from scipy import stats
import math
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd


def evaluate(p, r, isprint=True):
    if (isprint):
        print('\nnum of test flows:', len(r))
        print('real_min:', min(r), ', real_max:', max(r), 'sum:', int(sum(r)))
        print('pred_min:', min(p), ', pred_max:', max(p), 'sum:', int(sum(p)))
        print('real:', list(map(int, r[0:20])))
        print('pred:', list(map(int, p[0:20])))
    p = np.array(list(map(int, p))).reshape(-1, 1)
    r = np.array(list(map(int, r))).reshape(-1, 1)
    # print('MAE:', round(np.mean(np.abs(r - p)),3))
    c1 = 0
    mape = 0
    c2 = 0
    ssi = 0
    for i in range(p.shape[0]):
        if r[i]:
            mape += np.abs((r[i] - p[i]) / r[i])
            c1 += 1
        if r[i] + p[i]:
            ssi += min(r[i], p[i]) / (r[i] + p[i])
            c2 += 1
    stack = np.column_stack((p, r))
    smc = stats.spearmanr(r, p)
    if (isprint):
        print('MAPE:', np.round(mape / c1, 3))
        # print('MSE:', round(np.mean(np.square(r - p)), 3))
        print('RMSE:', np.round(np.sqrt(np.mean(np.square(r - p))), 3))
        print('CPC:', np.round(2 * np.sum(np.min(stack, axis=1)) / np.sum(stack), 3))
        print('RTAE:', np.round(np.sum(np.abs(p-r)) / np.sum(r), 3))
        print('SMC: correlation =', round(
            smc[0], 3), ', p-value =', round(smc[1], 3))
        print('r-squared:', correlation(r, p)[0])
    result = {'num': len(r), 'real_min': int(min(r)[0]), 'real_max': int(max(r)[0]), 'real_sum': int(sum(r)), 'pred_min': int(min(p)[0]), 'pred_max': int(max(p)[0]), 'pred_sum': int(sum(p)),
              'MAPE': float(np.round(mape / c1, 3)[0]), 'RMSE': np.round(np.sqrt(np.mean(np.square(r - p))), 3), 'CPC': np.round(2 * np.sum(np.min(stack, axis=1)) / np.sum(stack), 3), 
              'RTAE': np.round(np.sum(np.abs(p-r)) / np.sum(r), 3), 'correlation': round(smc[0], 3), 'r-squared': correlation(r, p)[0]}
    return result


def correlation(X, Y):
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(0, len(X)):
        diffXXBar = X[i] - xBar
        diffYYBar = Y[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar**2
        varY += diffYYBar**2
        SST = math.sqrt(varX * varY)
    # print ("使用math库：r：", SSR / SST,"r-squared：", (SSR / SST)**2)
    return (SSR / SST)**2


def Rsquare(p, r):
    predlist = p
    actlist = r
    regline = np.polyfit(actlist, predlist, 1)
    estlist = [regline[0]*val+regline[1] for val in actlist]
    SSE = 0
    SST = 0
    meanv = np.mean(predlist)
    for i in range(len(predlist)):
        SSE += (estlist[i]-predlist[i])**2
        SST += (meanv-predlist[i])**2
    return 1 - SSE/SST


def load_relation(relationfile):
    reg_rela = {}
    reg_rela_file = np.genfromtxt(relationfile, dtype=np.int64, delimiter=',')
    for i in reg_rela_file:
        if (i[1] not in reg_rela):
            reg_rela[i[1]] = [i[0]]
        else:
            reg_rela[i[1]].append(i[0])
    # print(reg_rela)
    return reg_rela


def finetuning(countydf, county_city_rela, citydf, predname):
    countyflow = {}
    for i in countydf.iterrows():
        countyflow[(i[1]['startcode'], i[1]['endcode'])] = i[1][predname]
    cityflow = {}
    countyflowfine = {}
    for i in citydf.iterrows():
        cityflow[(i[1]['startcode'], i[1]['endcode'])] = i[1]['flow_intensity']
    for _ in range(3):
        for i in citydf.iterrows():
            origin_start_id = i[1]['startcode']
            origin_end_id = i[1]['endcode']
            start_ids = county_city_rela[origin_start_id]
            end_ids = county_city_rela[origin_end_id]
            pre_all = []
            for k in start_ids:
                for b in end_ids:
                    if ((k, b) in countyflow and countyflow[(k, b)] > 0):
                        pre_all.append(countyflow[(k, b)])
                    elif ((k, b) not in countyflow):
                        countyflow[(k, b)] = 0
            pre_sum = sum(pre_all)
            # print(pre_all)
            # 如果该区域流为0，那么子区域全部置为0
            if ((origin_start_id, origin_end_id) not in cityflow):
                for sid in start_ids:
                    for eid in end_ids:
                        countyflowfine[(sid, eid)] = 0
            elif (cityflow[(origin_start_id, origin_end_id)] > 0 and pre_sum != 0):
                # print(cityflow[(origin_start_id,origin_end_id)],pre_sum)
                cof = cityflow[(origin_start_id, origin_end_id)]/pre_sum
                for sid in start_ids:
                    for eid in end_ids:
                        if (countyflow[(sid, eid)] < 0):
                            countyflowfine[(sid, eid)] = 0
                        else:
                            countyflowfine[(sid, eid)
                                           ] = countyflow[(sid, eid)]*cof
                finesum = [countyflowfine[(sid, eid)]
                           for sid in start_ids for eid in end_ids]
                # print("finetuning",sum(finesum),cityflow[(origin_start_id,origin_end_id)])
    # print(countyflow)
    county_flow_finetune = []
    for i in countydf.iterrows():
        county_flow_finetune.append(
            countyflowfine[(i[1]['startcode'], i[1]['endcode'])])
    countydf[predname+'_finetune'] = county_flow_finetune
    # print(sum(county_flow_finetune),sum(citydf['flow_intensity']),sum(countydf['flow_intensity']))
    return countydf


def Sij(i, j, Dist, attr_list):
    dij = Dist[i][j]
    Disti_list = Dist[i]
    sij = 0
    for k in range(len(Disti_list)):
        if (k != j and k != i and Dist[i][k] < dij):
            sij += attr_list[k]
    return sij


def RM_Prob(i, j, Dist, attr_list):
    sij = Sij(i, j, Dist, attr_list)
    mi = attr_list[i]
    mj = attr_list[j]
    Prob = mi*mj/((sij+mi+mj)*(mi+sij))
    if (Prob > 1):
        print(i, j, sij, mi, mj)
    return Prob


def OPS_Prob(i, j, Dist, attr_list):
    sij = Sij(i, j, Dist, attr_list)
    mi = attr_list[i]
    mj = attr_list[j]
    Prob = mj/(sij+mi+mj)
    if (Prob > 1):
        print(i, j, sij, mi, mj)
    return Prob


def Get_idxmap(df):
    idx_map = {}
    cnt = 0
    for i in df.iterrows():
        idx_map[i[1]['code']] = cnt
        cnt += 1
    return idx_map


def UO_Prob(i, j, Dist, attr_list, a, b):
    # a=0 b=1时辐射模型
    # a=1 b=0时机会优先选择模型
    sij = Sij(i, j, Dist, attr_list)
    mi = attr_list[i]
    mj = attr_list[j]
    Prob = (mi+a*sij)*mj/((mi+(a+b)*sij+mj)*((a+b)*sij+mi))
    if (Prob > 1):
        print(i, j, sij, mi, mj)
    return Prob


def Fit_UOmodel(a, b, citydf, city_idx_map, city_Dist, city_attr_df, attr):
    city_attr_list = list(city_attr_df[attr])
    RM_Prob_list = []
    pop_prob_list = []
    # Oi_list=[]
    for i in citydf.iterrows():
        start_id = city_idx_map[i[1]['startcode']]
        end_id = city_idx_map[i[1]['endcode']]
        Prob = UO_Prob(start_id, end_id, city_Dist, city_attr_list, a, b)
        pop_prob = i[1]['o_pop']*Prob
        RM_Prob_list.append(Prob)
        pop_prob_list.append(pop_prob)
        # Oi_list.append(city_Oi[start_id])

    # citydf['RM_Prob']=RM_Prob_list
    citydf[attr+'_prob'] = pop_prob_list
    citydf[attr+'_prob_l'] = np.log(pop_prob_list)
    # citydf['Oi']=Oi_list
    # citydf['RM_flow']=citydf['Oi']*citydf['RM_Prob']
    return citydf


def Fit_RMmodel(citydf, city_idx_map, city_Dist, city_attr_df, attr):
    # 拟合city
    city_attr_list = list(city_attr_df[attr])
    RM_Prob_list = []
    pop_prob_list = []
    Oi_list = []
    for i in citydf.iterrows():
        start_id = city_idx_map[i[1]['startcode']]
        end_id = city_idx_map[i[1]['endcode']]
        Prob = RM_Prob(start_id, end_id, city_Dist, city_attr_list)
        pop_prob = i[1]['o_pop']*Prob
        RM_Prob_list.append(Prob)
        pop_prob_list.append(pop_prob)

    # citydf['RM_Prob']=RM_Prob_list
    citydf[attr+'_prob'] = pop_prob_list
    citydf[attr+'_prob_l'] = np.log(pop_prob_list)
    # citydf['RM_flow']=citydf['Oi']*citydf['RM_Prob']
    return citydf


def Fit_OPSmodel(citydf, city_idx_map, city_Dist, city_attr_df, attr):
    # 拟合city
    city_attr_list = list(city_attr_df[attr])
    RM_Prob_list = []
    pop_prob_list = []
    Oi_list = []
    for i in citydf.iterrows():
        start_id = city_idx_map[i[1]['startcode']]
        end_id = city_idx_map[i[1]['endcode']]
        Prob = OPS_Prob(start_id, end_id, city_Dist, city_attr_list)
        pop_prob = i[1]['o_pop']*Prob
        RM_Prob_list.append(Prob)
        pop_prob_list.append(pop_prob)
        # Oi_list.append(city_Oi[start_id])

    # citydf['RM_Prob']=RM_Prob_list
    citydf[attr+'_prob'] = pop_prob_list
    citydf[attr+'_prob_l'] = np.log(pop_prob_list)
    # citydf['Oi']=Oi_list
    # citydf['RM_flow']=citydf['Oi']*citydf['RM_Prob']
    return citydf


def Sij(i, j, Dist, attr_list):
    dij = Dist[i][j]
    Disti_list = Dist[i]
    sij = 0
    for k in range(len(Disti_list)):
        if (k != j and k != i and Dist[i][k] < dij):
            sij += attr_list[k]
    return sij


def RM_Prob(i, j, Dist, attr_list):
    sij = Sij(i, j, Dist, attr_list)
    mi = attr_list[i]
    mj = attr_list[j]
    Prob = mi*mj/((sij+mi+mj)*(mi+sij))
    if (Prob > 1):
        print(i, j, sij, mi, mj)
    return Prob


def OPS_Prob(i, j, Dist, attr_list):
    sij = Sij(i, j, Dist, attr_list)
    mi = attr_list[i]
    mj = attr_list[j]
    Prob = mj/(sij+mi+mj)
    if (Prob > 1):
        print(i, j, sij, mi, mj)
    return Prob


def Get_idxmap(df):
    idx_map = {}
    cnt = 0
    for i in df.iterrows():
        idx_map[i[1]['code']] = cnt
        cnt += 1
    return idx_map


def UO_Prob(i, j, Dist, attr_list, a, b):
    # a=0 b=1时辐射模型
    # a=1 b=0时机会优先选择模型
    sij = Sij(i, j, Dist, attr_list)
    mi = attr_list[i]
    mj = attr_list[j]
    Prob = (mi+a*sij)*mj/((mi+(a+b)*sij+mj)*((a+b)*sij+mi))
    if (Prob > 1):
        print(i, j, sij, mi, mj)
    return Prob


def Fit_UOmodel(a, b, citydf, city_idx_map, city_Dist, city_attr_df, attr):
    city_attr_list = list(city_attr_df[attr])
    RM_Prob_list = []
    pop_prob_list = []
    # Oi_list=[]
    for i in citydf.iterrows():
        start_id = city_idx_map[i[1]['startcode']]
        end_id = city_idx_map[i[1]['endcode']]
        Prob = UO_Prob(start_id, end_id, city_Dist, city_attr_list, a, b)
        pop_prob = i[1]['o_pop']*Prob
        RM_Prob_list.append(Prob)
        pop_prob_list.append(pop_prob)
        # Oi_list.append(city_Oi[start_id])

    # citydf['RM_Prob']=RM_Prob_list
    citydf[attr+'_prob'] = pop_prob_list
    citydf[attr+'_prob_l'] = np.log(pop_prob_list)
    # citydf['Oi']=Oi_list
    # citydf['RM_flow']=citydf['Oi']*citydf['RM_Prob']
    return citydf


def Fit_RMmodel(citydf, city_idx_map, city_Dist, city_attr_df, attr):
    # 拟合city
    city_attr_list = list(city_attr_df[attr])
    RM_Prob_list = []
    pop_prob_list = []
    Oi_list = []
    for i in citydf.iterrows():
        start_id = city_idx_map[i[1]['startcode']]
        end_id = city_idx_map[i[1]['endcode']]
        Prob = RM_Prob(start_id, end_id, city_Dist, city_attr_list)
        pop_prob = i[1]['o_pop']*Prob
        RM_Prob_list.append(Prob)
        pop_prob_list.append(pop_prob)

    # citydf['RM_Prob']=RM_Prob_list
    citydf[attr+'_prob'] = pop_prob_list
    citydf[attr+'_prob_l'] = np.log(pop_prob_list)
    # citydf['RM_flow']=citydf['Oi']*citydf['RM_Prob']
    return citydf


def calRMSE(p, r):
    r = list(r)
    p = list(p)
    p = np.array(p).reshape(-1, 1)
    r = np.array(r).reshape(-1, 1)
    rmse = np.round(np.sqrt(np.mean(np.square(r - p))), 3)
    return rmse


def Fit_OPSmodel(citydf, city_idx_map, city_Dist, city_attr_df, attr):
    # 拟合city
    city_attr_list = list(city_attr_df[attr])
    RM_Prob_list = []
    pop_prob_list = []
    Oi_list = []
    for i in citydf.iterrows():
        start_id = city_idx_map[i[1]['startcode']]
        end_id = city_idx_map[i[1]['endcode']]
        Prob = OPS_Prob(start_id, end_id, city_Dist, city_attr_list)
        pop_prob = i[1]['o_pop']*Prob
        RM_Prob_list.append(Prob)
        pop_prob_list.append(pop_prob)
        # Oi_list.append(city_Oi[start_id])

    # citydf['RM_Prob']=RM_Prob_list
    citydf[attr+'_prob'] = pop_prob_list
    citydf[attr+'_prob_l'] = np.log(pop_prob_list)
    # citydf['Oi']=Oi_list
    # citydf['RM_flow']=citydf['Oi']*citydf['RM_Prob']
    return citydf


# key为区划编码
def get_idxmap(idxarray):
    origin = idxarray[:, 0]
    rearrange = idxarray[:, 1]
    idx_map = dict(zip(origin, rearrange))
    return idx_map

# key为顺序


def get_reidxmap(idxarray):
    origin = idxarray[:, 1]
    rearrange = idxarray[:, 0]
    reidx_map = dict(zip(origin, rearrange))
    return reidx_map


def cal_VIF(X_train):
    vif = [variance_inflation_factor(X_train, ix)
           for ix in range(X_train.shape[1])]
    return vif


def output_result(region, name, column):
    countydf = pd.read_csv('../result/'+region+'_county_flow_detail.csv')
    countydf[name] = column
    countydf.to_csv('../result/'+region+'_county_flow_detail.csv', index=False)

def output2csv(filename, colname, columndf):
    countydf = pd.read_csv(filename)
    countydf[colname] = columndf
    countydf.to_csv(filename, index=False)