# -*- coding: utf-8 -*-
"""
Created on Fri May 19 13:44:53 2023

@author: yjzha
"""
from DataLoader import getTS,LOCALDATA
from dlm import DynamicLinearModel
from fselection import forward_regression,backward_regression
from pandas.tseries.offsets import BDay
from scipy.stats import t
from sklearn.linear_model import LinearRegression,Ridge
from statsmodels.tools.tools import pinv_extended
import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import streamlit as st

td = datetime.datetime.today().date()

def stat(pnl):
    ir = pnl.mean() / (pnl.std()+1e-5) * np.sqrt(255)
    tot = pnl.sum()
    n_win = len(pnl[pnl>0])
    n_lss = len(pnl[pnl<0])
    wrate = n_win / max(n_win+n_lss, 1)
    a_win = sum(pnl[pnl>0]) / max(n_win,1)
    a_lss = sum(pnl[pnl<0]) / max(n_lss,1)
    cumu = np.cumsum(pnl)
    hwm = [max(cumu[:i]) for i in range(1,len(cumu))]
    DD = hwm - cumu[:-1]
    maxDD = np.max(DD)
    Calmar = tot / (maxDD + 1e-5)
    dret = {'Sharpe ': int(ir*100)/100,
            'Total ': int(tot*100)/100,
            'num win ': int(n_win),
            'num lss ': int(n_lss),
            'win % ': int(wrate*100)/100,
            'per win': int(a_win*100)/100,
            'per lss': int(a_lss*100)/100,
            'max DD': int(maxDD*100)/100,
            'Calmar': int(Calmar*100)/100}
    return dret

def loadDataFromDB(ldInd, ldSpx, ldNdx, ldRut):
    tmp = []
    if ldInd and 'indexData' not in st.session_state:
        ts = getTS(LOCALDATA+'EQDB.db', 'eqIndex', None)
        st.session_state.indexData = ts
        tmp.extend(ts.columns)
    if ldSpx and 'spxData' not in st.session_state:
        ts = getTS(LOCALDATA+'EQDB.db', 'spx', None)
        st.session_state.spxData = ts
        tmp.extend(ts.columns)
    if ldNdx and 'ndxData' not in st.session_state:
        ts = getTS(LOCALDATA+'EQDB.db', 'ndx', None)
        st.session_state.ndxData = ts
        tmp.extend(ts.columns)
    if ldRut and 'rutData' not in st.session_state:
        ts = getTS(LOCALDATA+'EQDB.db', 'rut', None)
        st.session_state.rutData = ts
        tmp.extend(ts.columns)
    if 'allTkrs' not in st.session_state:
        st.session_state.allTkrs = tmp
    else:
        origList = st.session_state.allTkrs
        st.session_state.allTkrs = origList + tmp
    st.write('Data loading finished')

def genPairs(sd, ed, pval, stkUni):
    """ main function to identify pairs """
    df = []
    if 'Index' in stkUni:
        if 'indexData' not in st.session_state:
            st.error('genPairs: please load stock index data')
            return
        else:
            df.append(st.session_state.indexData.dropna(how='all',axis=1))
    if 'SP500' in stkUni:
        if 'spxData' not in st.session_state:
            st.error('genPairs: please load spx stock data')
            return
        else:
            df.append(st.session_state.spxData.dropna(how='all',axis=1))
    if 'NASDAQ100' in stkUni:
        if 'ndxData' not in st.session_state:
            st.error('genPairs: please load nasdaq stock data')
            return
        else:
            df.append(st.session_state.ndxData.dropna(how='all',axis=1))
    if 'RUSSELL2000' in stkUni:
        if 'rutData' not in st.session_state:
            st.error('genPairs: please load russell stock data')
            return
        else:
            df.append(st.session_state.rutData.dropna(how='all',axis=1))
    df = pd.concat(df, axis=1)
    df = df.loc[:,~df.columns.duplicated()]
    df = df.loc[sd:ed].fillna(method='ffill')
    df = df.dropna(how='any', axis=1)
    tkrList = df.columns.tolist()
    nTkr = len(tkrList)
    tStat = t.ppf(pval, 2)
    lr = LinearRegression()
    pInfo = []
    for j in range(nTkr-1):
        lr.fit(df.iloc[:,[nTkr-1-j]], df.iloc[:,:nTkr-1-j])
        dfB = pd.DataFrame(lr.coef_,index=tkrList[:nTkr-1-j],columns=[tkrList[nTkr-1-j]])
        yFit = np.dot(df.iloc[:,[nTkr-1-j]], dfB.T.values) + lr.intercept_
        yFit = pd.DataFrame(yFit,index=df.index,columns=tkrList[:nTkr-1-j])
        dfResid = df.iloc[:,:nTkr-1-j] - yFit
        dfResid.columns = [x+'-'+tkrList[nTkr-1-j] for x in dfResid.columns]
        # run adf test, not using python built-in library since it's too slow
        tmpY = dfResid.diff().iloc[1:,:].values
        tmpX = dfResid.shift().iloc[1:,:].values
        c = np.sum((tmpX-np.mean(tmpX,axis=0))*(tmpY-np.mean(tmpY,axis=0)), axis=0)
        v = np.sum((tmpX-np.mean(tmpX,axis=0))*(tmpX-np.mean(tmpX,axis=0)), axis=0)
        b = c/v
        b0 = np.mean(tmpY - b*tmpX, axis=0)
        fitY = b0 + b*tmpX
        mse = (np.sum((tmpY-fitY)**2,axis=0))/(tmpX.shape[0]-2)
        var_b = mse/v
        stat_b = b/np.sqrt(var_b)
        hflife = -np.log(2)/b
        stat_b,hflife = pd.Series(stat_b,index=dfResid.columns),pd.Series(hflife,index=dfResid.columns)
        tmpTkr = stat_b[stat_b<tStat]
        tmpB = dfB.loc[[x.split('-')[0] for x in tmpTkr.index]]
        tmpB.columns = ['b']
        tmpB.index = tmpTkr.index
        tmpInfo = [tmpTkr,hflife.loc[tmpTkr.index],tmpB]
        pInfo.append(tmpInfo)
    pTstat = pd.concat([x[0] for x in pInfo], axis=0)
    pHlf = pd.concat([x[1] for x in pInfo], axis=0)
    hr = pd.concat([x[2] for x in pInfo], axis=0)
    pInfo = pd.concat((pTstat,pHlf,hr), axis=1)
    pInfo.columns = ['Tstat','Half life','Hedge Ratio']
    st.session_state.ptInfo = pInfo
    
def genBacktestData(sd, ed, yTkr, xTkr, hr, useKF):
    """ run backtest """
    if (not yTkr) or (not xTkr):
        return None
    yTs,xTs = None,None
    for dfName in ['indexData','spxData','ndxData','rutData']:
        if dfName not in st.session_state:
            continue
        if xTkr in st.session_state[dfName].columns:
            xTs = st.session_state[dfName][xTkr]
        if yTkr in st.session_state[dfName].columns:
            yTs = st.session_state[dfName][yTkr]
        if (xTs is not None) and (yTs is not None):
            break
    btData = pd.concat((yTs,xTs), axis=1).loc[sd:ed]
    btData = btData.dropna()
    if btData.empty:
        st.error('Either y or x ticker has missing data during the backtest period')
        return None
    if useKF:
        # kalman filter
        dynamic = DynamicLinearModel(include_constant=True)
        dynamic.fit(btData[xTkr], btData[yTkr], method='filter', delta=0.01)
        b0_kf,b_kf = dynamic.beta[:,0],dynamic.beta[:,1]
        residTs = btData[yTkr] - (b0_kf+b_kf*btData[xTkr])
        btData['b'] = b_kf
    else:
        residTs = btData[yTkr] - hr*btData[xTkr]
        btData['b'] = hr
    residTs.name = 'resid'
    btData = pd.concat((btData,residTs), axis=1)
    btData['resid'] -= btData['resid'].mean()
    st.session_state.btData = btData

def runBacktest(btData, yTkr, xTkr, btSigType, btSigVal, btHP):
    """ run pair trading backtest """
    if btData is None or (not yTkr) or (not xTkr):
        return
    def mrTrd(x, sigX):
        if x['Sig']>sigX:
            return -1
        elif x['Sig']<-sigX:
            return 1
        else:
            return 0
    cap = 1000000
    btData['Sig'] = btData['resid'] if btSigType=='Residual Level' else \
                    (btData['resid']-btData['resid'].rolling(21).mean())/btData['resid'].rolling(21).std()
    btData['Trd'] = btData.apply(lambda x:mrTrd(x,btSigVal), axis=1)
    btData['aTrd'] = 0
    btData['N_'+yTkr],btData['N_'+xTkr] = 0,0
    preTD = None
    for d in btData.index:
        if btData.loc[d,'Trd'] == 0:
            if preTD is not None and preTD+BDay(btHP)>d:
                btData.loc[d,'aTrd'] = btData.loc[preTD,'Trd']
                btData.loc[d,['N_'+yTkr,'N_'+xTkr]] = btData.loc[preTD,['N_'+yTkr,'N_'+xTkr]]
        else:
            btData.loc[d,'aTrd'] = btData.loc[d,'Trd']
            btData.loc[d,'N_'+yTkr] = int(cap/btData.loc[d,yTkr])
            btData.loc[d,'N_'+xTkr] = int(btData.loc[d,'N_'+yTkr]*btData.loc[d,'b'])
            preTD = d
    btData['N_'+yTkr] *= btData['aTrd']
    btData['N_'+xTkr] *= (-btData['aTrd'])
    btData['Pnl'] = btData['N_'+yTkr].shift()*btData[yTkr].diff() + btData['N_'+xTkr].shift()*btData[xTkr].diff()
    btData['cPnl'] = btData['Pnl'].cumsum() + cap
    fig = px.line(btData, x=btData.index, y='cPnl', title='Equity with 1mm starting capital', markers=True)
    st.plotly_chart(fig, use_container_width=True)
    # show performance info
    perf = stat(btData['Pnl'].fillna(0).values)
    th_props = [('font-size','12px'),('font-weight','bold')]
    td_props = [('font-size','14px')]
    tbl = pd.Series(perf,name='Strategy Stats').to_frame().T
    tbl = tbl.applymap(lambda x:'{:.2f}'.format(x))
    styles = [dict(selector='th',props=th_props),dict(selector='td',props=td_props)]
    tbl = tbl.style.set_table_styles(styles)
    st.table(tbl)
    st.write('Backtest Info')
    st.dataframe(btData.style.format({yTkr:'{:.1f}',xTkr:'{:.1f}','resid':'{:.2f}','b':'{:.2f}','Sig':'{:.2f}',\
                                      'Pnl':'{:.0f}','cPnl':'{:.0f}'}))
        
def runIndexReg(tgtInd, ssList, ssData, regType, sd, ed):
    if not tgtInd or not ssList:
        return None
    indexTs = st.session_state.indexData[tgtInd].copy()
    stockTs = ssData[ssList].copy()
    regData = pd.concat((indexTs,stockTs), axis=1).loc[sd:ed]
    regData = regData.dropna()
    if regData.empty:
        st.error('Missing data during the selected regression period')
        return None
    dfRet = regData.pct_change().iloc[1:]
    if regType == 'OLS':
        X,y = sm.add_constant(dfRet.iloc[:,1:]),dfRet[tgtInd]
        model = sm.OLS(y,X).fit()
        y_pred = model.predict(X)
        res = model
    else:
        X,y = dfRet.iloc[:,1:],dfRet[tgtInd]
        x = sm.add_constant(X)
        model = Ridge(alpha=1e-2)
        model.fit(X,y)
        y_pred = model.predict(X)
        olsModel = sm.OLS(y, x)
        pinv_wexog,_ = pinv_extended(x)
        normalized_cov_params = np.dot(pinv_wexog, np.transpose(pinv_wexog))
        model_params = np.hstack([np.array([model.intercept_]), model.coef_])
        res = sm.regression.linear_model.OLSResults(olsModel, model_params, normalized_cov_params)
    fig = px.scatter(x=y, y=y_pred, labels={'x':'Actual','y':'prediction'})
    fig.add_shape(type='line',line=dict(dash='dash'),x0=y.min(),y0=y.min(),x1=y.max(),y1=y.max())
    st.plotly_chart(fig, use_container_width=True)
    with st.expander('Regression summary'):
        st.write(res.summary())
    return model

def runFselection(inNum, tgtInd, ssData, useBwdStep, sd, ed):
    """ select subset of stocks to best explain index """
    if not tgtInd or ssData is None:
        return None
    indexTs = st.session_state.indexData[tgtInd].copy()
    df = pd.concat((indexTs,ssData), axis=1).loc[sd:ed]
    df = df.dropna(how='all',axis=1).fillna(method='ffill').dropna(how='any',axis=1)
    dfRet = df.pct_change().iloc[1:]
    X = dfRet.loc[:,dfRet.columns!=tgtInd]
    X = sm.add_constant(X)
    y = dfRet[tgtInd]
    if useBwdStep:
        tkrInc = backward_regression(X, y, inNum+1, initial_list=[], verbose=False)
    else:
        tkrInc = forward_regression(X, y, inNum+1, initial_list=['const'], verbose=False)
    model = sm.OLS(y, X[tkrInc]).fit()  
    st.session_state.fselect = (tkrInc, model)

st.set_page_config(page_title='Test', initial_sidebar_state='expanded', layout='wide')

with st.sidebar:
    st.write('Data Loading')
    loadIndex = st.checkbox('Load stock index data')
    loadSpx = st.checkbox('Load sp500 stock data')
    loadNdx = st.checkbox('Load nasdaq100 stock data')
    loadRut = st.checkbox('Load russel2000 stock data')
    st.button('Load Data', on_click=loadDataFromDB, args=(loadIndex, loadSpx, loadNdx, loadRut))
       
    
tab1,tab2,tab3 = st.tabs(['Pair Trading','Pair Backtest','Index Regression'])
with tab1:
    col1_tab1,col2_tab1 = st.columns([1,3])
    with col1_tab1:
        ptStart = st.date_input('Cointegration Test start', datetime.date(2020,1,1), max_value=td)
        ptEnd = st.date_input('Cointegration Test end', td)
        pvalThresh = st.number_input('Unit root p-value threshold:', value=0.05)
        ptUni = st.multiselect('Trading Universe', ['Index','SP500','NASDAQ100','RUSSELL2000'], ['SP500'])
        st.button('Find Pairs', on_click=genPairs, args=(ptStart, ptEnd, pvalThresh, ptUni))
    with col2_tab1:
        if 'ptInfo' in st.session_state:
            df = st.session_state.ptInfo.sort_values('Half life')
            df['Y'],df['X'] = [x.split('-')[0] for x in df.index],[x.split('-')[1] for x in df.index]
            st.write('There are {} pairs identified, only show 20 for performance purpose'.format(df.shape[0]))
            st.dataframe(df.head(20).style.format({'Tstat':'{:.2f}','Half life':'{:.1f}','Hedge Ratio':'{:.2f}'}))
with tab2:
    col1_tab2,col2_tab2 = st.columns([1,3])
    with col1_tab2:
        if 'allTkrs' in st.session_state:
            tkrList = st.session_state.allTkrs.copy()
            btY = st.selectbox('Backtest Y ticker', tkrList)
            btX = st.selectbox('Backtest X ticker', tkrList)
        else:
            btY,btX = None,None
        btStart = st.date_input('Backtest start', datetime.date(2020,1,1), max_value=td)
        btEnd = st.date_input('Backtest end', td)
        useKF = st.checkbox('Use Kalman Filter Beta?')
        btHr = st.number_input('Input hedge ratio:', disabled=useKF)
    with col2_tab2:
        st.button('Generate Signal(residual) Data', on_click=genBacktestData, args=(btStart, btEnd, btY, btX, btHr, useKF))
        if 'btData' in st.session_state:
            btData = st.session_state.btData.copy()
            fig = px.line(btData, x=btData.index, y='resid', title='Residual between pair', markers=True)
            st.plotly_chart(fig, use_container_width=True)
    st.divider()
    col3_tab2,col4_tab2 = st.columns([1,3])
    with col3_tab2:
        btSigType = st.selectbox('Signal Type', ['Residual Level','Residual Zscore(21d)'])
    with col4_tab2:
        if not 'btData' in st.session_state:
            st.error('Please generate signal data first.')
            btData,btSigVal,btHP = None,None,None
        else:
            btData = st.session_state.btData.copy()
            if btSigType == 'Residual Level':
                btSigVal = st.slider('Signal Threshold', btData['resid'].abs().min(), btData['resid'].abs().max(), btData['resid'].abs().median())
            else:
                btSigVal = st.slider('Signal Threshold', 0.0, 3.0, .5)
            btHP = st.slider('Holding Period', 1, 10, 5)
    runBacktest(btData, btY, btX, btSigType, btSigVal, btHP)
with tab3:
    col1_tab3,col2_tab3 = st.columns([1,2])
    with col1_tab3:
        tgtIndex = st.selectbox('Target Index', ['','^GSPC','^NDX','^RUT'])
        ssList,ssData = None,None
        if tgtIndex:
            if ('indexData' not in st.session_state) or (tgtIndex not in st.session_state.indexData):
                st.error('missing index data '+tgtIndex)
            else:
                if tgtIndex == '^GSPC':
                    if 'spxData' not in st.session_state:
                        st.error('please load spx stock data')
                    else:
                        ssList = st.multiselect('Select single stocks to explain spx', st.session_state.spxData.columns.tolist(), None)
                        ssData = st.session_state.spxData.copy()
                if tgtIndex == '^NDX':
                    if 'ndxData' not in st.session_state:
                        st.error('please load ndx stock data')
                    else:
                        ssList = st.multiselect('Select single stocks to explain ndx', st.session_state.ndxData.columns.tolist(), None)
                        ssData = st.session_state.ndxData.copy()
                if tgtIndex == '^RUT':
                    if 'rutData' not in st.session_state:
                        st.error('please load rut stock data')
                    else:
                        ssList = st.multiselect('Select single stocks to explain rut', st.session_state.rutData.columns.tolist(), None)
                        ssData = st.session_state.rutData.copy()
        regType = st.selectbox('Regression Method', ['OLS','Ridge(Handle correlated single stocks)'])
        regStart = st.date_input('Regression start', datetime.date(2020,1,1), max_value=td)
        regEnd = st.date_input('Regression end', td)
    with col2_tab3:
        mod = runIndexReg(tgtIndex, ssList, ssData, regType, regStart, regEnd)
    st.divider()
    numSs = st.number_input('Number of stocks to explain the index best', 1, 10, 3)
    useBwdStep = st.checkbox('Use backward stepwise regression? (default is forward stepwise)')
    st.button('Run Stepwise regression', on_click=runFselection, args=(numSs, tgtIndex, ssData, useBwdStep, regStart, regEnd))
    if 'fselect' in st.session_state:
        st.write('Best subset of stocks: '+', '.join(st.session_state.fselect[0]))
        with st.expander('Regression summary'):
            st.write(st.session_state.fselect[1].summary())
    
    
    