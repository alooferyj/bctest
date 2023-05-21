# -*- coding: utf-8 -*-
"""
Created on Sat May  6 11:10:19 2023

@author: yjzha
"""
from collections import namedtuple
import datetime
import os
import numpy as np
import pandas as pd
import sqlite3
import yfinance as yf

LOCALDATA = 'C:/Apps/pythonlib/bctest/data/'

DBCONFIG = namedtuple('dbconfig',['dbName','dbDir','dataDir','dataSource','dbTables'])
DBTABLECONFIG = namedtuple('tableconfig',['tblName','tblFile','tblDesc'])
SP500Tbl = DBTABLECONFIG('spx','SP500 HIST','sp500 component stocks')
NDX100Tbl = DBTABLECONFIG('ndx','NDX100 HIST','ndx100 component stocks')
RUT2000Tbl = DBTABLECONFIG('rut','RUT2000 HIST','rut2000 component stocks')
INDEXTbl = DBTABLECONFIG('eqIndex','INDEX HIST','indices')
EQDBCfg = DBCONFIG('EQDB',LOCALDATA,LOCALDATA,'YFINANCE',(SP500Tbl,NDX100Tbl,RUT2000Tbl,INDEXTbl))


def addTable(dbDir, inFile, tblName=None, ifExist='replace'):
    """ add new table to database """
    if isinstance(inFile, pd.DataFrame):
        df = inFile
        if not tblName: raise Exception('Have to provide a table name')
    else:
        df = pd.read_csv(inFile+'.csv', index_col=0, parse_dates=True)
    df.index = [x.date() for x in df.index.tolist()]
    if not tblName:
        tblName = inFile
    con = sqlite3.connect(dbDir)
    with con:
        df.to_sql(tblName, con, index=True, index_label='Date', if_exists=ifExist)
    con.close()
    
def removeTable(dbDir, tbName):
    con = sqlite3.connect(dbDir)
    with con:
        cur = con.cursor()
        cur.execute('DROP TABLE ' + tbName)
        cur.execute('vacuum')
    con.close()
    
def getTables(dbDir):
    con = sqlite3.connect(dbDir)
    with con:
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        res = cur.fetchall()
    con.close()
    res = [x[0] for x in res]
    return res

def getTS(dbDir, tbName, tsNames=None, fromDate=None, toDate=None):
    if not os.path.exists(dbDir):
        raise Exception('Database does not exist')
    if tsNames is None:
        tsQuery = '*'
    else:
        tsNames = ['"'+x+'"' for x in tsNames]
        tsQuery = 'Date,' + ','.join(tsNames)
    tsQuery = 'SELECT ' +tsQuery + ' from ' + tbName
    if fromDate:
        tsQuery += " where Date>'{}'".format(fromDate.strftime('%Y-%m-%d'))
    if toDate:
        tsQuery = tsQuery + " and Date<='{}'".format(toDate.strftime('%Y-%m-%d')) if fromDate else \
            tsQuery + " where Date<='{}'".format(toDate.strftime('%Y-%m-%d'))
    con = sqlite3.connect(dbDir)
    with con:
        qdf = pd.read_sql(tsQuery, con)
    con.close()
    qdf = qdf.set_index('Date')
    qdf.index = pd.to_datetime(qdf.index)
    qdf = qdf.fillna(np.nan)
    return qdf

def buildDB(dbConfigs):
    for cfg in dbConfigs:
        dbName = cfg.dbName
        dbDir = LOCALDATA
        dataDir = cfg.dataDir
        dataSource = cfg.dataSource
        dbTables = cfg.dbTables
        replaceOld = 'replace' if os.path.exists(dbDir+dbName+'.db') else 'fail'
        for tb in dbTables:
            tbName = tb.tblName
            tbFile = tb.tblFile
            print(tbName)
            tblDf = pd.read_csv(dataDir+tbFile+'.csv', header=[0,1], index_col=0, parse_dates=True)['Adj Close']
            addTable(dbDir+dbName+'.db', tblDf, tblName=tbName, ifExist=replaceOld)
        print('Following tables have been added: ')
        tbs = getTables(dbDir+dbName+'.db')
        print(tbs)
        
def downloadStockData(histStart, histEnd):
    # sp500
    tkr = pd.read_excel(LOCALDATA+'company list.xlsx',sheet_name='SP500')
    tkrList = ' '.join(tkr['Symbol'].unique().tolist())
    df = yf.download(tkrList, start=histStart, end=histEnd)
    df.to_csv(LOCALDATA+SP500Tbl.tblFile+'.csv')
    # NDX100
    tkr = pd.read_excel(LOCALDATA+'company list.xlsx',sheet_name='NASDAQ100')
    tkrList = ' '.join(tkr['Symbol'].unique().tolist())
    df = yf.download(tkrList, start=histStart, end=histEnd)
    df.to_csv(LOCALDATA+NDX100Tbl.tblFile+'.csv')
    # RUSSELL2000
    tkr = pd.read_excel(LOCALDATA+'company list.xlsx',sheet_name='RUSSEL2000')
    tkrList = ' '.join(tkr['Symbol'].unique().tolist())
    df = yf.download(tkrList, start=histStart, end=histEnd)
    df.to_csv(LOCALDATA+RUT2000Tbl.tblFile+'.csv')
    # index
    indList = ['^GSPC','^NDX','^RUT']
    df = yf.download(indList, start=histStart, end=histEnd)
    df.to_csv(LOCALDATA+INDEXTbl.tblFile+'.csv')
        
        
if __name__ == '__main__':
    
    histEnd = datetime.datetime.today()
    histStart = datetime.date(2017,1,1)
    
    #downloadStockData(histStart, histEnd)
    buildDB([EQDBCfg])
    ts = getTS(LOCALDATA+'EQDB.db', 'eqIndex', None, fromDate=datetime.date(2020,1,1))
    print(ts.head())
    
    
    
    
    
    