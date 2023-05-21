# -*- coding: utf-8 -*-
"""
Created on Sun May 21 07:19:36 2023

@author: yjzha
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import statsmodels.api as sm

def forward_regression(X, y, targetNum, initial_list=[], verbose=True):
    included = list(initial_list)
    while True:
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_col in excluded:
            model = sm.OLS(y, X[included+[new_col]]).fit()
            new_pval[new_col] = model.pvalues[new_col]
        best_pval = new_pval.min()
        best_feature = new_pval.idxmin()
        included.append(best_feature)
        if verbose:
            print('Add {} with p-value {}'.format(best_feature, best_pval))
        if len(included) == targetNum:
            break
    return included

def backward_regression(X, y, targetNum, initial_list=[], verbose=True):
    included = list(X.columns) if len(initial_list)==0 else initial_list
    while True:
        model = sm.OLS(y, X[included]).fit()
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()
        worst_feature = pvalues.idxmax()
        included.remove(worst_feature)
        if verbose:
            print('Drop {} with p-value {}'.format(worst_feature, worst_pval))
        if len(included) == targetNum:
            break
    return included