from fairlearn.metrics import MetricFrame, selection_rate
from aif360.sklearn.metrics import consistency_score, conditional_demographic_disparity, generalized_entropy_error
from utils.utils import attributes_names_mapping
import numpy as np
import pandas as pd

def demographic_parity_ratio_(df, sensitive_feature):
    """
        df: pandas.DataFrame with TARGET col             : y_true
                                  Predicted_Result col   : y_pred
                                  <sensitive_feature> col: sensitive_features
    """
    mf = MetricFrame(metrics = selection_rate, 
                    y_true = df["TARGET"], 
                    y_pred = df["Predicted_Result"], 
                    sensitive_features = df[sensitive_feature])
    groups_dict = {}
    for group in list(mf.by_group.keys()):
        groups_dict[group] = mf.by_group.loc[group]
        #print(sensitive_feature, "fairness ",min(groups_dict.values()),max(groups_dict.values()))    
    if 'Unknown' in groups_dict:
        groups_dict.pop('Unknown')
    dp = None
    if len(groups_dict) > 1:  
        dp = min(groups_dict.values()) / max(groups_dict.values())
    return dp

def average_odds_difference_(df, sensitive_feature):
    groups = list(df[sensitive_feature].unique())
    if 'Unknown' in groups:
        groups.remove('Unknown')
    groups_TPR_dict = {}
    groups_FPR_dict = {}
    for group in groups:
        target_value_counts = df[df[sensitive_feature] == group]["TARGET"].value_counts()
        ## TPR
        TP = len(df.loc[(df["Predicted_Result"] == 1) & 
                                   (df["TARGET"] == 1) & 
                                   (df[sensitive_feature] == group)])        
        if 1 in target_value_counts:
            P = int(target_value_counts[1])
        else:
            P = 0  
        if P:
            groups_TPR_dict[group] = TP/P
        else:
            groups_TPR_dict[group] = None
        ## FPR
        FP = len(df.loc[(df["Predicted_Result"] == 1) & 
                                    (df["TARGET"] == 0) & 
                                   (df[sensitive_feature] == group)])
        if 0 in target_value_counts:
            N = int(target_value_counts[0])
        else:
            N = 0
        if N:
            groups_FPR_dict[group] = FP/N
        else:
            groups_FPR_dict[group] = None
    if len(groups) == 1:
        ## Not defined
        aod = None
    else:    
        ## TPR
        TPR_diff = None
        tmp = list(filter(lambda x: x[1] is not None, groups_TPR_dict.items()))
        if len(tmp):
            max_TPR = list(max(tmp, key=lambda x: x[1])) ## tuple (group,sel_rate)
            max_TPR[0] = str(max_TPR[0])+', \n TP'
            max_TPR[1] = max_TPR[1]
            if len(tmp)!=1:            
                min_TPR = list(min(tmp, key=lambda x: x[1]))
                min_TPR[0] = str(min_TPR[0])+', \n TP'
                min_TPR[1] = min_TPR[1]
                TPR_diff = max_TPR[1] - min_TPR[1]          
                 
        ## FPR
        FPR_diff = None
        tmp = list(filter(lambda x: x[1] is not None, groups_FPR_dict.items()))
        if len(tmp):
            max_FPR = list(max(tmp, key=lambda x: x[1])) ## tuple (group,sel_rate)
            max_FPR[0] = str(max_FPR[0])+', \n FP'
            max_FPR[1] = max_FPR[1]
            if len(tmp)!=1:                          
                min_FPR = list(min(tmp, key=lambda x: x[1]))
                min_FPR[0] = str(min_FPR[0])+', \n FP'
                min_FPR[1] = min_FPR[1] 
                FPR_diff = max_FPR[1] - min_FPR[1]              
        ##
        if TPR_diff is not None and FPR_diff is not None:
            aod = (TPR_diff + FPR_diff)/2.
        else:
            ## Not defined
            aod = None ## not defined
    return aod

def equal_opportunity_difference_(df, sensitive_feature):
    groups = list(df[sensitive_feature].unique())
    if 'Unknown' in groups:
        groups.remove('Unknown')
    groups_TPR_dict = {}
    for group in groups:
        target_value_counts = df[df[sensitive_feature] == group]["TARGET"].value_counts()
        ## TPR
        TP = len(df.loc[(df["Predicted_Result"] == 1) & 
                                   (df["TARGET"] == 1) & 
                                   (df[sensitive_feature] == group)])        
        if 1 in target_value_counts:
            P = int(target_value_counts[1])
        else:
            P = 0  
        if P:
            groups_TPR_dict[group] = TP/P
        else:
            groups_TPR_dict[group] = None        
    if len(groups) == 1:
        ## Not defined
        eed = None
    else:    
        ## TPR
        TPR_diff = None
        tmp = list(filter(lambda x: x[1] is not None, groups_TPR_dict.items()))
        if len(tmp):
            max_TPR = list(max(tmp, key=lambda x: x[1])) ## tuple (group,sel_rate)
            max_TPR[0] = str(max_TPR[0])+', \n TP'
            max_TPR[1] = max_TPR[1]
            if len(tmp)!=1:            
                min_TPR = list(min(tmp, key=lambda x: x[1]))
                min_TPR[0] = str(min_TPR[0])+', \n TP'
                min_TPR[1] = min_TPR[1]
                TPR_diff = max_TPR[1] - min_TPR[1]                     
        ##
        if TPR_diff is not None:
            eed = TPR_diff
        else:
            ## Not defined
            eed = None ## not defined
    return eed

def predictive_parity_value_difference_(df, sensitive_feature):
    groups = list(df[sensitive_feature].unique())
    if 'Unknown' in groups:
        groups.remove('Unknown')
    groups_dict = {}
    for group in groups:
        predicted_value_counts = df[df[sensitive_feature] == group]["Predicted_Result"].value_counts()
        ## TPR
        TP = len(df.loc[(df["Predicted_Result"] == 1) & 
                                   (df["TARGET"] == 1) & 
                                   (df[sensitive_feature] == group)])        
        if 1 in predicted_value_counts:
            P_pred = int(predicted_value_counts[1])
        else:
            P_pred = 0  
        if P_pred:
            groups_dict[group] = TP/P_pred
        else:
            groups_dict[group] = None        
    if len(groups) == 1:
        ## Not defined
        ppv = None
    else:    
        ## PPV
        PPV_diff = None
        tmp = list(filter(lambda x: x[1] is not None, groups_dict.items()))
        if len(tmp):
            max_PPV = list(max(tmp, key=lambda x: x[1])) ## tuple (group,sel_rate)
            max_PPV[0] = str(max_PPV[0])+', \n TP'
            max_PPV[1] = max_PPV[1]
            if len(tmp)!=1:            
                min_PPV = list(min(tmp, key=lambda x: x[1]))
                min_PPV[0] = str(min_PPV[0])+', \n TP'
                min_PPV[1] = min_PPV[1]
                PPV_diff = max_PPV[1] - min_PPV[1]                     
        ##
        if PPV_diff is not None:
            ppv = PPV_diff
        else:
            ## Not defined
            ppv = None ## not defined
    return ppv

def predictive_parity_value_(df, sensitive_feature):
    groups = list(df[sensitive_feature].unique())
    groups_dict = {}
    for group in groups:
        TP = len(df.loc[(df["Predicted_Result"] == 1) & 
                                   (df["TARGET"] == 1) & 
                                   (df[sensitive_feature] == group)])
        P_pred = df[df[sensitive_feature] == group]["Predicted_Result"].value_counts()[1]
        ##
        groups_dict[group] = TP/P_pred
    ##    
    ppv = min(groups_dict.values()) - max(groups_dict.values())
    return ppv

def conditional_demographic_disparity_(df, sensitive_feature, pos_label):
    cdd = conditional_demographic_disparity(df['Predicted_Result'], 
                                            y_pred = df['Predicted_Result'], 
                                            prot_attr = df[sensitive_feature], 
                                            pos_label = pos_label)
    return cdd

def theil_index_(df, pos_label):
    return generalized_entropy_error(df["TARGET"], 
                                        y_pred = df['Predicted_Result'], 
                                        alpha = 1,
                                        pos_label = pos_label)

def consistency_score_(X, Y, k):
    return consistency_score(X,Y,k)

def group_fairness(sensitive_attrs, train_df_test_bin):
    """
        sensitive_attrs:      List of sensitive attributes
        train_df_test_binned: pandas.DataFrame of binned test set 
    """
    fairness_metrics_per_feature = {} 
    fairness_metrics_per_feature["Feature"] = []
    fairness_metrics_per_feature["DemographicParityRatio"] = []
    fairness_metrics_per_feature["ConditionalDemographicDisparity"] = []
    fairness_metrics_per_feature["EqualOpportunityDifference"] = []
    fairness_metrics_per_feature["AverageOddsDifference"] = []
    fairness_metrics_per_feature["PredictiveParity"] = []
    for attr in sensitive_attrs:
        fairness_metrics_per_feature['Feature'].append(attr)
        ## DEMOGRAPHIC PARITY (DP) RATIO
        dp = demographic_parity_ratio_(train_df_test_bin, attr)
        fairness_metrics_per_feature["DemographicParityRatio"].append(dp)
        ## CONDITIONAL DEMOGRAPHIC DISPARITY (CDD)
        CDD = conditional_demographic_disparity_(train_df_test_bin, attr, 1)
        fairness_metrics_per_feature['ConditionalDemographicDisparity'].append(CDD)

        ## EQUAL OPPORTUNITY DIFFERENCE
        eed = equal_opportunity_difference_(train_df_test_bin, attr)
        fairness_metrics_per_feature['EqualOpportunityDifference'].append(eed)

        ## AVERAGE ODDS DIFFERENCE
        aod = average_odds_difference_(train_df_test_bin, attr)
        fairness_metrics_per_feature['AverageOddsDifference'].append(aod)

        ## PREDICTIVE PARITY
        pp = predictive_parity_value_difference_(train_df_test_bin, attr)
        fairness_metrics_per_feature['PredictiveParity'].append(pp)
    return fairness_metrics_per_feature