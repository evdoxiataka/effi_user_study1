from fairlearn.metrics import MetricFrame
from fairlearn.metrics import selection_rate

from aif360.sklearn.metrics import conditional_demographic_disparity, consistency_score, generalized_entropy_error

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
    dp = min(groups_dict.values()) / max(groups_dict.values())
    return dp

def conditional_demographic_disparity_(df, sensitive_feature, pos_label):
    cdd = conditional_demographic_disparity(df['Predicted_Result'], 
                                            y_pred = df['Predicted_Result'], 
                                            prot_attr = df[sensitive_feature], 
                                            pos_label = pos_label)
    return cdd

def equal_opportunity_difference_(df, sensitive_feature):
    groups = list(df[sensitive_feature].unique())
    groups_dict = {}
    for group in groups:
        TP = len(df.loc[(df["Predicted_Result"] == 1) & 
                                   (df["TARGET"] == 1) & 
                                   (df[sensitive_feature] == group)])
        P = df[df[sensitive_feature] == group]["TARGET"].value_counts()[1]
        ##
        groups_dict[group] = TP/P
    ##    
    eed = min(groups_dict.values()) - max(groups_dict.values())
    return eed

def average_odds_difference_(df, sensitive_feature):
    groups = list(df[sensitive_feature].unique())
    groups_TPR_dict = {}
    groups_FPR_dict = {}
    for group in groups:
        TP = len(df.loc[(df["Predicted_Result"] == 1) & 
                                   (df["TARGET"] == 1) & 
                                   (df[sensitive_feature] == group)])
        P = df[df[sensitive_feature] == group]["TARGET"].value_counts()[1]
        FP = len(df.loc[(df["Predicted_Result"] == 1) & 
                                    (df["TARGET"] == 0) & 
                                   (df[sensitive_feature] == group)])
        N = df[df[sensitive_feature] == group]["TARGET"].value_counts()[0]
        ##
        groups_TPR_dict[group] = TP/P
        groups_FPR_dict[group] = FP/N
    ##    
    aod = ((min(groups_TPR_dict.values()) - max(groups_TPR_dict.values())) + (min(groups_FPR_dict.values()) - max(groups_FPR_dict.values())))/2.
    return aod

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
        pp = predictive_parity_value_(train_df_test_bin, attr)
        fairness_metrics_per_feature['PredictiveParity'].append(pp)
    return fairness_metrics_per_feature