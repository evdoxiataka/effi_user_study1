import numpy as np
import pandas as pd
from sklearn import metrics

from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def manipulate_categ_values(df):
    ## Some categorical data manipulation
    df.replace({'NAME_CONTRACT_TYPE' : { 'Cash loans' :'Fixed', 'Revolving loans' : 'Not Fixed'}},inplace=True)
    df.replace({'FLAG_OWN_CAR' : { "N" :'No', "Y" : 'Yes'}},inplace=True)
    df.replace({'FLAG_OWN_REALTY' : { "N" :'No', "Y" : 'Yes'}},inplace=True)
    df.replace({'FLAG_MOBIL' : { 0 :'No', 1 : 'Yes'}},inplace=True)
    df.replace({'FLAG_EMP_PHONE' : { 0 :'No', 1 : 'Yes'}},inplace=True)
    df.replace({'FLAG_WORK_PHONE' : { 0 :'No', 1 : 'Yes'}},inplace=True)
    df.replace({'FLAG_CONT_MOBILE' : { 0 :'No', 1 : 'Yes'}},inplace=True)
    df.replace({'FLAG_PHONE' : { 0 :'No', 1 : 'Yes'}},inplace=True)
    df.replace({'FLAG_EMAIL' : { 0 :'No', 1 : 'Yes'}},inplace=True)
    df.replace({'REG_REGION_NOT_LIVE_REGION' : { 0 :'Same', 1 : 'Different'}},inplace=True)
    df.replace({'REG_REGION_NOT_WORK_REGION' : { 0 :'Same', 1 : 'Different'}},inplace=True)
    df.replace({'LIVE_REGION_NOT_WORK_REGION' : { 0 :'Same', 1 : 'Different'}},inplace=True)
    df.replace({'REG_CITY_NOT_LIVE_CITY' : { 0 :'Same', 1 : 'Different'}},inplace=True)
    df.replace({'REG_CITY_NOT_WORK_CITY' : { 0 :'Same', 1 : 'Different'}},inplace=True)
    df.replace({'LIVE_CITY_NOT_WORK_CITY' : { 0 :'Same', 1 : 'Different'}},inplace=True)
    df.replace({'CODE_GENDER' : { "F" :'Female', "M" :'Male'}},inplace=True)
    df.replace({'YEARS_EMPLOYED' : { -1001 :np.nan}}, inplace=True)
    
def binning(train_df_test_or, train_df_test):
    """
    train_df_test_or: pandas.DataFrame with data before hot encoding and missing values imputation
    train_df_test:    pandas.DataFrame with data after hot encoding and missing values imputation
    """
    ## binning
    train_df_test_bin = train_df_test_or.copy()
    for j in train_df_test_bin['AGE'].unique().tolist():
        if j<=25.:
            train_df_test_bin['AGE'].replace([j],"Young Adults (18-25)",inplace=True)
        elif j>25. and j<=40.:
            train_df_test_bin['AGE'].replace([j],"Adults (26-40)",inplace=True)
        elif j>40. and j<=60.:
            train_df_test_bin['AGE'].replace([j],"Middle Age Adults (41 - 60)",inplace=True)
        elif j>60.:
            train_df_test_bin['AGE'].replace([j],"Older Adults (60+)",inplace=True)
    ##
    train_df_test_bin['OBS_30_CNT_SOCIAL_CIRCLE'] = train_df_test['OBS_30_CNT_SOCIAL_CIRCLE'].tolist()
    for j in train_df_test_bin['OBS_30_CNT_SOCIAL_CIRCLE'].unique().tolist():
        if j<=5:
            train_df_test_bin['OBS_30_CNT_SOCIAL_CIRCLE'].replace([j],"<=5",inplace=True)
        elif j<=9:
            train_df_test_bin['OBS_30_CNT_SOCIAL_CIRCLE'].replace([j],"5-9",inplace=True)
        else:
            train_df_test_bin['OBS_30_CNT_SOCIAL_CIRCLE'].replace([j],">=10",inplace=True)

    ##
    train_df_test_bin['DEF_30_CNT_SOCIAL_CIRCLE'] = train_df_test['DEF_30_CNT_SOCIAL_CIRCLE'].tolist()
    for j in train_df_test_bin['DEF_30_CNT_SOCIAL_CIRCLE'].unique().tolist():
        if j<=1:
            train_df_test_bin['DEF_30_CNT_SOCIAL_CIRCLE'].replace([j],"0-2",inplace=True)
        else:
            train_df_test_bin['DEF_30_CNT_SOCIAL_CIRCLE'].replace([j],">=2",inplace=True)

    ##
    train_df_test_bin['OBS_60_CNT_SOCIAL_CIRCLE'] = train_df_test['OBS_60_CNT_SOCIAL_CIRCLE'].tolist()
    for j in train_df_test_bin['OBS_60_CNT_SOCIAL_CIRCLE'].unique().tolist():
        if j<=5:
            train_df_test_bin['OBS_60_CNT_SOCIAL_CIRCLE'].replace([j],"<=5",inplace=True)
        elif j<=9:
            train_df_test_bin['OBS_60_CNT_SOCIAL_CIRCLE'].replace([j],"5-9",inplace=True)
        else:
            train_df_test_bin['OBS_60_CNT_SOCIAL_CIRCLE'].replace([j],">=10",inplace=True)

    ##
    train_df_test_bin['DEF_60_CNT_SOCIAL_CIRCLE'] = train_df_test['DEF_60_CNT_SOCIAL_CIRCLE'].tolist()
    for j in train_df_test_bin['DEF_60_CNT_SOCIAL_CIRCLE'].unique().tolist():
        if j<=1:
            train_df_test_bin['DEF_60_CNT_SOCIAL_CIRCLE'].replace([j],"0-2",inplace=True)
        else:
            train_df_test_bin['DEF_60_CNT_SOCIAL_CIRCLE'].replace([j],">=2",inplace=True)
    return train_df_test_bin
    
def k_means_optimize_parameter(array_of_vectors, parameters, metric):
    """
        array_of_vectors: List of lists containing the vectors 
        parameters: List of number of clusters for grid searching
        metric: Str in {'mean','median'}
    """
    best_score = -1
    silhouette_scores = []
    # evaluation based on silhouette_score
    for p in parameters:
        if metric=='mean':
            kmeans_model = KMeans(n_clusters=p, init='k-means++', random_state=13, n_init='auto')
        elif metric=='median':
            kmeans_model = KMedoids(n_clusters=p, init='k-medoids++', random_state=13)
        kmeans_model.fit(array_of_vectors)          # fit model on dataset, this will find clusters based on parameter p
        ss = metrics.silhouette_score(array_of_vectors, kmeans_model.labels_)   # calculate silhouette_score
        silhouette_scores += [ss]       # store all the scores
        # print('Parameter:', p, 'Score', ss)
        # check p which has the best score
        if ss > best_score:
            best_score = ss
            best_grid = p
    return silhouette_scores, best_score, best_grid

def k_means_pca(array_of_vectors, n_clusters, metric, n_components=2):
    """
        array_of_vectors: List of lists containing the vectors 
        n_clusters: Number of clusters for k-means
        n_components: number of components for PCA
        metric: Str in {'mean','median'}
    """
    if metric=='mean':
        kmeans_model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=13, n_init='auto')
    elif metric=='median':
        kmeans_model = KMedoids(n_clusters=n_clusters, init='k-medoids++', random_state=13)
    kmeans_model.fit(array_of_vectors)
    cluster_ids = kmeans_model.labels_
    cluster_centroids = kmeans_model.cluster_centers_
    ##
    pca = PCA(n_components=2)
    array_of_vectors_standard = StandardScaler().fit_transform(array_of_vectors)
    pca.fit(array_of_vectors_standard)
    array_of_vectors_pca = pca.transform(array_of_vectors_standard)
    return cluster_ids, cluster_centroids, array_of_vectors_pca

def add_cma_data(df_group, group_fair, df_indiv, indiv_fair, df_acc, sensitive_attrs, fs):
    ## expand dfs with one CMA column per metric
    for gf in group_fair:
        df_group['CMA_'+gf] = [None]*len(df_group[gf])
    for idf in indiv_fair:
        df_indiv['CMA_'+idf] = [None]*len(df_indiv[idf])
    df_acc['CMA_accuracy'] = [None]*len(df_acc['accuracy'])
    ##
    for p,p_id in enumerate(df_group['participant_id'].unique()):     
        if isinstance(p_id, str):
            ## GROUP FAIRNESS
            for i,sens_attr in enumerate(sensitive_attrs):           
                for j,gf in enumerate(group_fair):
                    for k,fs_i in enumerate(fs):
                        cma = df_group.loc[(df_group['participant_id']==p_id) & (df_group['Feature']==sens_attr) & (df_group['fs']==fs_i),[gf]].expanding().mean()[gf].tolist()
                        df_group.loc[(df_group['participant_id']==p_id) & (df_group['Feature']==sens_attr) & (df_group['fs']==fs_i),['CMA_'+gf]] = cma
            ## INDIVIDUAL FAIRNESS
            for i,idf in enumerate(indiv_fair):
                for k,fs_i in enumerate(fs):
                    cma = df_indiv.loc[(df_indiv['participant_id']==p_id) & (df_indiv['fs']==fs_i),[idf]].expanding().mean()[idf].tolist()
                    df_indiv.loc[(df_indiv['participant_id']==p_id) & (df_indiv['fs']==fs_i),['CMA_'+idf]] = cma
            ## ACCURACY
            for k,fs_i in enumerate(fs):
                cma = df_acc.loc[(df_acc['participant_id']==p_id) & (df_acc['fs']==fs_i),['accuracy']].expanding().mean()['accuracy'].tolist()
                df_acc.loc[(df_acc['participant_id']==p_id) & (df_acc['fs']==fs_i),['CMA_accuracy']] = cma

def get_percentage_change_oneoff(df_group, group_fair, df_indiv, indiv_fair, df_acc, sensitive_attrs, fs):
    perc_change_dict = {} ## percentage change (value-baseline)/baseline*100
    for p,p_id in enumerate(df_group['participant_id'].unique()):     
        if isinstance(p_id, str):
            ## GROUP FAIRNESS
            df_p = df_group[df_group['participant_id']==p_id]
            for i,sens_attr in enumerate(sensitive_attrs):          
                for j,gf in enumerate(group_fair):
                    for k,fs_i in enumerate(fs):
                        ## get baseline value
                        df_p_null = df_group[df_group['participant_id'].isnull()]  
                        df = df_p_null[df_p_null['Feature']==sens_attr]
                        df = df[df['fs']==fs_i] 
                        baseline = df[gf].tolist()[0]
                        ## get diff from baseline
                        df = df_p[df_p['Feature']==sens_attr]
                        df = df[df['fs']==fs_i]
                        # av_diff_vec.append(((df[gf].tolist()[0]-baseline)/baseline)*100)
                        perc_change_dict[sens_attr+'_'+gf] = ((df[gf].tolist()[0]-baseline)/abs(baseline))*100
            ## INDIVIDUAL FAIRNESS
            df_p = df_indiv[df_indiv['participant_id']==p_id]
            for i,idf in enumerate(indiv_fair):
                for k,fs_i in enumerate(fs):
                    ## get baseline value
                    df_p_null = df_indiv[df_indiv['participant_id'].isnull()]
                    df = df_p_null[df_p_null['fs']==fs_i]                
                    baseline = df[idf].tolist()[0]
                    ## get diff from baseline
                    df = df_p[df_p['fs']==fs_i]
                    # av_diff_vec.append(((df[idf].tolist()[0]-baseline)/baseline)*100)
                    perc_change_dict[idf] = ((df[idf].tolist()[0]-baseline)/abs(baseline))*100
            ## ACCURACT
            df_p = df_acc[df_indiv['participant_id']==p_id]
            for k,fs_i in enumerate(fs):
                ## get baseline value
                df_p_null = df_acc[df_acc['participant_id'].isnull()]
                df = df_p_null[df_p_null['fs']==fs_i]                
                baseline = df['accuracy'].tolist()[0]
                ## get diff from baseline
                df = df_p[df_p['fs']==fs_i]
                # av_diff_vec.append(((df['accuracy'].tolist()[0]-baseline)/baseline)*100)
                perc_change_dict['accuracy'] = ((df['accuracy'].tolist()[0]-baseline)/abs(baseline))*100
    return pd.DataFrame([perc_change_dict])

def get_percentage_change_IML(df_group, group_fair, df_indiv, indiv_fair, sensitive_attrs, fs):
    # av_diff_vecs = [] ## Average of Differences from Baseline Values
    # cma_av_diff_vecs = [] ## Average of Differences of Cumulative Moving Average lines from Baseline Values
    perc_change_dict = {} ## percentage change (value-baseline)/baseline*100
    cma_perc_change_dict = {} 
    p_ids = []
    for p,p_id in enumerate(df_group['participant_id'].unique()):     
        if isinstance(p_id, str): 
            # av_diff_vec = []
            # cma_av_diff_vec = []
            p_ids.append(p_id)
            ## GROUP FAIRNESS
            df_p = df_group[df_group['participant_id']==p_id]
            for i,sens_attr in enumerate(sensitive_attrs):          
                for j,gf in enumerate(group_fair):
                    diff = []
                    cma_diff = []
                    for k,fs_i in enumerate(fs):
                        ## get baseline value
                        df_p_null = df_group[df_group['participant_id'].isnull()]  
                        df = df_p_null[df_p_null['Feature']==sens_attr]
                        df = df[df['fs']==fs_i] 
                        baseline = df[gf].tolist()[0]
                        ## get diff from baseline
                        df = df_p[df_p['Feature']==sens_attr]
                        df = df[df['fs']==fs_i]
                        diff.extend([df[gf].tolist()[-1]-baseline])
                        ## get diff of last iteration in CMA from baseline
                        cma_diff.extend([df['CMA_'+gf].tolist()[-1]-baseline])
                    ###
                    # av_diff_vec.append(np.array(av_diff).mean())
                    # cma_av_diff_vec.append(np.array(cma_av_diff).mean())
                    if sens_attr+'_'+gf not in perc_change_dict:
                        perc_change_dict[sens_attr+'_'+gf] = []
                        cma_perc_change_dict[sens_attr+'_'+gf] = []
                    perc_change_dict[sens_attr+'_'+gf].append((diff[0]/abs(baseline))*100)
                    cma_perc_change_dict[sens_attr+'_'+gf].append((cma_diff[0]/abs(baseline))*100)
            ## INDIVIDUAL FAIRNESS
            df_p = df_indiv[df_indiv['participant_id']==p_id]
            for i,idf in enumerate(indiv_fair):
                diff = []
                cma_diff = []
                for k,fs_i in enumerate(fs):
                    ## get baseline value
                    df_p_null = df_indiv[df_indiv['participant_id'].isnull()]
                    df = df_p_null[df_p_null['fs']==fs_i]                
                    baseline = df[idf].tolist()[0]
                    ## get diff from baseline
                    df = df_p[df_p['fs']==fs_i]
                    diff.extend([df[idf].tolist()[-1]-baseline])
                    ## get diff of CMA from baseline
                    cma_diff.extend([df['CMA_'+idf].tolist()[-1]-baseline])
                    ###
                    # av_diff_vec.append(np.array(av_diff).mean())
                    # cma_av_diff_vec.append(np.array(cma_av_diff).mean())                
                if idf not in perc_change_dict:
                    perc_change_dict[idf] = []
                    cma_perc_change_dict[idf] = []
                perc_change_dict[idf].append((diff[0]/abs(baseline))*100)
                cma_perc_change_dict[idf].append((cma_diff[0]/abs(baseline))*100)
            # av_diff_vecs.append(av_diff_vec)
            # cma_av_diff_vecs.append(cma_av_diff_vec)
    perc_change_dict['participant_id'] = p_ids
    cma_perc_change_dict['participant_id'] = p_ids
    return pd.DataFrame(perc_change_dict), pd.DataFrame(cma_perc_change_dict)
    
class PCAForPandas(PCA):
    """This class is just a small wrapper around the PCA estimator of sklearn including normalization to make it 
    compatible with pandas DataFrames.
    """

    def __init__(self, **kwargs):
        self._z_scaler = StandardScaler()
        super(self.__class__, self).__init__(**kwargs)

        self._X_columns = None

    def fit(self, X, y=None):
        """Normalize X and call the fit method of the base class with numpy arrays instead of pandas data frames."""

        X = self._prepare(X)

        self._z_scaler.fit(X.values, y)
        z_data = self._z_scaler.transform(X.values, y)

        return super(self.__class__, self).fit(z_data, y)

    def fit_transform(self, X, y=None):
        """Call the fit and the transform method of this class."""

        X = self._prepare(X)

        self.fit(X, y)
        return self.transform(X, y)

    def transform(self, X, y=None):
        """Normalize X and call the transform method of the base class with numpy arrays instead of pandas data frames."""

        X = self._prepare(X)

        z_data = self._z_scaler.transform(X.values, y)

        transformed_ndarray = super(self.__class__, self).transform(z_data)

        pandas_df = pd.DataFrame(transformed_ndarray)
        pandas_df.columns = ["pca_{}".format(i) for i in range(len(pandas_df.columns))]

        return pandas_df

    def _prepare(self, X):
        """Check if the data is a pandas DataFrame and sorts the column names.

        :raise AttributeError: if pandas is not a DataFrame or the columns of the new X is not compatible with the 
                               columns from the previous X data
        """
        if not isinstance(X, pd.DataFrame):
            raise AttributeError("X is not a pandas DataFrame")

        X.sort_index(axis=1, inplace=True)

        if self._X_columns is not None:
            if self._X_columns != list(X.columns):
                raise AttributeError("The columns of the new X is not compatible with the columns from the previous X data")
        else:
            self._X_columns = list(X.columns)

        return X