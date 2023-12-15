import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# from matplotlib.ticker import FixedLocator

def get_x_axis_lim(df_group, group_fair, sensitive_attrs, fs):
    ## get max length of feedback to define lim of x-axis
    feed_len = []
    for p,p_id in enumerate(df_group['participant_id'].unique()):
        df_p = df_group[df_group['participant_id']==p_id]
        for i,sens_attr in enumerate(sensitive_attrs):    
            for j,gf in enumerate(group_fair):
                for k,fs_i in enumerate(fs):
                    df = df_p[df_p['Feature']==sens_attr]
                    df = df[df['fs']==fs_i]
                    feed_len.append(len(df))
    return max(feed_len)
    
def joint_plot_all_participants(title, folder, filename, image_type, sensitive_attrs, group_fair, group_fair_codes, indiv_fair, fs, df_group, df_indiv, df_acc, colors, show_timeseries, show_cma):
    matplotlib.rcParams.update({'font.size': 26})
    xlim = get_x_axis_lim(df_group, group_fair, sensitive_attrs, fs)
    ## 
    fig, axes = plt.subplots(len(sensitive_attrs)+1, len(group_fair), figsize=(50, 45), layout="constrained")
    fig.suptitle(title)

    ## TIME SERIES
    if show_timeseries:
        flag = True
        ## GROUP FAIRNESS
        for i,sens_attr in enumerate(sensitive_attrs):    
            for j,gf in enumerate(group_fair):
                for k,fs_i in enumerate(fs):
                    df = df_group[df_group['Feature']==sens_attr]
                    df = df[df['fs']==fs_i]
                    ## draw one curve per participant
                    for p,p_id in enumerate(df['participant_id'].unique()):
                        df_p = df[df['participant_id']==p_id]
                        ## draw time series
                        if isinstance(p_id, str):## not None                    
                            line, = axes[i,j].plot(df_p['iteration'], df_p[gf], c = colors[2])  
                            if flag:
                                line.set_label('Feedback Integration')
                                flag = False
        ## INDIVIDUAL FAIRNESS
        for i,idf in enumerate(indiv_fair):
            for k,fs_i in enumerate(fs):
                df = df_indiv[df_indiv['fs']==fs_i]
                ## draw one curve per participant
                for p,p_id in enumerate(df['participant_id'].unique()):
                    df_p = df[df['participant_id']==p_id]
                    ## draw time series
                    if isinstance(p_id, str):## not None
                        axes[len(sensitive_attrs),i].plot(df_p['iteration'], df_p[idf], c = colors[2])
        ## ACCURACY
        for k,fs_i in enumerate(fs):
            df = df_acc[df_acc['fs']==fs_i]
            ## draw one curve per participant
            for p,p_id in enumerate(df['participant_id'].unique()):
                df_p = df[df['participant_id']==p_id]
                ## draw time series
                if isinstance(p_id, str):## not None 
                    axes[len(sensitive_attrs),len(indiv_fair)].plot(df_p['iteration'], df_p['accuracy'], c = colors[2])
    ## CUMULATIVE MOVING AVERAGE LINES
    if show_cma:  
        flag = True
        ## GROUP FAIRNESS
        for i,sens_attr in enumerate(sensitive_attrs):    
            for j,gf in enumerate(group_fair):
                for k,fs_i in enumerate(fs):
                    df = df_group[df_group['Feature']==sens_attr]
                    df = df[df['fs']==fs_i]
                    ## draw one curve per participant
                    for p,p_id in enumerate(df['participant_id'].unique()):
                        df_p = df[df['participant_id']==p_id]
                        ## draw cma line
                        if isinstance(p_id, str):## not None
                            line, = axes[i,j].plot(df_p['iteration'], df_p['CMA_'+gf], c = colors[0], linewidth=1)
                            if flag:
                                line.set_label('CMA of Feedback Integration')
                                flag = False
        ## INDIVIDUAL FAIRNESS
        for i,idf in enumerate(indiv_fair):
            for k,fs_i in enumerate(fs):
                df = df_indiv[df_indiv['fs']==fs_i]
                ## draw one curve per participant
                for p,p_id in enumerate(df['participant_id'].unique()):
                    df_p = df[df['participant_id']==p_id]
                    ## draw cma line
                    if isinstance(p_id, str):## not None                        
                        axes[len(sensitive_attrs),i].plot(df_p['iteration'], df_p['CMA_'+idf], c = colors[0], linewidth=1)
        ## ACCURACY
        for k,fs_i in enumerate(fs):
            df = df_acc[df_acc['fs']==fs_i]
            ## draw one curve per participant
            for p,p_id in enumerate(df['participant_id'].unique()):
                df_p = df[df['participant_id']==p_id]
                ## ddraw cma line
                if isinstance(p_id, str):## not None 
                    axes[len(sensitive_attrs),len(indiv_fair)].plot(df_p['iteration'], df_p['CMA_accuracy'], c = colors[0], linewidth=1)
    ## BASELINE LINES
    flag = True
    ## GROUP FAIRNESS
    for i,sens_attr in enumerate(sensitive_attrs):    
        for j,gf in enumerate(group_fair):
            for k,fs_i in enumerate(fs):
                df = df_group[df_group['Feature']==sens_attr]
                df = df[df['fs']==fs_i]
                ## draw baseline
                df_p = df[df['participant_id'].isnull()]                
                line, = axes[i,j].plot([l for l in range(xlim)], [df_p[gf].tolist()[0]]*xlim, c='black')
                if flag:
                    line.set_label('Baseline - No Feedback')
                    flag = False
            axes[i,j].set_xlabel("Iteration")
            axes[i,j].set_ylabel(group_fair_codes[j])
            axes[i,j].set_title(sens_attr)      
    ## INDIVIDUAL FAIRNESS
    for i,idf in enumerate(indiv_fair):
        for k,fs_i in enumerate(fs):
            df = df_indiv[df_indiv['fs']==fs_i]            
            ## draw baseline
            df_p = df[df['participant_id'].isnull()]                
            axes[len(sensitive_attrs),i].plot([l for l in range(xlim)], [df_p[idf].tolist()[0]]*xlim, c='black') 
        axes[len(sensitive_attrs),i].set_xlabel("Iteration")
        axes[len(sensitive_attrs),i].set_ylabel(idf)
    ## ACCURACY
    for k,fs_i in enumerate(fs):
        df = df_acc[df_acc['fs']==fs_i]        
        ## draw baseline
        df_p = df[df['participant_id'].isnull()]                
        axes[len(sensitive_attrs),len(indiv_fair)].plot([l for l in range(xlim)], [df_p['accuracy'].tolist()[0]]*xlim, c = 'black')   
    
    axes[0,0].legend(bbox_to_anchor=(0.0,2.0),loc='upper left') 
    axes[len(sensitive_attrs),len(indiv_fair)].set_xlabel("Iteration")
    axes[len(sensitive_attrs),len(indiv_fair)].set_ylabel('Accuracy %')    
    fig.delaxes(axes[len(sensitive_attrs),len(indiv_fair)+1])
    fig.delaxes(axes[len(sensitive_attrs),len(indiv_fair)+2])
    
    fig.savefig(folder+filename+image_type, dpi=300)
    plt.show()

def line_graphs_of_participant(title, folder, image_type, sensitive_attrs, group_fair, group_fair_codes, 
                               indiv_fair, fs, df_group, df_indiv, df_acc, colors, show_cma, participant_id):
    """
        show_cma: Boolean, if True plot Cumulative Moving Average Lines
    """
    title_code = {'CODE_GENDER':'GENDER','AGE':'AGE','NAME_FAMILY_STATUS':'MARIT. STAT.'}
    matplotlib.rcParams.update({'font.size': 26})
    xlim = get_x_axis_lim(df_group, group_fair, sensitive_attrs, fs)
    for p,p_id in enumerate(df_group['participant_id'].unique()):
        if isinstance(p_id, str) and p_id == participant_id:        
            ##
            fig, axes = plt.subplots(len(sensitive_attrs), len(group_fair), figsize=(25, 25), layout="constrained")
            fig.suptitle(title.format(p_id))
            ## GROUP FAIRNESS
            df_p = df_group[df_group['participant_id']==p_id]
            for i,sens_attr in enumerate(sensitive_attrs):    
                for j,gf in enumerate(group_fair):
                    for k,fs_i in enumerate(fs):
                        ## draw time series
                        df = df_p[df_p['Feature']==sens_attr]
                        df = df[df['fs']==fs_i]                   
                        axes[i,j].plot(df['iteration'], df[gf], c = colors[2], linewidth=2, label = 'Feedback Integration') #colors[j]    
                        ## draw cma line
                        if show_cma:
                            axes[i,j].plot(df['iteration'], df['CMA_'+gf], c = colors[0], linewidth=2, label = 'CMA of Feedback Integration')
                        ## draw baseline
                        df_p_null = df_group[df_group['participant_id'].isnull()]  
                        df = df_p_null[df_p_null['Feature']==sens_attr]
                        df = df[df['fs']==fs_i] 
                        axes[i,j].plot([l for l in range(xlim)], [df[gf].tolist()[0]]*xlim,c='black', label = 'Baseline - No Feedback')             
                    axes[i,j].set_xlabel("Iteration")
                    axes[i,j].set_ylabel(group_fair_codes[j])
                    axes[i,j].set_title(title_code[sens_attr])   
            axes[0,0].legend(bbox_to_anchor=(0.3,1.0),loc='upper left')             
            fig.savefig(folder+'{}'.format(p_id)+image_type, dpi=300)
            plt.show()
            
def plots_per_participant(title, folder, image_type, sensitive_attrs, group_fair, group_fair_codes, indiv_fair, fs, df_group, df_indiv, df_acc, colors, show_cma):
    """
        show_cma: Boolean, if True plot Cumulative Moving Average Lines
    """
    matplotlib.rcParams.update({'font.size': 26})
    xlim = get_x_axis_lim(df_group, group_fair, sensitive_attrs, fs)
    for p,p_id in enumerate(df_group['participant_id'].unique()):
        if isinstance(p_id, str):        
            ##
            fig, axes = plt.subplots(len(sensitive_attrs)+1, len(group_fair), figsize=(60, 45), layout="constrained")
            fig.suptitle(title.format(p_id))
            ## GROUP FAIRNESS
            df_p = df_group[df_group['participant_id']==p_id]
            for i,sens_attr in enumerate(sensitive_attrs):    
                for j,gf in enumerate(group_fair):
                    for k,fs_i in enumerate(fs):
                        ## draw time series
                        df = df_p[df_p['Feature']==sens_attr]
                        df = df[df['fs']==fs_i]                   
                        axes[i,j].plot(df['iteration'], df[gf], c = colors[2], linewidth=2, label = 'Feedback Integration') #colors[j]    
                        ## draw cma line
                        if show_cma:
                            axes[i,j].plot(df['iteration'], df['CMA_'+gf], c = colors[0], linewidth=2, label = 'CMA of Feedback Integration')
                        ## draw baseline
                        df_p_null = df_group[df_group['participant_id'].isnull()]  
                        df = df_p_null[df_p_null['Feature']==sens_attr]
                        df = df[df['fs']==fs_i] 
                        axes[i,j].plot([l for l in range(xlim)], [df[gf].tolist()[0]]*xlim,c='black', label = 'Baseline - No Feedback')             
                    axes[i,j].set_xlabel("Iteration")
                    axes[i,j].set_ylabel(group_fair_codes[j])
                    axes[i,j].set_title(sens_attr)        
    #                 axes[i,j].grid(axis = 'x',which='major')
    #                 axes[i,j].xaxis.set_major_locator(FixedLocator([it for it in range(xlim)]))
    #                 axes[i,j].set_xlim(0,xlim)
    #                 axes[i,j].xaxis.set_ticklabels([])        
            ## INDIVIDUAL FAIRNESS
            df_p = df_indiv[df_indiv['participant_id']==p_id]
            for i,idf in enumerate(indiv_fair):
                for k,fs_i in enumerate(fs):
                    ## draw time series
                    df = df_p[df_p['fs']==fs_i]
                    axes[len(sensitive_attrs),i].plot(df['iteration'], df[idf], c = colors[2], linewidth=2)#colors[len(group_fair)+i]
                    ## draw cma line
                    if show_cma:
                        axes[len(sensitive_attrs),i].plot(df['iteration'], df['CMA_'+idf], c = colors[0], linewidth=2)
                    ## draw baseline
                    df_p_null = df_indiv[df_indiv['participant_id'].isnull()]
                    df = df_p_null[df_p_null['fs']==fs_i]
                    axes[len(sensitive_attrs),i].plot([l for l in range(xlim)], [df[idf].tolist()[0]]*xlim, c='black') 
    #             axes[len(sensitive_attrs),i].grid(axis = 'x',which='major')
    #             axes[len(sensitive_attrs),i].xaxis.set_major_locator(FixedLocator([it for it in range(xlim)]))
                axes[len(sensitive_attrs),i].set_xlabel("Iteration")
                axes[len(sensitive_attrs),i].set_ylabel(idf)
    #             axes[len(sensitive_attrs),i].set_xlim(0,xlim)
    #             axes[len(sensitive_attrs),i].xaxis.set_ticklabels([])
            ## ACCURACY
            df_p = df_acc[df_acc['participant_id']==p_id]
            for k,fs_i in enumerate(fs):
                ## draw time series
                df = df_p[df_p['fs']==fs_i]
                axes[len(sensitive_attrs),len(indiv_fair)].plot(df['iteration'], df['accuracy'], c = colors[2], linewidth=2)#colors[len(group_fair)+len(indiv_fair)]
                ## draw cma line
                if show_cma:
                    axes[len(sensitive_attrs),len(indiv_fair)].plot(df['iteration'], df['CMA_accuracy'], c = colors[0], linewidth=2)
                ## draw baseline
                df = df_acc[df_acc['participant_id'].isnull()]
                df = df[df['fs']==fs_i]
                axes[len(sensitive_attrs),len(indiv_fair)].plot([l for l in range(xlim)], [df['accuracy'].tolist()[0]]*xlim, c = 'black')             
    #         axes[len(sensitive_attrs),len(indiv_fair)].grid(axis = 'x',which='major')
    #         axes[len(sensitive_attrs),len(indiv_fair)].xaxis.set_major_locator(FixedLocator([it for it in range(xlim)]))
            axes[0,0].legend(bbox_to_anchor=(0.0,2.0),loc='upper left') 
            axes[len(sensitive_attrs),len(indiv_fair)].set_xlabel("Iteration")
            axes[len(sensitive_attrs),len(indiv_fair)].set_ylabel('Accuracy %')
    #         axes[len(sensitive_attrs),len(indiv_fair)].set_xlim(0,xlim)
    #         axes[len(sensitive_attrs),len(indiv_fair)].xaxis.set_ticklabels([])
            fig.delaxes(axes[len(sensitive_attrs),len(indiv_fair)+1])
            fig.delaxes(axes[len(sensitive_attrs),len(indiv_fair)+2])
            
            fig.savefig(folder+'{}'.format(p_id)+image_type, dpi=300)
            plt.show()

def plot_silhouette_scores(silhouette_scores, parameters):
    matplotlib.rcParams.update({'font.size': 16})
    ## 
    fig, axes = plt.subplots(1, 1, figsize=(8, 5), layout="constrained")
    fig.suptitle('Silhouette Score', fontweight='bold')
    ##
    axes.bar(range(len(silhouette_scores)), list(silhouette_scores), align='center', color='#722f59', width=0.5)
    axes.set_xticks(range(len(silhouette_scores)))
    axes.set_xticklabels(list(parameters))
    # plt.title('Silhouette Score', fontweight='bold')
    axes.set_xlabel('Number of Clusters')
    plt.show()

def plot_clusters(array_of_vectors_pca, cluster_ids, colors):
    matplotlib.rcParams.update({'font.size': 16})
    n_clusters = len(np.unique(cluster_ids))
    max_cluster_id = max(cluster_ids)
    ## 
    fig, axes = plt.subplots(1, 1)
    fig.suptitle('Clusters K-Means {}'.format(n_clusters), fontweight='bold')
    ##
    scat = axes.scatter(array_of_vectors_pca[:, 0], array_of_vectors_pca[:, 1], 
                c = cluster_ids, 
                cmap = matplotlib.colors.ListedColormap(colors[0:max_cluster_id+1]))
    cb = plt.colorbar(scat,ax=axes)
    loc = np.arange(0, max_cluster_id, max_cluster_id/float(max_cluster_id+1))
    cb.set_ticks(loc)
    cb.set_ticklabels(['cluster '+str(i) for i in range(max_cluster_id+1)])
    plt.show()

def perc_change_plots_per_cluster(perc_ch_df, cluster_df, title, file_name, folder, attrs, attrs_codes, group_fair, group_fair_codes):
    colors = plt.cm.tab10
    matplotlib.rcParams.update({'font.size': 30})
    ##
    for cl in cluster_df['cluster_id'].unique().tolist():
        if len(group_fair) == 2 and 'DemographicParityRatio' in group_fair and 'AverageOddsDifference' in group_fair:
            fig, axes = plt.subplots(len(group_fair), len(attrs), figsize=(25, 16), layout="constrained")
        else:
            fig, axes = plt.subplots(len(group_fair), len(attrs), figsize=(25, 20), layout="constrained")
        fig.suptitle(title.format(str(cl)))
        part_in_cl = cluster_df[cluster_df['cluster_id'] == cl]['participant_id'].tolist()
        for i,attr in enumerate(attrs):
            for j,fm in enumerate(group_fair):
                df = perc_ch_df[perc_ch_df['participant_id'].isin(part_in_cl)][['participant_id',attr+'_'+fm]].sort_values(by=[attr+'_'+fm], ascending=False)            
                axes[j,i].bar(df['participant_id'], df[attr+'_'+fm], color = list(colors(cl+1)))
                axes[j,i].set_xlabel("Participants\n in desc. order of perc. ch.")
                axes[j,i].set_ylabel(group_fair_codes[group_fair.index(fm)]+'\n Perc. Ch. %')
                axes[j,i].set_title(attrs_codes[i])
                if len(group_fair) == 2 and 'DemographicParityRatio' in group_fair and 'AverageOddsDifference' in group_fair:
                    if fm == 'AverageOddsDifference':
                        axes[j,i].set_ylim(-11.5,6.) 
                    else:
                        if attr == 'CODE_GENDER':
                            axes[j,i].set_ylim(-1.,1.1)
                        elif attr == 'NAME_FAMILY_STATUS':
                            axes[j,i].set_ylim(-3.,2)
                        else:
                            axes[j,i].set_ylim(-6.5,2)
                axes[j,i].xaxis.set_ticklabels([])
                if i ==0 and j==0:
                    handl, lab = axes[j,i].get_legend_handles_labels()
                    by_label = dict(zip(lab, handl))
                    by_label = dict(sorted(by_label.items()))
                    
        fig.savefig(folder+file_name.format(str(cl)), dpi=300)

def perc_change_plots(perc_ch_df, title, file_name, folder, attrs, attrs_codes, group_fair, group_fair_codes):
    colors = plt.cm.tab10
    matplotlib.rcParams.update({'font.size': 30})
    # participant_color_map = {}
    # colors = cc.linear_bmy_10_95_c78[0:256:4]
    # for i,part in enumerate(perc_ch_df[['participant_id','CODE_GENDER_DemographicParityRatio']].sort_values(by=['CODE_GENDER_DemographicParityRatio'], ascending=False)['participant_id'].unique()):
    # #     participant_color_map[part] = colors[i]
    # for i,cl_id in enumerate(cluster_df.sort_values(by=['cluster_id'], ascending=False)['cluster_id'].unique()):
    #     participant_color_map[cl_id] = list(colors(i))
    ##
    if len(group_fair) == 2 and 'DemographicParityRatio' in group_fair and 'AverageOddsDifference' in group_fair:
        fig, axes = plt.subplots(len(group_fair), len(attrs), figsize=(25, 16), layout="constrained")
    else:
        fig, axes = plt.subplots(len(group_fair), len(attrs), figsize=(25, 25), layout="constrained")
    fig.suptitle(title)
    for i,attr in enumerate(attrs):
        for j,fm in enumerate(group_fair):
            if fm == 'indiv.':
                if attr == 'CODE_GENDER':
                    fm = 'consistency_10'
                    code = ' (+)'
                elif attr =='NAME_FAMILY_STATUS':
                    fm = 'theil_index'
                    code = ' (-)'
                else:
                    fig.delaxes(axes[len(group_fair)-1,len(attrs)-1])
                    continue
                df = perc_ch_df[['participant_id',fm]].sort_values(by=[fm], ascending=False)                
                axes[j,i].bar(df['participant_id'], df[fm])
                axes[j,i].set_xlabel("Participants\n in desc. order of perc. ch.")
                axes[j,i].set_ylabel(fm+code+'\n Perc. Ch. %')
                # axes[j,i].set_title(attrs_codes[i]) 
                axes[j,i].xaxis.set_ticklabels([])
            else:
                df = perc_ch_df[['participant_id',attr+'_'+fm]].sort_values(by=[attr+'_'+fm], ascending=False)
                # cls = [participant_color_map[part] for part in df['participant_id']]
                # cls = [participant_color_map[cluster_df[cluster_df['participant_id']==part]['cluster_id'].tolist()[0]] for part in df['participant_id']]
                # labels = ['cluster '+str(cluster_df[cluster_df['participant_id']==part]['cluster_id'].tolist()[0]) for part in df['participant_id']]
                axes[j,i].bar(df['participant_id'], df[attr+'_'+fm])#,color=cls,label = labels
                axes[j,i].set_xlabel("Participants\n in desc. order of perc. ch.")
                axes[j,i].set_ylabel(group_fair_codes[group_fair.index(fm)]+'\n Perc. Ch. %')
                axes[j,i].set_title(attrs_codes[i]) 
                axes[j,i].xaxis.set_ticklabels([])
                # if i ==0 and j==0:
                #     handl, lab = axes[j,i].get_legend_handles_labels()
                #     by_label = dict(zip(lab, handl))
                #     by_label = dict(sorted(by_label.items()))
                    # axes[j,i].legend(handles=by_label.values(),labels=by_label.keys())
    fig.savefig(folder+file_name, dpi=300)