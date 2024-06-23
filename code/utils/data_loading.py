import os
import csv
import ast
import pandas as pd
import numpy as np
import copy
from collections import Counter

def preprocess_participant_log(file):
    """
        This method preprocesses and cleans particiapants' logs:
        1) Transform string values representing numbers into numerical values
        2) We remove redundant information not used in the analysis, or blank columns never used to record data
        3) We fix the order some weight changes were recorded
        4) We remove multiple recordings of the same application-selection due to a glitch in code

        Parameter:
        --------------------
        file: Pandas DataFrame containing contents of participant feedback csv file
    """
    ## Conversion of App ID to int     
    file["App ID"] = file["App ID"].fillna(0.0).astype(int)
    
    ## Reset_index 
    file.index = np.arange(0, len(file))
    
    ## Drop record for Login - not used in analysis
    file = file[file.Function != "Login"]
    file.reset_index(drop = True, inplace = True)
    
    ## Drop repeated entries of application selections
    ## Due a glitch in code the same selection of an application was captured multiple times
    drop_index = []
    for i in range(1,len(file.index)):
        try:
            if ((file.at[i,"App ID"] == int(file.at[i,"Value"])) and (file.loc[i,"ID"] in file.loc[0:i-1,"ID"].unique()) and (file.loc[i,"App ID"]  in file.loc[0:i-1,"App ID"].unique())):          
                    drop_index.append(i)
        except: 
            continue    
    file.drop(index = drop_index,inplace = True)
    file.reset_index(drop = True, inplace = True)

    ## Drop blank Pattern column
    file.drop(columns = "Pattern",inplace = True)
    file.reset_index(drop = True, inplace = True)

    ## Drop the entries of "apply_refine_search" (i.e., filtering for attribute) 
    ## when there is no attribute present that was filtered for (i.e., when clearing a filtering)     
    df_with_null_attribute = file[file["Attribute"].isna()]
    null_attribute_index = df_with_null_attribute[df_with_null_attribute["Function"] == "apply_refine_search"].index
    file.drop(index = null_attribute_index,inplace = True)
    file.reset_index(drop = True, inplace = True)

    ## Convert weight feedback elements from string to list of dicts
    feedback_group = file[file["Function"] == "select_reject_application"].index.tolist()
    for idx in feedback_group:
        file.at[idx, 'Value'] = ast.literal_eval(file.loc[idx]['Value'])
    return file

def get_all_participants_logs(prolific_export_filePath, interaction_logs_filePath):
    prolific_export = pd.read_csv(prolific_export_filePath)
    
    ## GET data of only approved participants
    prolific_export = prolific_export[prolific_export['Status']=='APPROVED']
    
    ## GET prolific id of approved participants
    participants = prolific_export['Participant id'].tolist()
    
    ## Collect all participants' feedback together
    logs_df = pd.DataFrame()    
    for p_id in participants:    
        file_name = p_id+".csv"    
        file = pd.read_csv(interaction_logs_filePath+file_name, delimiter=',')       

        ## clean and pre-process feedback
        file = preprocess_participant_log(file)
        
        ## add participant feedback to feedback df
        logs_df = pd.concat([logs_df,file],axis=0)  
        
    logs_df.reset_index(drop = True, inplace = True)
    return logs_df

def update_clear_value(cl_value, el):
    """
        cl_value: List of elements {'attribute':<>,'value':<>}
        el: Dict element {'attribute':<>,'value':<>}
    """
    ## If attribute already in cl_value
    ## update value
    ## else add element to cl_value
    if len(cl_value):                    
        attr_found = False
        for el_prev_i,el_prev in enumerate(cl_value):
            if el_prev['attribute'] == el['attribute']:
                cl_value[el_prev_i]['value'] = el['value']
                attr_found = True
                break
        if not attr_found:
            cl_value.append(el)
    else:
        cl_value.append(el)
    return cl_value
## TEST ###
# pid = '6120211d8e1eab16fcb7ad69'## example of consecutive same app where same feature weight changes 
# pid = '6120211d8e1eab16fcb7ad69'
# p_id = '5e465ab4c07877130185c306'## case 3: 3 consecutive times same app
# pid = "5ebed23204aa470f9e1299bb"## case 1
# pid = '5f02fbee212873485f6ce5b8'## general debug
# pid = '5f256074297aac1d8ef7382d'## general debug
# pid = '60dc5726c777d9a0e550d1ca'## case 1 idx 497-498
# pid = '60e3303c9f5ea2fc726d58d8'## case 1 -6 -7 you need to update apps_ids_idx when new row is added with previous label feedback
def get_all_participants_feedback(logs_df):
    """
        logs_df: Pandas DataFrame produced by the method get_all_participants_logs
    """
    count_idx = -1
    df_all = pd.DataFrame()
    feedback_group = logs_df[logs_df["Function"] == "select_reject_application"]
    for part in feedback_group['ID'].unique().tolist():
        file = feedback_group[feedback_group["ID"]== part]    
        ## indices of recorded feedback instances
        part_indices = file.index.tolist()
        ## state after reordering weight feedback elements
        app_ids_cl = []
        app_ids_idx_cl = []
        ## state before current reordering action
        prev_idx = None
        prev_app_id = None
        prev_value = None
        ##
        for idx in part_indices:
            app_id = file.loc[idx]['App ID']
            app_ids_cl.append(app_id)
            app_ids_idx_cl.append(idx)
            value = file.loc[idx].Value 
            ## GET new weight feedback element recorded 
            ## since previous recorded feedback instance
            ## store them in new_wf_els
            if len(app_ids_cl)>1:
                new_wf_els = copy.deepcopy(value)
                for el in prev_value:
                    if el in new_wf_els:
                        new_wf_els.remove(el)            
            else:
                new_wf_els = copy.deepcopy(value)
            ## CASE 1: new_wf_els is not empty
            ## Disrtibute each new weight feedback element 
            ## to the correct feedback instance
            cl_cur_value = []
            cl_prev_value = []
            cl_missed_rec_values = {}
            cl_missed_rec_attributes = {}
            for el in new_wf_els:
                underscore_idx = el['attribute'].index('_')        
                fd_app_id =  int(el['attribute'][0:underscore_idx])
                el['attribute'] = el['attribute'][underscore_idx+1:]
                ## CASE 1.1: include it to current feedback instance (idx)
                if fd_app_id == app_id:     
                    ## If there is previous label feedback for current app_id 
                    ## copy weights from there - 
                    ## participants saw last set weights for each app
                    if len(cl_cur_value)==0 and app_ids_cl.count(fd_app_id) > 1: 
                        matched_idx = app_ids_idx_cl[len(app_ids_cl) - 1 - (app_ids_cl[0:len(app_ids_cl)-1][::-1].index(fd_app_id)+1)]
                        cl_cur_value = copy.deepcopy(file.loc[matched_idx].Value)                    
                    cl_cur_value = update_clear_value(cl_cur_value, el)                
                ## CASE 1.2: include it to previous feedback instance (prev_idx)
                ## cases 1.2 and 1.3 fix recording order of weight feedback elements
                elif fd_app_id == prev_app_id:# append to previous feedback instance
                    ## Get previous clean value
                    if len(cl_prev_value)==0:                    
                        cl_prev_value = copy.deepcopy(file.loc[prev_idx].Value)
                    cl_prev_value = update_clear_value(cl_prev_value, el)                
                ## CASE 1.3: include it to previous feedback instance (prev_idx)
                else:
                    if fd_app_id not in cl_missed_rec_values:
                        cl_missed_rec_values[fd_app_id] = []
                        cl_missed_rec_attributes[fd_app_id] = None
                    ## If there is previous label feedback for current app_id 
                    ## copy weights from there - 
                    ## participants saw last set weights for each app
                    if len(cl_missed_rec_values[fd_app_id])==0 and fd_app_id in app_ids_cl: 
                        matched_idx = app_ids_idx_cl[len(app_ids_cl) - 1 - (app_ids_cl[0:len(app_ids_cl)-1][::-1].index(fd_app_id)+1)]
                        cl_missed_rec_values[fd_app_id] = copy.deepcopy(file.loc[matched_idx].Value)
                        cl_missed_rec_attributes[fd_app_id] = file.loc[matched_idx].Attribute
                    cl_missed_rec_values[fd_app_id] = update_clear_value(cl_missed_rec_values[fd_app_id], el)
            ## handle missed feedback records
            if len(cl_missed_rec_values):
                for missed_app_id in cl_missed_rec_values:
                    new_row_dict = {"serial number": None, 
                                    "timestamp": None,
                                    "ID":part,
                                    "App ID":missed_app_id,
                                    "Function":"select_reject_application",
                                    "Attribute":cl_missed_rec_attributes[missed_app_id],
                                    "Value":None,
                                    "Lower bound":0.0,"Upper bound":0.0}
                    new_row = pd.DataFrame(new_row_dict,index=[count_idx])
                    new_row.at[count_idx,'Value'] = cl_missed_rec_values[missed_app_id]
                    split_idx = file.index.tolist().index(idx)
                    file = pd.concat([file.iloc[:split_idx], new_row, file.iloc[split_idx:]])
                    app_ids_cl.insert(len(app_ids_cl)-1, missed_app_id)
                    app_ids_idx_cl.insert(len(app_ids_idx_cl)-1, count_idx)
                    count_idx = count_idx - 1  
            ## Update data of previous feedback record
            if len(cl_prev_value):
                file.at[prev_idx,'Value'] = cl_prev_value
            ## Skip last feedback instance
            ## because we are not sure if there are missed feedback weight elements
            if idx==part_indices[-1]:
                file.drop(index = idx,inplace = True)
            else:
                ## CASE 2: new_wf_els was empty
                ## check if app_id appeared previously
                ## and copy weights feedback
                if len(cl_cur_value)==0 and app_ids_cl.count(app_id) > 1: 
                    matched_idx = app_ids_idx_cl[len(app_ids_cl) - 1 - (app_ids_cl[0:len(app_ids_cl)-1][::-1].index(app_id)+1)]
                    cl_cur_value = copy.deepcopy(file.loc[matched_idx].Value)            
                file.at[idx,'Value'] = cl_cur_value                  
            ## Update state variables
            prev_idx = idx
            prev_app_id = app_id
            prev_value = copy.deepcopy(value)
        df_all = pd.concat([df_all,file])
    return df_all