import os
import csv
import ast
import pandas as pd
import numpy as np

def clear_participant_feedback(file):
    """
        This method preprocesses and cleans the feedback participants provided:
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

    # #########################################################################
    # feedback_group = feedback_df[feedback_df["Function"] == "select_reject_application"]
    # for part in feedback_group['ID'].unique():
    #     part_feedback = feedback_group[feedback_group["ID"]== part]
    #     last_feedback_weight = part_feedback.loc[part_feedback.index[-1]]['Value']
    return file
    
# def get_feedback(folderPath, fileEnding):
#     participants_responses_dict = {} ## <prolific_id>: Dict <question_id>: response
#     ##
#     for root,dirs,files in os.walk(folderPath):
#         for file in files:
#             if file.endswith(fileEnding):
#                 with open(folderPath+"\\"+file, "r",encoding='utf-8') as f:
#                     reader = csv.reader(f)
#                     next(reader) 
#                     feedback_dict = {}
#                     for row in reader:                       
#                         if row[5] == 'OKBUTTON_CLICKED_DECIDE_MODAL_Applications_List':
#                             weights = row[7]
#                             weights = ast.literal_eval(weights)
#                             feedback_dict[row[3]]={'label':row[6],'init_weights':weights['initial_weights'],'changed_weights':weights['changed_weights']}

#                 f.close()
#                 user_id = file.replace(fileEnding, "")
#                 #print(file.replace(fileEnding, "") )
#                 participants_responses_dict[user_id] = feedback_dict
#     return participants_responses_dict

def get_all_participant_feedback_clean(prolific_export_filePath, interaction_logs_filePath):
    prolific_export = pd.read_csv(prolific_export_filePath)
    
    ## GET data of only approved participants
    prolific_export = prolific_export[prolific_export['Status']=='APPROVED']
    
    ## GET prolific id of approved participants
    participants = prolific_export['Participant id'].tolist()
    
    ## Collect all participants' feedback together
    feedback_df = pd.DataFrame()    
    for p_id in participants:    
        file_name = p_id+".csv"    
        file = pd.read_csv(interaction_logs_filePath+file_name, delimiter=',')
        
        # ## Conversion of App ID to int     
        # file["App ID"] = file["App ID"].fillna(0.0).astype(int)
        
        # ## Reset_index 
        # file.index = np.arange(0, len(file))
        
        # ## Dropping Login
        # file = file[file.Function != "Login"]
        # file.reset_index(drop = True, inplace = True)
        
        # ## Dropping Repeated Entries of Application Selections
        # ## Due a glitch in code the selection of an application was captured multiple times
        # drop_index = []
        # for i in range(1,len(file.index)):
        #     try:
        #         if ((file.at[i,"App ID"] == int(file.at[i,"Value"])) and (file.loc[i,"ID"] in file.loc[0:i-1,"ID"].unique()) and (file.loc[i,"App ID"]  in file.loc[0:i-1,"App ID"].unique())):          
        #                 drop_index.append(i)
        #     except: 
        #         continue    
        # file.drop(index = drop_index,inplace = True)
        # file.reset_index(drop = True, inplace = True)

        ## clean and pre-process feedback
        file = clear_participant_feedback(file)
        
        ## add participant feedback to feedback df
        feedback_df = pd.concat([feedback_df,file],axis=0)
     
    # feedback_df.drop(columns = "Pattern",inplace = True)
    # feedback_df.reset_index(drop = True, inplace = True)

    # ## Dropping the entires where function chosen is apply_refine_search (i.e., filtering for attribute) 
    # ## but there is no attribute present that was filtered for.     
    # df_with_null_attribute = feedback_df[feedback_df["Attribute"].isna()]
    # null_attribute_index = df_with_null_attribute[df_with_null_attribute["Function"] == "apply_refine_search"].index
    
    # feedback_df.drop(index = null_attribute_index,inplace = True)
    
    feedback_df.reset_index(drop = True, inplace = True)
    return feedback_df