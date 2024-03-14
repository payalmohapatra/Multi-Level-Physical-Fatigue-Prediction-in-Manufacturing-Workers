"""
Author : Payal Mohapatra
Contact : PayalMohapatra2026@u.northwestern.edu
Project : MxD - Fatigue prediction for Operator 4.0 (2021-2022)
"""
import gdown
import os
import pandas as pd

def download_folder_gdrive(folder_url, user_id, gender) :
    ## Base paths
    source_data_path_composite = 'MxD_Data_User_Study/composite' ## TODO :: Update the path where you want to download data
    
    ## Add a check --> if folder exists dont download
    gdown.download_folder(folder_url)
    ## Replace folder name 'Sensor Segmented' with user id
    ## FIXME :: If its Folder Name is not Sensor Segmented Flag
    os.rename('Sensor Segmented', user_id)

    ## Check every file in the folder to add the gender 
    for file in os.listdir(user_id):
       path = os.path.join(source_data_path_composite, user_id, file)
       pd_test = pd.read_csv(path)
       if 'Gender' in pd_test.columns :
        print('Skipping for gender at : ', path)
       else :
        pd_test['Gender'] = gender
        pd_test.to_csv(path)

def add_gender_existing_folders(list_existing_folders) :
    source_data_path_composite = 'MxD_Data_User_Study/composite' ## TODO :: Update the path where you want to download data
    for folder_name in list_existing_folders :
        print(folder_name)
        folder_path = os.path.join(source_data_path_composite, folder_name)
        for file in os.listdir(folder_path):
            path = os.path.join(folder_path, file)
            # print(path)
            pd_test = pd.read_csv(path)
            if 'Gender' in pd_test.columns :
                print('Skipping')
            else :
                # Hardcoding the gender values
                if ((folder_name == 'P007C007S005') or (folder_name == 'P010C010S001') or (folder_name == 'P017C017S002')) :
                    pd_test['Gender'] = 1
                    print('Female at ', folder_path)
                else :
                    pd_test['Gender'] = 0    
            
            pd_test.to_csv(path)

            
def main() :
    ## Download for every user

    # P032C032S014
    download_folder_gdrive('https://drive.google.com/drive/u/1/folders/13r9lL1vGgNhDeX_gjScuLQUrbnwiKOYt', 'P032C032S014', 1)

    # P030C030S007
    download_folder_gdrive('https://drive.google.com/drive/u/1/folders/1gwpCHBfwMKp5ixMT3WGHhV81NpfaBPJg', 'P030C030S007', 0)

    # P029C029S003
    download_folder_gdrive('https://drive.google.com/drive/u/1/folders/1XmO61Dtoai7wriC9MpSz5CnWGpY2uE1t', 'P029C029S003', 0)

    # P028C028S006
    download_folder_gdrive('https://drive.google.com/drive/u/1/folders/15eXHCQmrEHEGE1-8vTYuAQD-mvsKaLlA', 'P028C028S006', 1)

    # P026C026S007
    download_folder_gdrive('https://drive.google.com/drive/u/1/folders/1WYQfXaF43Fi0gKMEZ4yliZEoInF36V2J', 'P026C026S007', 0)

    # P025C025S006
    download_folder_gdrive('https://drive.google.com/drive/u/1/folders/1NZWJh3AZl42dLFt-0T7LOHmOYq_jpq1k', 'P025C025S006', 1)

    # P024C024S003
    download_folder_gdrive('https://drive.google.com/drive/u/1/folders/1-4rfRFjh5QiTQykBUdbJKK1QlvGcqzXb', 'P024C024S003', 0)

    # P023C023S002
    download_folder_gdrive('https://drive.google.com/drive/u/1/folders/1bG0W80sIW13FFSFPsAIv7e01kml1wfgz', 'P023C023S002', 0)

    # P022C022S007
    download_folder_gdrive('https://drive.google.com/drive/u/1/folders/1_CZARMrzjktzAQLvGhnnoa1BblL9wq1M', 'P022C022S007', 0)

    # P021C021S006
    download_folder_gdrive('https://drive.google.com/drive/u/1/folders/1UStMMrr-2c0fkJViqUqz4BZI9LI2dh-f', 'P021C021S006', 0)

    # P031C031S030
    download_folder_gdrive('https://drive.google.com/drive/u/1/folders/1M9XnmATnVLGVbXRjEqsik1rkG24rT9hw', 'P031C031S030', 0)
    
    # P034C034S040
    download_folder_gdrive('https://drive.google.com/drive/u/1/folders/12CUbOiXSGrlCTxmuf-jHPqfg2B73IMiL', 'P034C034S040', 0)

    # P036C036S008
    download_folder_gdrive('https://drive.google.com/drive/u/1/folders/1sj8khXi90DEAB6AKR2GKekhKIEQliw6d', 'P036C036S008', 0)
    
    # P037C037S004
    download_folder_gdrive('https://drive.google.com/drive/u/1/folders/1hsNuY1zb3K1G21mITNbfLj9Vh0S9_AUz', 'P037C037S004', 0)
    
    # P038C038S010
    download_folder_gdrive('https://drive.google.com/drive/u/1/folders/1YWu9uEQ7kXyxj_pHJLxyrOzoe3bVLNRJ', 'P038C038S010', 0)
    
    # P039C039S011
    download_folder_gdrive('https://drive.google.com/drive/u/1/folders/1PId1vqlkAPMrLlMLNEwhlYEoR41_2zNz', 'P039C039S011', 0)
    
    # P043C043S031
    download_folder_gdrive('https://drive.google.com/drive/u/1/folders/1ixru3mu7JgVqbB0a6PJzZ1WAXXpbkUON', 'P043C043S031', 1)
    
    # P042C042S021
    download_folder_gdrive('https://drive.google.com/drive/u/1/folders/1eLQogy3A6C8ABaK8lUgcVug4wEX7Ipoy', 'P042C042S021', 0)
    
    # P041C041S016
    download_folder_gdrive('https://drive.google.com/drive/u/1/folders/1Ky04jOlXZsLldBJgUf8cpq1LOOD-FMVy', 'P041C041S016', 1)
    
    # P040C040S014
    download_folder_gdrive('https://drive.google.com/drive/u/1/folders/1OIb9HrZvhkcX1nABhibYRatk9qdGh5D9', 'P040C040S014', 1)

if __name__ == '__main__':
    main()
