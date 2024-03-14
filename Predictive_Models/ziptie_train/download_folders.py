import gdown
import os
import pandas as pd


def download_folder_gdrive(folder_url, user_id, gender) :
    ## Base paths
    source_data_path_composite = '/home/payal/MxD_Data/MxD_Data_User_Study/ziptie'
    
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
    source_data_path_composite = '/home/payal/MxD_Data/MxD_Data_User_Study/ziptie'
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
                if ((folder_name == 'P007Z007S004') or (folder_name == 'P015Z015S003') or (folder_name == 'P017Z017S001') or (folder_name == 'P025Z025S005')) :
                    pd_test['Gender'] = 1
                    print('Female at ', folder_path)
                else :
                    pd_test['Gender'] = 0    
            
            pd_test.to_csv(path, index=False)

            
def main() :
    ## Download for every user

    # P004Z004S001
    # download_folder_gdrive('https://drive.google.com/drive/u/1/folders/1CrWpqjiBPKMKzKqZ5j32RBV89JIhFCpj','P004Z004S001', 0)

    #P027Z027S001
    # download_folder_gdrive('https://drive.google.com/drive/u/1/folders/1lF8elIa5Ix5gt6qj0MINwryFTLoJKgq3', 'P027Z027S001', 0)
    
    # #P028Z028S005
    # download_folder_gdrive('https://drive.google.com/drive/u/1/folders/1bPSE7LZVgQpCFqQziDPn02DdifW7ysjk','P028Z028S005',1)

    # #P029Z029S004
    # download_folder_gdrive('https://drive.google.com/drive/u/1/folders/1UclEchHTgD0hzX5PNVNMl0vcrHNWPSdo','P029Z029S004', 0)

    # #P030Z030S008
    # download_folder_gdrive('https://drive.google.com/drive/u/1/folders/1R5UL6YoeRGVu5rb6pQoxsbFFY1rr_Vl4','P030Z030S008', 0)

    # #P032Z032S013
    # download_folder_gdrive('https://drive.google.com/drive/u/1/folders/1UB5pMzPiA1lOkCW-lVQT4wmdm8dVBY2s', 'P032Z032S013', 1)

    # #P033Z033S016
    # download_folder_gdrive('https://drive.google.com/drive/u/1/folders/1UKo8JV8YbF4RBD5T4kDMbEnD2hIDx4iZ','P033Z033S016', 0)

    # # P036Z036S007
    # download_folder_gdrive('https://drive.google.com/drive/u/1/folders/1GwEDKiu77wjiT30W7Zzqmo5PYtdCImNZ', 'P036Z036S007', 0)

    # # P037Z037S005
    # download_folder_gdrive('https://drive.google.com/drive/u/1/folders/1H8nsfRGPVnfwR8MSZ7yHmz69Ab6pGXTz','P037Z037S005', 0)

    # # P038Z038S009
    # download_folder_gdrive('https://drive.google.com/drive/u/1/folders/1dsK2o8oGiFCHhhrRxv2jEcqcAyfekx1k', 'P038Z038S009', 0)

    # # P039Z039S012
    # download_folder_gdrive('https://drive.google.com/drive/u/1/folders/1BumyfmXoy1kxDjupfiqGgNSIYx50v5cM', 'P039Z039S012', 0)

    # # P043Z043S032
    # download_folder_gdrive('https://drive.google.com/drive/u/1/folders/1Img3xBAlKfLoZNnxEYAoxPZwkM2zw3ZF', 'P043Z043S032', 1)

    # # P042Z042S022
    # download_folder_gdrive('https://drive.google.com/drive/u/1/folders/1bfwzX9qshgW9_NEKIc4Avbr1--9i6K-a', 'P042Z042S022', 0)

    # # P041Z041S017
    # download_folder_gdrive('https://drive.google.com/drive/u/1/folders/1YgbXPxM9apaiMbykCrrM6xPJmZ_ajXKP', 'P041Z041S017', 1)

    # P040Z040S015
    download_folder_gdrive('https://drive.google.com/drive/u/1/folders/1S8KydmLB7NNq84GaUbYF2G_hSNewDkU3', 'P040Z040S015', 1)



    


if __name__ == '__main__':
    main()
