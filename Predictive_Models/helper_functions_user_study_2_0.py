import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import sys
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns


## Scikit related
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, chi2, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy import signal
from scipy import integrate
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import cross_val_score
from scipy.stats import skew
from scipy.stats import norm, kurtosis
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score




########################### File Parser script ########################################
'''
Developed on 4th July 2022 : Assumes a download_folders.py in the composite and ziptie folders. 
The corrupted files are manually deleted after consulting Vasu. 
This function then just returns the file_list of all the filenames in a root folder.
'''

def get_all_file_paths(source_data_path_composite, folder_list) :
    num_folders  = len(folder_list)
    file_names = []
    for i in range(0, num_folders) :
        # print(i)
        if (folder_list[i] != 'download_folders.py') :
            for file in os.listdir(os.path.join(source_data_path_composite, folder_list[i])):
                d = os.path.join(source_data_path_composite,folder_list[i], file)
                file_names.append(d)
                # print(d)
    return file_names            

def __read_csv_drop_unnamed__(csv_path) :
    pd_test = pd.read_csv(csv_path)
    pd_test.columns.str.match("Unnamed")
    pd_test = pd_test.loc[:,~pd_test.columns.str.match("Unnamed")]
    return pd_test
 

###################################REGRESSION SPECIFIC#####################################################################
"""These functions are curated for regression tasks mostly.
"""
########################################################################################################
""" Created on 26th Novemeber to avoid code duplication for Boeing Demo. We can re-use these for future as well."""
def conv_segment_df (segment) :
    df = pd.DataFrame(segment, columns=[
        ['Time',
 'HR_Processed',
 'HR',
 'HRV',
 'RR',
 'RRSQI',
 'ECGSQI',
 'ECG',
 'acc_X',
 'acc_Y',
 'acc_Z',
 'Temperature',
 'IMU_1_ax_g_',
 'IMU_1_ay_g_',
 'IMU_1_az_g_',
 'IMU_1_gx_dps_',
 'IMU_1_gy_dps_',
 'IMU_1_gz_dps_',
 'IMU_2_ax_g_',
 'IMU_2_ay_g_',
 'IMU_2_az_g_',
 'IMU_2_gx_dps_',
 'IMU_2_gy_dps_',
 'IMU_2_gz_dps_',
 'IMU_3_ax_g_',
 'IMU_3_ay_g_',
 'IMU_3_az_g_',
 'IMU_3_gx_dps_',
 'IMU_3_gy_dps_',
 'IMU_3_gz_dps_',
 'IMU_4_ax_g_',
 'IMU_4_ay_g_',
 'IMU_4_az_g_',
 'IMU_4_gx_dps_',
 'IMU_4_gy_dps_',
 'IMU_4_gz_dps_',
 'IMU_5_ax_g_',
 'IMU_5_ay_g_',
 'IMU_5_az_g_',
 'IMU_5_gx_dps_',
 'IMU_5_gy_dps_',
 'IMU_5_gz_dps_',
 'Physical Fatigue - Initial',
 'Physical Fatigue - Final',
 'Mental Fatigue - Initial',
 'Mental Fatigue - Final',
 'Performance rating',
 'Age',
 'Weight',
 'Height',
 'Weights added',
 'Gender']
    ])
    return df 

def file_segment_features(test_seg) :
#    print('Inside segment features') 
   # Give segmented 3D array to this --> Features dataframe
   test_one_seg = conv_segment_df(test_seg[0])
   features_df = regr_feature_extract_one_segment(test_one_seg)
   #features_df.head()
   for i in range(1, np.shape(test_seg)[0]) :
    # print('===============================================================I am in segment ', i)
    temp_seg = conv_segment_df(test_seg[i])
    df = regr_feature_extract_one_segment(temp_seg)
    features_df = features_df.append(df)
    
   return features_df 

############################ Use this read_csv function to drop the unnecessary unnamed columns if any ################
def __read_csv_drop_unnamed__(csv_path) :
    pd_test = pd.read_csv(csv_path)
    pd_test.columns.str.match("Unnamed")
    pd_test = pd_test.loc[:,~pd_test.columns.str.match("Unnamed")]
    return pd_test
############################ IMU feature extraction per segment ########################################

def regr_imu_feature_extraction(acc_x_np, acc_y_np, acc_z_np, gyro_x_np, gyro_y_np, gyro_z_np, window) :
    ## Do this to convert to array from a ragged list
    acc_x_np = np.reshape(acc_x_np, (window))
    acc_y_np = np.reshape(acc_y_np, (window))
    acc_z_np = np.reshape(acc_z_np, (window))
    
    gyro_x_np = np.reshape(gyro_x_np, (window))
    gyro_y_np = np.reshape(gyro_x_np, (window))
    gyro_z_np = np.reshape(gyro_x_np, (window))

    # Expects the length of each input argument as (1, length(segment))
    rms_acc_raw = (acc_x_np**2 + acc_y_np**2 + acc_z_np**2)/3
    # print('RMS_raw = ', rms_acc_raw)
    rms_acc = np.sqrt(np.sum(acc_x_np**2 + acc_y_np**2 + acc_z_np**2)/3)
    # plt.plot(acc_x_np**2)
    # plt.plot(acc_y_np**2)
    # plt.plot(acc_z_np**2)
    # plt.show()
    # print('RMS_ACC',rms_acc)
    # acc_corr = np.correlate(rms_acc_raw, rms_acc_raw)
    # returns a single value : Area under the curve using numerical methods
    # print('Acc is ',acc_x_np) 
    vel_x = np.trapz(acc_x_np)
    vel_y = integrate.trapz(acc_y_np)
    vel_z = integrate.trapz(acc_z_np)
    
    rms_vel =  np.sqrt(np.sum(vel_x**2 + vel_y**2 + vel_z**2)/3)
    

    rms_gyro = np.sqrt(np.sum(gyro_x_np**2 + gyro_y_np**2 + gyro_z_np**2)/3)
    # gyro_corr = autocorr(rms_gyro)
    # range of angular velocity
    range_gyro = (np.max(gyro_x_np**2 + gyro_y_np**2 + gyro_z_np**2)/3) - (np.min(gyro_x_np**2 + gyro_y_np**2 + gyro_z_np**2)/3)



    ## dimensionless jerk estimate LDLJ - https://www.mendeley.com/reference-manager/reader/f7512adf-29ad-381e-8031-bdae9c2c775e/3817899e-f4f4-58cb-556e-bd883f186399
    # Compute individual terms
    ## FIXME :: Review once
    dt = 500 # sampling rate
    # Hack :: Need array of (1, n_dim) for diff to work
    derivative_acc_x = (np.diff(acc_x_np.T)/dt)**2
    derivative_acc_y = (np.diff(acc_y_np.T)/dt)**2
    derivative_acc_z = (np.diff(acc_z_np.T)/dt)**2
    # print(np.shape(acc_x_np))
    # print('-----Printing Derivatives--------')
    derivative_sum = derivative_acc_x, derivative_acc_y, derivative_acc_z
    # print(derivative_acc_x)
    # print(derivative_acc_y)
    # print(derivative_acc_z)
    
    sum_ldlj_term = np.sum(derivative_sum)*dt
    # print('Sum terms =', sum_ldlj_term)
    # print(np.mean(rms_acc_raw))
    a_2_peak = ((np.max(rms_acc_raw)) - (np.mean(rms_acc_raw)))**2
    # print('a2 peak = ', a_2_peak)
    del_time = len(acc_x_np) * dt
    # print(len(acc_x_np))
    ldlj_jerk = -np.log((del_time/a_2_peak)*sum_ldlj_term)
    # print(ldlj_jerk)
    

    return rms_acc, rms_vel, rms_gyro, range_gyro, ldlj_jerk

def imu_kinetic_regr(acc_x, acc_y, acc_z, weight, add_weight, window) :
    ## Do this to convert to array from a ragged list
    acc_x = np.reshape(acc_x, (window))
    acc_y = np.reshape(acc_y, (window))
    acc_z = np.reshape(acc_z, (window))
    rms_acc = np.sqrt(np.sum(acc_x**2 + acc_y**2 + acc_z**2)/3)
    # returns a single value : Area under the curve using numerical methods 
    vel_x = integrate.trapz(acc_x)
    vel_y = integrate.trapz(acc_y)
    vel_z = integrate.trapz(acc_z)
    
    rms_vel =  np.sqrt(np.sum(vel_x**2 + vel_y**2 + vel_z**2)/3)
    kinect = (1/2)*np.mean(weight + add_weight)*rms_vel**2
    return kinect

############################ Vital Signs feature extraction per segment ########################################

def _vital_feature_extract (HR, HRV, temp, SQI, age) :
    # Expects the length of each input argument as (1, length(segment))
    # discard elements corrsponding to SQI<0.4
    HR_new = []
    HRV_new = []
    for i in range(1, len(HR)) :
        if (SQI[i] > 0.5) :
            HR_new.append(HR[i])
            HRV_new.append(HRV[i])
    ## Handle in final dataframe. Drop HR=0
    """ This is more evident when using smaller windows, esp in case of regression.
    """
    if (len(HR_new) == 0) :
        # print('Length of updated HR', np.shape(HR_new))
        # Extract HR features        
        avg_hr  = 0
        # Average the first 10% values and the last 10% values and then take a differnce.
        del_hr = 0
        median_hr = 0
        std_hr = 0
        skew_hr = 0
        kurt_hr = 0
        max_hr = 0
        cardiac_capacity = 0

        # Extract HRV features        
        avg_hrv  = 0
        # Average the first 10% values and the last 10% values and then take a differnce.
        del_hrv = 0
        median_hrv = 0
        std_hrv = 0
        skew_hrv = 0
        kurt_hrv = 0
        max_hrv = 0
    else :
        # Extract HR features        
        avg_hr  = np.mean(HR_new)
        # Average the first 10% values and the last 10% values and then take a differnce.
        del_hr = np.mean(HR_new[int(0.9*len(HR_new)):len(HR_new)]) - np.mean(HR_new[0:int(0.1*len(HR_new))])
        median_hr = np.median(HR_new)
        std_hr = np.std(HR_new)
        skew_hr = skew(HR_new)
        kurt_hr = kurtosis(HR_new)
        max_hr = np.max(HR_new)
        cardiac_capacity = max_hr/(220-np.mean(age))

        # Extract HRV features        
        avg_hrv  = np.mean(HRV_new)
        # Average the first 10% values and the last 10% values and then take a differnce.
        del_hrv = np.mean(HRV_new[int(0.9*len(HRV_new)):len(HRV_new)]) - np.mean(HRV_new[0:int(0.1*len(HRV_new))])
        median_hrv = np.median(HRV_new)
        std_hrv = np.std(HRV_new)
        skew_hrv = skew(HRV_new)
        kurt_hrv = kurtosis(HRV_new)
        max_hrv = np.max(HRV_new)

    # Extract temperature features
    avg_temp = np.mean(temp)
    max_temp = np.max(temp)
    del_temp = np.mean(temp[int(0.9*len(temp)):len(temp)]) - np.mean(temp[0:int(0.1*len(temp))])
    std_temp = np.std(temp)
    duration = (len(temp) * 500)/1000 # sampling rate is 500 ms in this case
    ## Add RR here
    return avg_hr, del_hr, median_hr, std_hr, skew_hr, kurt_hr, max_hr, cardiac_capacity, avg_hrv, del_hrv, median_hrv, std_hrv, skew_hrv, kurt_hrv, max_hrv, avg_temp, max_temp, del_temp, std_temp, duration


# ###################### Extract features per segment ########################################################
def regr_feature_extract_one_segment(test_one_seg) :
    window = len(test_one_seg) 
    
    rms_acc_imu1, rms_imu1_vel, rms_imu1_gyro, range_gyro_imu1, ldlj_jerk_1 = regr_imu_feature_extraction(test_one_seg[['IMU_1_ax_g_']].to_numpy(), test_one_seg[['IMU_1_ay_g_']].to_numpy(), test_one_seg[['IMU_1_az_g_']].to_numpy(), test_one_seg[['IMU_1_gx_dps_']].to_numpy(), test_one_seg[['IMU_1_gy_dps_']].to_numpy(), test_one_seg[['IMU_1_gz_dps_']].to_numpy(), window)
    rms_acc_imu2, rms_imu2_vel, rms_imu2_gyro, range_gyro_imu2, ldlj_jerk_2 = regr_imu_feature_extraction(test_one_seg[['IMU_2_ax_g_']].to_numpy(), test_one_seg[['IMU_2_ay_g_']].to_numpy(), test_one_seg[['IMU_2_az_g_']].to_numpy(), test_one_seg[['IMU_2_gx_dps_']].to_numpy(), test_one_seg[['IMU_2_gy_dps_']].to_numpy(), test_one_seg[['IMU_2_gz_dps_']].to_numpy(), window)
    rms_acc_imu3, rms_imu3_vel, rms_imu3_gyro, range_gyro_imu3, ldlj_jerk_3 = regr_imu_feature_extraction(test_one_seg[['IMU_3_ax_g_']].to_numpy(), test_one_seg[['IMU_3_ay_g_']].to_numpy(), test_one_seg[['IMU_3_az_g_']].to_numpy(), test_one_seg[['IMU_3_gx_dps_']].to_numpy(), test_one_seg[['IMU_3_gy_dps_']].to_numpy(), test_one_seg[['IMU_3_gz_dps_']].to_numpy(), window)
    rms_acc_imu4, rms_imu4_vel, rms_imu4_gyro, range_gyro_imu4, ldlj_jerk_4 = regr_imu_feature_extraction(test_one_seg[['IMU_4_ax_g_']].to_numpy(), test_one_seg[['IMU_4_ay_g_']].to_numpy(), test_one_seg[['IMU_4_az_g_']].to_numpy(), test_one_seg[['IMU_4_gx_dps_']].to_numpy(), test_one_seg[['IMU_4_gy_dps_']].to_numpy(), test_one_seg[['IMU_4_gz_dps_']].to_numpy(), window)
    rms_acc_imu5, rms_imu5_vel, rms_imu5_gyro, range_gyro_imu5, ldlj_jerk_5 = regr_imu_feature_extraction(test_one_seg[['IMU_5_ax_g_']].to_numpy(), test_one_seg[['IMU_5_ay_g_']].to_numpy(), test_one_seg[['IMU_5_az_g_']].to_numpy(), test_one_seg[['IMU_5_gx_dps_']].to_numpy(), test_one_seg[['IMU_5_gy_dps_']].to_numpy(), test_one_seg[['IMU_5_gz_dps_']].to_numpy(), window)
    # print('After imu features')
    kinetic_expense = imu_kinetic_regr(test_one_seg[['IMU_5_ax_g_']].to_numpy(), test_one_seg[['IMU_5_ay_g_']].to_numpy(), test_one_seg[['IMU_5_az_g_']].to_numpy(), test_one_seg[['Weight']].to_numpy(), test_one_seg[['Weights added']].to_numpy(), window) 
    # print('After kinetic expense')
    #    avg_hr, avg_hrv, avg_temp, avg_SQI, del_temp, del_hr,  median_hr, median_hrv, skew_hr, kurt_hr = _vital_feature_extract (test_one_seg['HR_Processed'], test_one_seg['HRV'], test_one_seg['Temperature'], test_one_seg['ECGSQI'])
    # FIXME :: Use HR_Processed
    avg_hr, del_hr, median_hr, std_hr, skew_hr, kurt_hr, max_hr, cardiac_capacity, avg_hrv, del_hrv, median_hrv, std_hrv, skew_hrv, kurt_hrv, max_hrv, avg_temp, max_temp, del_temp, std_temp, duration = _vital_feature_extract (test_one_seg[['HR']].to_numpy(), test_one_seg[['HRV']].to_numpy(), test_one_seg[['Temperature']].to_numpy(), test_one_seg[['ECGSQI']].to_numpy(), test_one_seg[['Age']].to_numpy())
    # print('After vital signs features')

    init_fatigue = np.mean(test_one_seg[['Physical Fatigue - Initial']].to_numpy() )
    fin_fatigue = np.mean(test_one_seg[['Physical Fatigue - Final']].to_numpy())
    weight = np.mean(test_one_seg[['Weight']].to_numpy())
    height = np.mean(test_one_seg[['Height']].to_numpy())
    age = np.mean(test_one_seg[['Age']].to_numpy())
    gender = np.mean(test_one_seg[['Gender']].to_numpy())
   
    # Make a dataframe
    feature_list = [fin_fatigue, init_fatigue, 
    avg_hr, del_hr, median_hr, std_hr, skew_hr, kurt_hr, max_hr, cardiac_capacity, avg_hrv, del_hrv, median_hrv, std_hrv, skew_hrv, kurt_hrv, max_hrv, avg_temp, max_temp, del_temp, std_temp, duration, # vital signs features (19)
    rms_acc_imu1, rms_imu1_vel, rms_imu1_gyro, range_gyro_imu1, ldlj_jerk_1,
    rms_acc_imu2, rms_imu2_vel, rms_imu2_gyro, range_gyro_imu2, ldlj_jerk_2,
    rms_acc_imu3, rms_imu3_vel, rms_imu3_gyro, range_gyro_imu3, ldlj_jerk_3,
    rms_acc_imu4, rms_imu4_vel, rms_imu4_gyro, range_gyro_imu4, ldlj_jerk_4,
    rms_acc_imu5, rms_imu5_vel, rms_imu5_gyro, kinetic_expense,
    age, height, weight, gender
    # borg_rating
    ]
    
    df = pd.DataFrame(feature_list).T
    df.columns=['final_fatigue','init_fatigue', 
    'avg_hr', 'del_hr', 'median_hr', 'std_hr', 'skew_hr', 'kurt_hr', 'max_hr', 'cardiac_capacity', 'avg_hrv', 'del_hrv', 'median_hrv', 'std_hrv', 'skew_hrv', 'kurt_hrv', 'max_hrv', 'avg_temp', 'max_temp', 'del_temp', 'std_temp', 'duration', # vital signs features (19)
    'rms_acc_imu1', 'rms_imu1_vel', 'rms_imu1_gyro', 'range_gyro_imu1', 'ldlj_jerk_1',
    'rms_acc_imu2', 'rms_imu2_vel', 'rms_imu2_gyro', 'range_gyro_imu2', 'ldlj_jerk_2',
    'rms_acc_imu3', 'rms_imu3_vel', 'rms_imu3_gyro', 'range_gyro_imu3', 'ldlj_jerk_3',
    'rms_acc_imu4', 'rms_imu4_vel', 'rms_imu4_gyro', 'range_gyro_imu4', 'ldlj_jerk_4',
    'rms_acc_imu5', 'rms_imu5_vel', 'rms_imu5_gyro', 'kinetic_expense' ,
    'age', 'height', 'weight', 'gender']
    return df 


###################### Segment the time and interpolate the y_label ########################################################    
# Trim data length as window * ceil(sample_length/window)
# Divide into sample_length/ window segment
def regr_segment_time_series(window, data) :
    y1 = np.mean(data['Physical Fatigue - Initial'])
    y2 = np.mean(data['Physical Fatigue - Final'] )
    x2 = len(data)
    
    interp_fatigue = np.linspace(y1, y2, num=len(data), endpoint=True)
    data['Physical Fatigue - Final'] = interp_fatigue
    # plt.plot(data['Physical Fatigue - Final'])
    # plt.xlabel('Time/120 (mins)')
    # plt.ylabel('Interpolated Fatigue Score')
    # plt.show()
    # Convert dataframes to numpy
    data = pd.DataFrame.to_numpy(data)
    # Expects data to be a numpy array of the shape [num_sample, num_features]
    # Feature scale here before converting to 3D
    # sc1 = MinMaxScaler()
    # data = sc1.fit_transform(data)
    sample_length = np.shape(data)[0]
    num_segments = int(sample_length/window)
    data_temp = data[0:(num_segments*window)]
    x_r = data_temp.reshape(int(np.shape(data_temp)[0]/window),window,np.shape(data_temp)[1])
    return x_r

###################### SVR based regression functions ########################################################

def SVR_poly (X_train, Y_train, X_test, Y_test) :
    regressor = SVR(kernel='poly')
    regressor.fit(X_train,Y_train)
    y_pred = regressor.predict(X_test)
    y_test = np.asarray(Y_test)
    print('SVM Poly regressor MSE is', mean_squared_error(y_pred, y_test))
    print('SVM Poly regressor MAE is', mean_absolute_error(y_pred, y_test))
    print('SVM Poly regressor R2-Score is', r2_score(y_pred, y_test))
    plt.figure(1)
    plt.plot(y_pred, 'o-')
    plt.plot(y_test)
    plt.title('SVM Poly regressor')
    plt.legend(['y_pred', 'y_actual'])

    plt.figure(2)
    plt.scatter(y_pred, y_test)
    y = np.linspace(start = 0, stop = 10, num = 50)
    plt.xlabel('Predicted Fatigue Level')
    plt.ylabel('Actual Fatigue Level')
    plt.plot(y,y, 'r')
    plt.show()
    
    
    print('=============================================================================================')
    return mean_squared_error(y_pred, y_test), mean_absolute_error(y_pred, y_test), r2_score(y_pred, y_test), y_pred, y_test

def SVR_rbf (X_train, Y_train, X_test, Y_test) :
    regressor = SVR(kernel='rbf')
    regressor.fit(X_train,Y_train)
    y_pred = regressor.predict(X_test)
    y_test = np.asarray(Y_test)
    
    print('SVM RBF regressor MSE is', mean_squared_error(y_pred, y_test))
    print('SVM RBF regressor MAE is', mean_absolute_error(y_pred, y_test))
    print('SVM RBF regressor R2-Score is', r2_score(y_pred, y_test))


    plt.figure(1)
    plt.plot(y_pred, 'o-')
    plt.plot(y_test)
    
    plt.legend(['y_pred', 'y_actual'])
    plt.show()

    plt.figure(2)
    plt.scatter(y_pred, y_test)
    y = np.linspace(start = 0, stop = 10, num = 50)
    plt.xlabel('Predicted Fatigue Level')
    plt.ylabel('Actual Fatigue Level')
    plt.plot(y,y, 'r')
    plt.show()

    print('=============================================================================================')
    return mean_squared_error(y_pred, y_test), mean_absolute_error(y_pred, y_test), r2_score(y_pred, y_test), y_pred, y_test


def SVR_linear (X_train, Y_train, X_test, Y_test) :
    regressor = SVR(kernel='linear')
    regressor.fit(X_train,Y_train)
    #5 Predicting a new result
    
    y_pred = regressor.predict(X_test)
    y_test = np.asarray(Y_test)
    
    print('SVM Linear regressor MSE is', mean_squared_error(y_pred, y_test))
    print('SVM Linear regressor MAE is', mean_absolute_error(y_pred, y_test))
    print('SVM Linear regressor R2-Score is', r2_score(y_pred, y_test))
    
    plt.figure(1)
    plt.plot(y_pred, 'o-')
    plt.plot(y_test)
    plt.title('SVM Linear regressor')
    plt.legend(['y_pred', 'y_actual'])
    plt.show()

    plt.figure(2)
    plt.scatter(y_pred, y_test)
    y = np.linspace(start = 0, stop = 10, num = 50)
    plt.xlabel('Predicted Fatigue Level')
    plt.ylabel('Actual Fatigue Level')
    plt.plot(y,y, 'r')
    plt.show()
    
    print('=============================================================================================')
    return mean_squared_error(y_pred, y_test), mean_absolute_error(y_pred, y_test), r2_score(y_pred, y_test), y_pred, y_test    
###################### Feature scaling function ########################################################
def feature_scaling(min_val, max_val, dataframe_name) :
    sc1 = MinMaxScaler((min_val, max_val))
    df_scaled = sc1.fit_transform(dataframe_name[
        list(dataframe_name)
    ])    
    

    df_scaled = pd.DataFrame(df_scaled, columns= list(dataframe_name))
    return df_scaled  


###################### Split the fatigue levels into 3 classes ########################################################

def __split_3_classes__ (df_scaled_composite) :
    for i in range(0, len(df_scaled_composite)) :
        if (df_scaled_composite['final_fatigue'][i] < 3.0) :
            df_scaled_composite['final_fatigue'][i] = 0.0
        if (df_scaled_composite['final_fatigue'][i] > 2.0) & (df_scaled_composite['final_fatigue'][i] < 7.0) :
            df_scaled_composite['final_fatigue'][i] = 1.0   
        if (df_scaled_composite['final_fatigue'][i] > 6.0) & (df_scaled_composite['final_fatigue'][i] < 11.0) :
            df_scaled_composite['final_fatigue'][i] = 2.0     
    return df_scaled_composite        



################## TSNE Visualisation ##########################################
def __tsne_visualise__(X_train, Y_train, X_test, Y_test) :
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(X_train) 
    df = pd.DataFrame()
    df["y"] = Y_train
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                # palette=sns.color_palette("hls", 10),
                data=df).set(title="Combined data T-SNE projection")


