import pandas as pd
import numpy as np
import mlflow
from pycaret.regression import *
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import PolynomialFeatures
import shutil 
import dask.dataframe as dd



outputs=['RESULT_InclinationBeltDirection__deg_',
       'RESULT_Inclination90ToBeltDirection__deg_']
impo_input=[
       'RESULT_InclinationBeltDirection__deg_',
       'RESULT_Inclination90ToBeltDirection__deg_',
       'PARAM_InclinationBeltDirectionOffset__deg_',
       'PARAM_Inclination90ToBeltDirectionOffset__deg_',
       'AP3_2_Data_of_Actual_Position_Y_deg',
       'AP3_2_Data_of_Actual_Position_X_deg',
       'AP3_2_Data_of_Actual_Position_Z_mm',
       'AP3_2_Settings_M358_Montage_Position_mm',
       'AP3_2_Settings_M359_Montage_Position_mm',
       'AP3_2_Actual_Part_to_Service','Palette']


def smooth(df, col):
    y_old=df[col]
    x_old=np.linspace(0,1,len(y_old))
    clf = LocalOutlierFactor(n_neighbors=50)
    df_pred_tmp=pd.DataFrame(data=df[col])
    #print(df_pred_test)
    y_pred = clf.fit_predict(df_pred_tmp)
    print(y_pred)
    df = df[y_pred == 1]    
    window_size = 11
    order = 3
    df[col] = savgol_filter(df[col], window_size, order)
    y_new=df[col]
    x_new=np.linspace(0,1,len(y_new))
    return(df)

def step_back(df):
    df=df.reset_index()
    print(df)
    df2=df
    df=df.drop(0)
    df=df.reset_index()
    lst=[]
    for col in df2.columns:
        col_new=str(col)+'_sb'
        lst.append(col_new)
    df2.columns=lst 
    df2=df2[df2.columns[1:len(df2.columns)]]
    cols_to_drop=[
       'AP3_2_Actual_Part_to_Service_sb','Palette_sb']
    df2=df2.drop(columns=cols_to_drop)
    df=df[impo_input]
    df_out=pd.concat([df2,df], axis=1)
    df_out=df_out.drop(len(df_out)-1)
    return(df_out)
    
def devide_df(df,n):
    size=int(round(len(df)/n,0))
    dfs=[]
    indx=0
    for i in range(n):
        df_small=df[indx:indx+round(size,0)]
        dfs.append(df_small)
        indx=indx+size+1
    return(dfs)

def df_col_clean(df):
    df=df.loc[df.RESULT_InclinationBeltDirection__deg_>0.1]
    df=df.loc[df.RESULT_InclinationBeltDirection__deg_<0.9]
    df=df.loc[df.RESULT_Inclination90ToBeltDirection__deg_<0]
    df=df.loc[df.RESULT_Inclination90ToBeltDirection__deg_>-0.7]
    df=df.replace('',np.nan)
    df=df.replace('#VALUE!',np.nan)
    df=df.dropna()
    df=df.reset_index(drop=True)
    df = df.drop(columns=df.select_dtypes(include=['object']).columns)
    return(df)


def df_drop_invar(df):
    variance = df.var()
    col_to_drop=variance.loc[variance<0.00001].index.tolist()
    df=df.drop(columns=col_to_drop)
    return(df)


def polyfunc(df):
    df_out=df[outputs]
    df=df.drop(columns=outputs)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    df_poly=poly.fit_transform(df)
    feature_names = poly.get_feature_names_out(df.columns)
    df=pd.DataFrame(df_poly, columns=feature_names)
    df=pd.concat([df,df_out], axis=1)
    return(df)

def corel_drop(df):
    df_out=df[outputs]
    df=df.drop(columns=outputs)
    corel=[]
    for out in outputs:
        corelation=df.corrwith(df_out[out])
        corel.append(corelation)
    df_corr=pd.DataFrame(corel)
    df_corr=df_corr.transpose()
    df_corr_drop = df_corr[(df_corr.abs() < 0.3).all(axis=1)]
    df_c_d_t=df_corr_drop.transpose()
    df_tmp=df
    df_filtered = df_corr_drop[~df_corr_drop.index.isin(impo_input)]
    comp=df_corr_drop.index.difference(df_filtered.index)
    df=df.drop(columns=df_filtered.index.tolist())
    #c=pd.DataFrame(df.columns)
    df=pd.concat([df,df_out], axis=1)
    return(df)
        
        
    


#       client = mlflow.MlflowClient(tracking_uri="http://127.0.0.1:5000")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
#os.remove('C:\\SVN\\Python projects\\Regression v1.0\\files\\')
main_path='C:/SVN/Python projects/Regression v1.0'
if os.path.exists(main_path):
    main_path=main_path
else:
    main_path='C:/Python projects/Regression v1.0'
if not os.path.exists(main_path):
          os.makedirs(main_path)
 
workfiles_path=main_path+'/files/'
import_path=main_path+'/files_csv/'
if not os.path.exists(workfiles_path):
          os.makedirs(workfiles_path) 
if not os.path.exists(import_path):
          shutil.copytree('C:/SVN/Python projects/files_csv',import_path) 



learn_db_name='100k_1.csv' #df learn 4000 #100k_1
test_db_name='test5.csv'
df=pd.read_csv(import_path+learn_db_name)
df_test_import=pd.read_csv(import_path+test_db_name)

df=df_col_clean(df)
df=df_drop_invar(df)
df_test_import=df_col_clean(df_test_import)
df_test_import=df_test_import[df.columns]

columns=df.columns

df=step_back(df)
df_test_import=step_back(df_test_import)

df=polyfunc(df)
df_test_import=polyfunc(df_test_import)
df=corel_drop(df)
df_test_import=df_test_import[df.columns]
df_train=df
df_test=df_test_import[df.columns]


"""

"""
df_train["Palette"] = df_train["Palette"].astype(int)   
palette=df_train.pop('Palette')
df_train_cat=pd.DataFrame()

df_train_cat['01']=(palette==1)*1.0
df_train_cat['02']=(palette==2)*1.0
df_train_cat['03']=(palette==3)*1.0
df_train_cat['04']=(palette==4)*1.0
df_train_cat['05']=(palette==5)*1.0
df_train_cat['06']=(palette==6)*1.0
df_train_cat['07']=(palette==7)*1.0
df_train_cat['08']=(palette==8)*1.0
#
#print(df_train['01'])
df_test_cat=pd.DataFrame()
df_test["Palette"] = df_test["Palette"].astype(int)  
palette_1=df_test.pop('Palette')
df_test_cat['01']=(palette_1==1.0)*1
df_test_cat['02']=(palette_1==2.0)*1
df_test_cat['03']=(palette_1==3.0)*1
df_test_cat['04']=(palette_1==4.0)*1
df_test_cat['05']=(palette_1==5.0)*1
df_test_cat['06']=(palette_1==6.0)*1
df_test_cat['07']=(palette_1==7.0)*1
df_test_cat['08']=(palette_1==8.0)*1

df_train=df_train.reset_index(drop=True)
df_train_cat=df_train_cat.reset_index(drop=True)
df_test=df_test.reset_index(drop=True)
df_test_cat=df_test_cat.reset_index(drop=True)

df_y_train=df_train[outputs]
df_y_test=df_test[outputs]

df_train_temp=df_train.drop(columns=outputs)
df_test_temp=df_test.drop(columns=outputs)
print(df_train_temp.columns)
print(df_test.columns)

df_train=pd.concat([df_train,df_train_cat],axis=1)    
df_test=pd.concat([df_test,df_test_cat],axis=1)

df_train=pd.concat([df_train,df_y_train],axis=1)    
df_test=pd.concat([df_test,df_y_test],axis=1)

df_train=df_train.drop(0)
df_test=df_test.drop(0)
df_train=df_train.drop(len(df_train))
df_test=df_test.drop(len(df_test))
from sklearn.linear_model import Lasso
inputs=df_train.columns
inputs=inputs.drop(outputs)
lasso = Lasso(alpha=0.01)
print("lasso starts")
temp_tf=np.empty((len(outputs),len(inputs)))
n=0
for col in outputs:
    lasso.fit(df_train[inputs], df_train[col])
    print(lasso.coef_)
    temp_tf[n,:]= lasso.coef_ != 0
    n+=1

cond= (temp_tf[0, :] == True) | (temp_tf[1, :] == True)
print(cond)

filtered_array = inputs[cond]
filtered_array=np.concatenate((filtered_array,outputs),axis=0)
filtered_array=np.concatenate((filtered_array,np.array(['01','02','03','04','05','06','07','08'])),axis=0)

df_train=df_train[filtered_array]
df_test=df_test[filtered_array]


arr=['AP3_2_Data_of_Actual_Position_Y_deg',
     'AP3_2_Data_of_Actual_Position_X_deg',
     'AP3_2_Data_of_Actual_Position_Z_mm',
     'AP3_2_Settings_M358_Montage_Position_mm',
     'AP3_2_Settings_M359_Montage_Position_mm']

df_train_comp=pd.read_csv(workfiles_path + 'df_train.csv')
df_test_comp=pd.read_csv(workfiles_path + 'df_test.csv')
if df_train_comp.equals(df_train):
    print("Train equal")
else:
    df_train.to_csv(workfiles_path + 'df_train.csv')
if df_test_comp.equals(df_test):
    print("Train equal")
else:
    df_test.to_csv(workfiles_path + 'df_test.csv')



print(df_train.columns)
if np.array_equal(df_train.columns,df_test.columns):
    print('Co;umns equal')
else:
    print('columns NOT equal')  
    exit()


df_train_columns=pd.DataFrame(df_train.columns)

df_train = df_train.T.drop_duplicates().T 
df_test=df_test[df_train.columns]

df_pred=pd.DataFrame()
result_path='C:\\Users\\zaizzhaim\\OneDrive - Mubea\\Documents\\02_101 Miling Machine Learning\\Results\\Experiments'
path='C:\\SVN\\'

normalize=True
remove_multicollinearity=True
pca=True
feature_selection=True
log=[
        ['normalize: ',normalize],
        ['remove_multicollinearity: ',remove_multicollinearity],
        ['pca: ',pca],
        ['feature_selection: ',feature_selection],
        ['train_df = ','C:/Users/zaizzhaim/OneDrive - Mubea/Documents/02_101 Miling Machine Learning/Data csv raw/'+learn_db_name],
        ['test_df = ','C:/Users/zaizzhaim/OneDrive - Mubea/Documents/02_101 Miling Machine Learning/Data csv raw/'+test_db_name]
    ]
n_exp=len(os.listdir(result_path))+1
result_path=result_path+'\\'+str(n_exp)
os.makedirs(result_path)
print(log)
df_log=pd.DataFrame(log)
#df_train.to_csv('df_train.csv')
    
with open(result_path+'\\log.txt', 'w') as file:
    for row in log:
        file.write(row[0].ljust(10) + str(row[1]).center(20) + "\n")
    
for output in outputs:

    
    
    
    s=setup(
       data=df_train,
       target=outputs[0], 
       #group_features=['RESULT_ZWheelDownAverageSensor3__mm_','RESULT_ZWheelDownAverageSensor4__mm_','RESULT_ZWheelDownAverageDiff__mm_','RESULT_ZWheelDownAngle__deg_','RESULT_ZAngle__deg_', 'RESULT_ZWheelUpAverageSensor3__mm_','RESULT_ZWheelUpAverageSensor4__mm_','RESULT_XAngleMax__deg_','RESULT_XWheelDownAverageDiff__mm_','RESULT_XWheelUpAverageSensor1__mm_','RESULT_XWheelUpAverageSensor2__mm_','RESULT_XWheelUpAverageDiff__mm_','RESULT_XWheelUpAngle__deg_'],
       test_data=df_test, 
       log_experiment = False, 
       experiment_name = outputs[0],
       index=False,
       use_gpu=False,
       log_data=True,
       normalize=True,
       preprocess=False,
       
       #remove_multicollinearity = remove_multicollinearity,
       categorical_features=['01','02','03','04','05','06','07','08'])  
       #pca=pca,
       #fold_strategy='timeseries',
       #data_split_shuffle=False,
       #fold_shuffle=False)
       #feature_selection=True)
       #feature_selection_threshold=0.8,
       #polynomial_features=True,
       #polynomial_degree=2,
       #remove_multicollinearity=True)
       #multicollinearity_threshold=0.9,
       #transformation=True)
       #transformation_method='yeo-johnson')
       #feature_selection=feature_selection)
    #best=compare_models()
    
    #predict_model(best)
    #X_train_proc = get_config('X_train')
    #y_train_porc = get_config('y_train')
    #X_test_proc = get_config('X_test')
    #y_test_proc = get_config('y_test')
    #dfs=[X_train_proc,y_train_porc,X_test_proc,y_test_proc]
    #for df_ in dfs:
    #    print(str(df_))e
    #print(df_test.columns)
    #model=load_model(path+output)
    #lst=predict_model(best, data=df_test)
    
    #df_pred=pd.concat([df_pred,lst[lst.columns[len(lst.columns)-2:len(lst.columns)]]],axis=1)
    #
    
    f_models=compare_models(exclude=['lr','lar','omp','ard','par','ransac','huber','kr','mlp','llar', 'tr','dummy','lasso','ridge','en','br','tr','svm'])
    lst_of_names=[]
    print(type(f_models))
    df_f_pred=pd.DataFrame()
    for model in f_models:
        name=str(model)
        name=name[0:name.find('(')]
        lst_f=predict_model(model, data=df_test)
        df_tmp=lst_f[lst_f.columns[len(lst_f.columns)-2:len(lst_f.columns)]]
        df_tmp.columns=[df_tmp.columns[0],name]
        print(df_tmp.columns)
        df_f_pred=pd.concat([df_tmp,df_f_pred],axis=1)
        #df_f_pred.to_csv(result_path+'\\'+output+'.csv')
        #plot_model(model, plot='feature')
        plt.scatter(df_tmp[df_tmp.columns[0]],df_tmp[df_tmp.columns[1]])
        plt.scatter([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1],[-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
        plt.axis('equal')
        plt.show()
        

        

    #print(df_pred)
    #plot_model(best,plot='feature')
df_pred.to_csv('df_pred.csv')
    
    
"""
#df_subs=df.diff()
##df_subs=df_subs.dropna()
#d#f_subs=df_subs.reset_index(drop=True)


#df=df.sample(frac=1)
#df_test=df_test.sample(frac=1)
df_train=df[df.columns[-6:-1]]
#print(df_train.columns)
df_train=pd.concat([df_train,df[df.columns[-5]]],axis=1)
df_train=pd.concat([df_train,df[df.columns[-12:-7]]],axis=1)
df_train=pd.concat([df_train,df[df.columns[-16:-14]]],axis=1)
df_train=pd.concat([df_train,df[df.columns[-17]]],axis=1)
df_train=pd.concat([df_train,df[df.columns[-20]]],axis=1)
df_train=pd.concat([df_train,df[df.columns[-26:-22]]],axis=1)
df_train=pd.concat([df_train,df[df.columns[-69]]],axis=1)
df_train=pd.concat([df_train,df[df.columns[-67]]],axis=1)
df_train=pd.concat([df_train,df[df.columns[37:40]]],axis=1)
df_train=pd.concat([df_train,df[df.columns[66:68]]],axis=1)
df_train=pd.concat([df_train,df[df.columns[68]]],axis=1)
df_train=pd.concat([df_train,df[df.columns[2]]],axis=1)
#df_train=pd.concat([df_train,df[df.columns[52:68]]],axis=1)
##df_train=pd.concat([df_train,df[df.columns[127:138]]],axis=1)
#df_train=pd.concat([df_train,df[df.columns[181:192]]],axis=1)
#df_train=pd.concat([df_train,df[df.columns[202:269]]],axis=1)




#df_test=pd.concat([df_test,df_test[df.columns[2]]],axis=1)
#df_test=pd.concat([df_test,df_test[df.columns[68]]],axis=1)



df_poly_t=poly.fit_transform(df_test)
feature_names = poly.get_feature_names_out(df_test.columns)
df_trans_t=pd.DataFrame(df_poly_t, columns=feature_names)
df_trans_t=df_trans_t.drop(columns=col_to_drop)
df_test=df_test_import[df_train.columns]
df_test=df_trans_t

#df_train=pd.concat([df_train,df[df.columns[2]]],axis=1)
#df_train=pd.concat([df_train,df[df.columns[3]]],axis=1)
#df_train=pd.concat([df_train,df[df.columns[5]]],axis=1)



#f_x=df[df.columns[66:68]]

df_train=pd.concat([df_train,df['Palette']],axis=1)
df_test=pd.concat([df_test,df_test_import['Palette']],axis=1)
#df_y=df_y.diff()
#df_y.to_csv(workfiles_path + 'dfy.csv')

#print(np.shape(df_train))
"""