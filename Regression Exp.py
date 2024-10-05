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
    cols=[
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
    cols_to_drop=[
       'AP3_2_Actual_Part_to_Service_sb']
    #cols_poly=['AP3_2_Actual_Part_to_Service', '01', '02', '03', '04', '05', '06','07', '08']
    

    #print(df[cols])
    df2=df2.drop(columns=cols_to_drop)
    df=df[cols]
    #print(df2.columns)
    #print(df.columns)
    df_out=pd.concat([df2,df], axis=1)
    df_out=df_out.drop(len(df_out)-1)
    #df_tmp=df_out[cols_poly]
    #print(df_out.columns)
    #df_out=df_out.drop(columns=cols_poly)
    #df_out = poly.fit_transform(df_out)
    #print(df_out)
    #df_out.to_csv('tmp.csv')
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

#end_import=time.time()
outputs=['RESULT_InclinationBeltDirection__deg_',
       'RESULT_Inclination90ToBeltDirection__deg_']
#
df=df.replace('',np.nan)
df_test_import=df_test_import.replace('',np.nan)
df=df.replace('#VALUE!',np.nan)
df_test_import=df_test_import.replace('#VALUE!',np.nan)
df=df.dropna()
df_test_import=df_test_import.dropna()
df=df.reset_index(drop=True)
df_test_import=df_test_import.reset_index(drop=True)



df=df.loc[df.RESULT_InclinationBeltDirection__deg_>0.1]
df=df.loc[df.RESULT_InclinationBeltDirection__deg_<0.9]
df=df.loc[df.RESULT_Inclination90ToBeltDirection__deg_<0]
df=df.loc[df.RESULT_Inclination90ToBeltDirection__deg_>-0.7]


df = df.drop(columns=df.select_dtypes(include=['object']).columns)
print(pd.unique(df.dtypes))
columns=df.columns
df=step_back(df)
#df_test_import=df_test_import[df.columns]
df_cor=df.corr()
dfs=devide_df(df,4)
df_concat=pd.DataFrame()
poly = PolynomialFeatures(degree=2, include_bias=False)
for ds in dfs:
    df_poly=poly.fit_transform(ds)
    feature_names_train = poly.get_feature_names_out(ds.columns)
    df_concat=pd.concat([df_concat,df_poly],axis=0)

    
    
    
    





df2_poly=poly.fit_transform(df2)



feature_names_2train = poly.get_feature_names_out(df2.columns)

#df_train.to_csv('dfx0.csv')

# Create a new DataFrame with the polynomial features
df1 = pd.DataFrame(df1_poly, columns=feature_names_train)
df2 = pd.DataFrame(df2_poly, columns=feature_names_2train)


df_cor.to_csv("cor.csv")

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

df_train=df_train
df_test=df_test_import[df_train.columns]


#df_train=pd.concat([df_train,df[df.columns[2]]],axis=1)
#df_train=pd.concat([df_train,df[df.columns[3]]],axis=1)
#df_train=pd.concat([df_train,df[df.columns[5]]],axis=1)



#f_x=df[df.columns[66:68]]


#df_y=df_y.diff()
#df_y.to_csv(workfiles_path + 'dfy.csv')

#print(np.shape(df_train))
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



#df_test.to_csv(workfiles_path+'\\'+'dfx.csv')
#df=func.dyn_df(df,df_train)
df_train=df_train.reset_index(drop=True)
df_train_cat=df_train_cat.reset_index(drop=True)
df_test=df_test.reset_index(drop=True)
df_test_cat=df_test_cat.reset_index(drop=True)





#print(df_train)
#print(df_train_cat)
#print(df_test)
#print(df_test_cat)


df_train.to_csv('dfx0.csv')
df_train_cat.to_csv('dfxcat.csv')
#print(np.shape(df_train_cat))
df_train=step_back(df_train)
df_test=step_back(df_test)
print(df_train.columns)
print(df_test.columns)
#print(df_te)

df_y_train=df_train[outputs]
df_y_test=df_test[outputs]

df_train_temp=df_train.drop(columns=outputs)
df_test_temp=df_test.drop(columns=outputs)
print(df_train_temp.columns)
print(df_test.columns)


#print(df_train.columns)
poly = PolynomialFeatures(degree=2, include_bias=False)

df_train_poly=poly.fit_transform(df_train_temp)
df_test_poly=poly.fit_transform(df_test_temp)
feature_names_train = poly.get_feature_names_out(df_train_temp.columns)
feature_names_test = poly.get_feature_names_out(df_test_temp.columns)
#df_train.to_csv('dfx0.csv')

# Create a new DataFrame with the polynomial features
df_train = pd.DataFrame(df_train_poly, columns=feature_names_train)
df_test = pd.DataFrame(df_test_poly, columns=feature_names_test)

#df_train.to_csv('dfx_poly.csv')







df_train=pd.concat([df_train,df_train_cat],axis=1)    
df_test=pd.concat([df_test,df_test_cat],axis=1)

df_train=pd.concat([df_train,df_y_train],axis=1)    
df_test=pd.concat([df_test,df_y_test],axis=1)

#print(len(df_train.columns))
#print(len(df_test.columns))



#df_train=df_train.replace('',np.nan)
#df_test=df_test.replace('',np.nan)

#df_train=df_train.dropna()
#df_test=df_test.dropna()
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

#print(array)
#print(filtered_array)
#df_train=df_train[filtered_array]
#df_test=df_test[filtered_array]
#df_train=step_back(df_train)
#df_test=step_back(df_test)
#print(df_train) 
#from sklearn.preprocessing import FunctionTransformer

#df_train = poly.fit_transform(df_train)
#df_test=poly.fit_transform(df_test)
#log_transformer = FunctionTransformer(np.log)
#print(df_train)




arr=['AP3_2_Data_of_Actual_Position_Y_deg',
     'AP3_2_Data_of_Actual_Position_X_deg',
     'AP3_2_Data_of_Actual_Position_Z_mm',
     'AP3_2_Settings_M358_Montage_Position_mm',
     'AP3_2_Settings_M359_Montage_Position_mm']
     #'RESULT_ZWheelDownAverageSensor3__mm_',
     #  'RESULT_ZWheelDownAverageSensor4__mm_',
     #  'RESULT_ZWheelDownAverageDiff__mm_', 'RESULT_ZWheelDownAngle__deg_',
     #  'RESULT_ZAngle__deg_', 'RESULT_ZWheelUpAverageSensor3__mm_',
     # 'RESULT_ZWheelUpAverageSensor4__mm_', 'RESULT_XAngleMax__deg_',
     # 'RESULT_XWheelDownAverageDiff__mm_',
     # 'RESULT_XWheelUpAverageSensor1__mm_',
     # 'RESULT_XWheelUpAverageSensor2__mm_', 'RESULT_XWheelUpAverageDiff__mm_',
      # 'RESULT_XWheelUpAngle__deg_']
#for col in arr:
#    df_train=smooth(df_train,col)
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


#df_train=pd.concat([df_train,df_y],axis=1)
#print(df_train["AP3_2_Data_of_Actual_Position_Y_deg"])
#df_test=df_test[df_train.columns]

#print(df_train.columns)
#loaded_model = load_model('OUT1')
#print(loaded_model)
#predictions = predict_model(loaded_model, data=df_test)
#predictions.head()
#df_pred=pd.DataFrame(predictions)
#df_tst=pd.DataFrame(df_test['AP3_2_Data_of_Actual_Position_Y_deg'])
#df_pred.to_csv('dfpred.csv')
#df_tst.to_csv('df_tst.csv')
# functional API
#save_model(best, 'OUT1')
#profile = ProfileReport(df, title="DF set")
#profile.to_file("DF set report.html")
  

df_pred=pd.DataFrame()
result_path='C:\\Users\\zaizzhaim\\OneDrive - Mubea\\Documents\\02_101 Miling Machine Learning\\Results\\Experiments'
path='C:\\SVN\\'

       #'RESULT_MaximalForceAngle__deg_',
       #'RESULT_InclinationInMaximalForceDirection__deg_',
       #'RESULT_Perpendicularity__mm_']
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
       target=output, 
       #group_features=['RESULT_ZWheelDownAverageSensor3__mm_','RESULT_ZWheelDownAverageSensor4__mm_','RESULT_ZWheelDownAverageDiff__mm_','RESULT_ZWheelDownAngle__deg_','RESULT_ZAngle__deg_', 'RESULT_ZWheelUpAverageSensor3__mm_','RESULT_ZWheelUpAverageSensor4__mm_','RESULT_XAngleMax__deg_','RESULT_XWheelDownAverageDiff__mm_','RESULT_XWheelUpAverageSensor1__mm_','RESULT_XWheelUpAverageSensor2__mm_','RESULT_XWheelUpAverageDiff__mm_','RESULT_XWheelUpAngle__deg_'],
       test_data=df_test, 
       log_experiment = False, 
       experiment_name = output,
       index=False,
       use_gpu=True,
       log_data=True,
       normalize=normalize, 
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
    f_models=compare_models(n_select=8)
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
        df_f_pred.to_csv(result_path+'\\'+output+'.csv')
        #plot_model(model, plot='feature')
        plt.scatter(df_tmp[df_tmp.columns[0]],df_tmp[df_tmp.columns[1]])
        plt.scatter([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1],[-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
        plt.axis('equal')
        plt.show()
        

        

    #print(df_pred)
    #plot_model(best,plot='feature')
df_pred.to_csv('df_pred.csv')
    