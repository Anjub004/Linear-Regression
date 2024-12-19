import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error



from sklearn.linear_model import LinearRegression   
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import  GradientBoostingRegressor
from sklearn.ensemble import   AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import time
from tqdm import tqdm_notebook as tqdm


# 1. Fucntion to create piechart 

def create_piechart(data, column):
    """
    Objective
    ---------- 
    Create Pichart for Categorical varaibles present in Pandas Dataframe
    
    parameters
    ----------
    data: this is pandas dataframe
    column: this is column name which is used to create plot
        
    returns
    ----------
    this will show piechart
    
    """
    labels = list(data[column].value_counts().to_dict().keys())
    sizes = list(data[column].value_counts().to_dict().values())
   
    plt.pie(sizes, 
            labels=labels, 
            autopct='%1.2f%%',
            shadow=False, 
            startangle=45)
    
    plt.axis('equal')  
    plt.title("Piechart - {}".format(column))
    plt.show()
    
# 1. Fucntion to check missing data 
    
def missing_data(df, plot_missing=True):
    """
    Objective
    ----------
    it shows the missing data in each column with 
    total missing values, percentage of missing value and
    its data type in descending order.
    
    parameters
    ----------
    df: pandas dataframe
        input data frame 
    
    returns
    ----------
    missing_data: output data frame(pandas dataframe)
    
    """
    
    total = df.isnull().sum().sort_values(ascending=False)
    
    percent = round((df.isnull().sum()/df.isnull().count()  * 100).sort_values(ascending=False),2)
    
    data_type = df.dtypes
    missing_data = pd.concat([total,percent,data_type],
                             axis=1,
                             keys=['Total','Percent','Data_Type']).sort_values("Total", 
                                                                               axis = 0,
                                                                               ascending = False)


    if plot_missing:

        plt.figure(figsize=(15,5))
        sns.heatmap(df.isnull(), cbar = False, yticklabels=False, cmap="magma" )
        print()

        
    return missing_data.head(missing_data.shape[0])


def drop_duplicates(df):
    """
    Objective
    ----------
    Drop duplicates rows in data frame except for the first occurrence.
    
    parameters
    ----------
    df: pandas dataframe
        input data frame 
        
    returns
    ----------
    dataframe with all unique rows
    """
        
    try:
        dr = df.duplicated().value_counts()[1]
        print("[INFO] Dropping {} duplicates records...".format(dr))
        f_df = df.drop_duplicates(keep="first")
        
        return f_df
    except KeyError:
        print("[INFO] No duplicates records found")
        return df


    
def boxplot(df,width=20,height=200):
    """
    Objective
    ----------
    Draw a box plot to show distributions, skiping all the object variables
    (adjust the width and height to get best possible result)
    
    parameters
    ----------
    df: pandas dataframe
        input data frame 
    width: int
        width for box plot
    height: int
        height for box plot
        
    returns
    ----------
    matplotlib Axes
    Returns the Axes object with the plot drawn onto it.   
    """
    sns.set_theme(style="darkgrid")
    
    cols = list(df.select_dtypes(["float64","int64"]).columns)
    sns.set(style="whitegrid")
    fig, axs = plt.subplots(len(cols),figsize=(width,height))
    
    
    for i, col in enumerate(cols):
        sns.boxplot(df[col] , ax = axs[i])
        
        
def feature_scaling(X_train, X_test, method = "StandardScaler", return_df = False):
    """
    Objective
    ----------
    performs normalization or Standardization on input dataset 
    for feature scaling
    
    parameters
    ----------
    X_train: pandas dataframe
        all independent features in dataframe for training 
    
    X_test: pandas dataframe
        all independent features in dataframe for testing

    method : str , options "StandardScaler" or "MinMax" (dfault="StandardScaler")
        type of method to perform feature scaling
        "StandardScaler" is used for standardization
        and "MinMax" is used for Normalization
    
    return_df : bool (defualt=False)
        True will return the output in pandas Dataframe format
        False will return the output in array format

    returns 
    ----------
     X_train_scale, X_test_scale , scale  object
        return sclae data in array or dataframe format
        
    """
    if method == "StandardScaler":
        
        sc = StandardScaler()
        
        if return_df:
        
            # return data frame format
            X_train_scale = pd.DataFrame(sc.fit_transform(X_train),columns=X_train.columns)
            X_test_scale = pd.DataFrame(sc.transform(X_test),columns=X_test.columns)

            return X_train_scale , X_test_scale, sc
        else:
            
            # return array format
            X_train_scale =sc.fit_transform(X_train)
            X_test_scale =sc.transform(X_test)
            
            return X_train_scale , X_test_scale, sc
    
    elif method =="MinMax":
        
        mm_scaler = MinMaxScaler()
        
        if return_df:
        
            # return data frame format
            X_train_scale = pd.DataFrame(mm_scaler.fit_transform(X_train),columns=X_train.columns)
            X_test_scale = pd.DataFrame(mm_scaler.transform(X_test),columns=X_test.columns)

            return X_train_scale , X_test_scale, mm_scaler
        else:
            
            # return array format
            X_train_scale =mm_scaler.fit_transform(X_train)
            X_test_scale =mm_scaler.transform(X_test)
            
            return X_train_scale , X_test_scale , mm_scaler
        
        
# Helper function to plot cunfusion matrix and classification report 

def plot_confusion_metrix(y_true, y_pred,classes,
                         normalize=False,
                         title='Confusion Matrix',
                         cmap=plt.cm.Blues):
    """
    Objective
    ----------
    plot confussion matrix, classification report and accuracy score
    
    parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.
    
    classes : list
        List of labels to index the matrix
        
    title : title for matrix
    cmap : colormap for matrix 
    
    returns 
    ----------
   all accruacy matrix 
    """
    
    
    cm = confusion_matrix(y_true,y_pred)
    
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print("Confusion Matrix, Without Normalisation")

    
    plt.imshow(cm, interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=35)
    plt.yticks(tick_marks,classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() /2.
    
    for i , j in itertools.product(range(cm.shape[0]), range(cm.shape[0])):
        plt.text(j, i, format(cm[i,j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    
    print("-----------------------------------------------------")
    print('Classification report')
    print(classification_report(y_true,y_pred))
    
    acc= accuracy_score(y_true,y_pred)
    print("Accuracy of the model: ", acc)

   

def plot_auc_roc(model, x_data, y_data, figsize=(12,7), title=""):
    """
    Objective
    ----------
    plot ROC and AUC curve for any data 
    
    parameters
    ----------
    model : trained ml model for prediction 

    x_data : array, all_independt varaibles 
    y_data : array, dependt varaibles 

    
    figsize :size for the final plot default (12,7)
        
    title : title for final plot, defualt is ""
   
    returns 
    ----------
    plot ruc and auc curve 
    """
    
    # Get ROC curve and AUC.
    y_hat_proba = model.predict_proba(x_data)
    lr_fpr, lr_tpr, thresholds = roc_curve(y_data, y_hat_proba[:,1])
    lr_roc_auc = auc(lr_fpr, lr_tpr)

    # Plot ROC curve and AUC for our logistic regression model.
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111)
    ax1.fill_between(lr_fpr, lr_tpr, 0, color='Olive', alpha=0.6)
    ax1 = plt.plot(lr_fpr, 
                   lr_tpr, 
                   linewidth=3, 
                   label='Logistic Regression ROC_curve; AUC = %0.2f' % lr_roc_auc)
    plt.legend(loc='upper left')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{} ROC AUC Curve'.format(title), size=14)


    
    

def seaprate_features(df):
    """
    Objective
    ----------
    seprate numerical and categorical features from the dataset
    
    parameters
    ----------
    df : dataframe, full dataframe

   
    returns 
    ----------
    num : list, list of all numeric variables
    cat : list, list of all categorical variables
    """



    cat = [fea for fea in df.columns if df[fea].dtypes == 'O']
    num = [fea for fea in df.columns if df[fea].dtypes != 'O']
    
    print("[INFO] Found {} Numerical Features and {} Categorical Features".format(len(num),
                                                                                  len(cat)))
    return num, cat
    
def Get_outliers(df,column):

    """
    Objective
    ----------
    remove outlier from the dataset based on one specific column
    
    parameters
    ----------
    df : dataframe, input dataframe to remove outliers
    column:  str, columns to remove outlier
   
    returns 
    ----------
    dataframe with no outlier based on given input column
    """
    
   
    des = df[column].describe()
    
    desPairs = {"count":0,"mean":1,"std":2,"min":3,"25":4,"50":5,"75":6,"max":7}
    Q1 = des[desPairs['25']]
    Q3 = des[desPairs['75']]
    
    IQR = Q3-Q1
    lowerBound = Q1-1.5*IQR
    upperBound = Q3+1.5*IQR
    print("[INFO] (IQR = {}) Outlier are anything outside this range: ({},{})".format(IQR,lowerBound,upperBound))
    
    data = df[(df[column] < lowerBound) | (df[column] > upperBound)]

    print("[INFO] Total Outliers {1} out of {0}".format(df[column].size, len(data[column])))
   
    outlierRemoved = df[~df[column].isin(data[column])]
    return outlierRemoved

def regression_evaluation(y_actual, y_pred, X, y, display=False):
    """
    Objective
    ----------
    this function is used to evaluate regression models
    
    parameters
    ----------
    y_actual : series, gound truth of the data points
    y_pred:  series, prediction from the models
    X: independent variables
    y: dependent variables
    display: bool, display metircs 
   
    returns 
    ----------
    r2, adj_r_squre, rmse

    """
    r2=r2_score(y_actual,y_pred)
    adj_r_squre = 1 - (1-r2)*(len(y)-1)/(len(y)-X.shape[1]-1)
    rmse=np.sqrt(mean_squared_error(y_actual,y_pred))
    
    if display:
        print("[INFO] R Square : {}".format(r2))
        print("[INFO] Adjusted R Square : {}".format(adj_r_squre))
        print("[INFO] RMSE : {}".format(rmse))
        
    return (r2, adj_r_squre, rmse)


def regression_model(model_object, X_train, y_train, X_test, y_test, X, y, plot=False,data_points=50):

    """
    Objective
    ----------
    build a single regression model on train data and evaluate on test data
    
    parameters
    ----------
    X_train : array, training scaled data
    X_test:  array, testing Scaled data
    y_train : sereis, training target data 
    y_test: series, testing target data
    X: independent variables
    y: dependent variables
    plot: bool, default is False, used to plot actual and predicted values
    data_points: int, total no of data points for ploting

    returns 
    ----------
    r2, adj_r_squre, rmse, model_object
    """
    

    model_object.fit(X_train,y_train)
    y_pred = model_object.predict(X_test)
    r2, adj_r_squre, rmse = regression_evaluation(y_test, y_pred,X,y,display=plot)


    if plot:   
        test = pd.DataFrame({'Predicted':y_pred,'Actual':y_test})
        fig= plt.figure(figsize=(16,8))
        test = test.reset_index()
        test = test.drop(['index'],axis=1)
                     
        if data_points >= 0:

                     
            plt.plot(test[:data_points])
            plt.title("{}".format(str(model_object.__str__).split()[-4]))
            plt.legend(['Actual','Predicted']) 
        else:
            data_points = 50
            plt.plot(test[:data_points])
            plt.title("{}".format(str(model_object.__str__).split()[-4]))
            plt.legend(['Actual','Predicted']) 
                    

    return r2, adj_r_squre, rmse, model_object


def regression_model_analysis(X_train, y_train, X_test, y_test, X, y, plot=False, get_best=True,  metrics="Adjusted Rsqure" ):
    
    """
    Objective
    ----------
    run all the possible regression models on train data and return the best model
    
    parameters
    ----------
    X_train : array, training scaled data
    X_test:  array, testing Scaled data
    y_train : sereis, training target data 
    y_test: series, testing target data
    X: independent variables
    y: dependent variables
    plot: bool, default is False, used to plot actual and predicted values
    get_best: bool, default is True, means return the best model based on metrics
    metrics: str, default is Adjusted Rsquare, so based this metrics return the best model
   
    returns 
    ----------
    model report and best model
    """



    # create model objects
    models = {"Linear Regression":LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Support Vector Regression": SVR(kernel='linear'),
    "Linear SVR": LinearSVR(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "Gradient Boosting Regressor": GradientBoostingRegressor(),
    "Ada Boost Regressor": AdaBoostRegressor(),
    "KNeighbors Regressor":KNeighborsRegressor(),
    "MLP Regressor": MLPRegressor()
    }

    
    # create acc variables
    models_name = []
    all_r2 = []
    all_adj = []
    all_rmse = []
    time_taken = []
    all_models = []

    counter =  0
    
    
    # run foor loop on all models 
    with tqdm(total=len(models)) as pbar:
        for name, model in models.items():

            start = time.time()
            r2, adj_r_squre, rmse, model_object = regression_model(model_object=model,
                                                                   X_train=X_train,
                                                                   X_test=X_test,
                                                                   y_train=y_train, 
                                                                   y_test=y_test,
                                                                   X=X,
                                                                   y=y,
                                                                   plot=plot)

            t = time.time() - start
            
            
            models_name.append(name)
            all_r2.append(round(r2,2))
            all_adj.append(round(adj_r_squre,2))
            all_rmse.append(round(rmse,2))
            time_taken.append(t)
            all_models.append(model_object)
            pbar.update(1)

    # creat report for acc
    model_report = pd.DataFrame({"Model Name":models_name,
                                "R Squre":all_r2,
                                "Adjusted Rsqure": all_adj,
                                "RMSE":all_rmse,
                                "Time Taken":time_taken})
    # return best model
    if get_best:
        best_model_name = tuple(model_report[model_report[metrics] == model_report[metrics].max()]["Model Name"])[0]
        best_model_obj = models[best_model_name]
        
        return model_report, best_model_obj
    return model_report, None
