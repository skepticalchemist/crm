import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def dataset_dimensions(df):
    """
    Show dimensions of the dataframe.
    """
    print("Dimensions of the dataset:")
    print(f" Number of rows: {df.shape[0]}")
    print(f" Number of columns: {df.shape[1]}\n")
    
    return


def column_unique_values(df):
    """
    Show unique values for each column in the dataframe.
    """
    for col in df.columns:
        print(f"{col: >24}: {df[col].nunique()}")
    
    return


def column_missing_values(df):
    """
    # Total number of missing values per columns, if they are present.
    """
    print("Columns with missing values:\n")

    return df.isnull().sum()[df.isnull().sum() > 0]        


def feature_engineering(df):
    """
    Perform data cleaning operations.
    """
    # remove rows with missing values
    df.dropna(axis=0, how="any", inplace=True)
    df = df.reset_index(drop=True)

    # removing the column id
    #df.drop(["id"], inplace=True, axis=1)

    # convert column from boolean to categorical
    df["gender"] = pd.Categorical(df.gender)
    #df["is_cardiologist"] = pd.Categorical(df.is_cardiologist)
    #df["is_gp"] = pd.Categorical(df.is_gp)

    # convert column from object to categorical
    df["office_or_hospital_based"] = pd.Categorical(df.office_or_hospital_based)
    
    # office=1 and hospital=0, rename column to office 
    #df.office_or_hospital_based.replace(['Office', 'Hospital'], [1, 0], inplace=True)
    #df.rename(columns={'office_or_hospital_based': 'office'}, inplace=True)

    # convert column from float to integer
    df["years_since_graduation"] = df["years_since_graduation"].astype(int)

    # replace True=male and False=female
    df["gender"].replace(True, "male", inplace=True)
    df["gender"].replace(False, "female", inplace=True)

    # replace True=1 and False=0
    df["is_cardiologist"].replace(True, "1", inplace=True)
    df["is_cardiologist"].replace(False, "0", inplace=True)
    df["is_cardiologist"] = pd.to_numeric(df["is_cardiologist"])
    
    # replace True=1, False=0
    df["is_gp"].replace(True, "1", inplace=True)
    df["is_gp"].replace(False, "0", inplace=True)
    df["is_gp"] = pd.to_numeric(df["is_gp"])

    return df


def do_one_hot(df, column_list):
    """
    Perform one-hot encoding
    """
    for col in column_list:
        # create the encoder
        encoder = OneHotEncoder(sparse=False)
        # encode the data
        onehot_result = pd.DataFrame(encoder.fit_transform(df[[col]]))
        # get feature names  
        onehot_result.columns = encoder.get_feature_names([col])
        # drop original column  
        df.drop([col], axis=1, inplace=True)
        # concatenate dataframes
        df = pd.concat([df, onehot_result], axis=1)

    return df


def plot_donut(df_col, labels, fig_title, plt, text_color='black', 
           font_size=16.0, explode=(0.01, 0.01), 
           circle_radius=0.5, title_color='blue',
           colors=['b', 'c']):
    '''
    Creates a donut plot.
    '''
    labels = labels
    explode = explode
    plt.rcParams['text.color'] = text_color
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.titlecolor'] = title_color
    plt.rcParams['axes.titlesize'] = 20
        
    #theme = plt.get_cmap('PiYG')
    #theme = plt.get_cmap('Set1')
    #theme = plt.get_cmap('gist_rainbow')
    #colors =[theme(1. * i / len(labels)) for i in range(len(labels))]
    colors=colors
    
    plt.pie(df_col.value_counts(), 
            labels=labels, 
            labeldistance=1.2,
            explode=explode, 
            shadow=True, 
            wedgeprops={'linewidth': 2},
            textprops={"fontsize":18},
            pctdistance=0.7,  # position of % numbers
            autopct="%.1f%%",
            colors=colors,
    )
    # add a circle at the center to transform it in a donut chart
    my_circle = plt.Circle( (0,0), circle_radius, color='white')
    p = plt.gcf()
    p.gca().add_artist(my_circle)
    
    plt.title(fig_title, loc='center', y=1.1)
    plt.axis('equal')
    plt.tight_layout()


def multiplot(data_frame, column_list, title, xlabel, ylabel, y_lim, width=0.3, fig_size=(14, 8), n_row=3, n_col=4):
    fig = plt.figure()
    num = 0
    fig.suptitle(title, fontsize=22, y=1.01)
    for item in column_list:
        num += 1
        counts = data_frame[item].value_counts(normalize=False).sort_index(ascending=True)
        ax = fig.add_subplot(n_row, n_col, num)
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=18)
        # ax.text(1000, 2, f'{num}')        
        plt.title(f"{item}", fontsize=16)
        ax.label_outer()
        plt.ylim(0, y_lim)
        counts.plot.bar(width=width, figsize=fig_size, rot=0)
        plt.grid()

    plt.tight_layout()
    # plt.savefig('xlabel.jpg')
    plt.show()


def add_columns(data_frame, suffix, sep):
    """
    Add all columns and remove them.
    """
    for i in range(1, 13):
        if i == 1:
            col = (f'{suffix}{sep}0{i}')
            data_frame[suffix] = data_frame[col]
            data_frame.drop([col], inplace=True, axis=1)
        if (1 < i < 10):       
            col = (f'{suffix}{sep}0{i}')
            data_frame[suffix] += data_frame[col]
            data_frame.drop([col], inplace=True, axis=1)
        if i >= 10:
            col = (f'{suffix}{sep}{i}')
            data_frame[suffix] += data_frame[col] 
            data_frame.drop([col], inplace=True, axis=1)
    return data_frame


def add_drop_columns(data_frame, suffix, sep):
    """
    Remove columns with overlaping information.
    Add columns 3, 6, 9, and 12.
    """
    for i in range(1, 13):
        
        drop_list_1 = [1, 2, 4, 5, 7, 8]
        drop_list_2 = [10, 11]
        
        if i in drop_list_1:
            col = (f'{suffix}{sep}0{i}')
            data_frame.drop([col], inplace=True, axis=1)
        
        if i in drop_list_2:
            col = (f'{suffix}{sep}{i}')
            data_frame.drop([col], inplace=True, axis=1)
            
        if i == 3:
            col = (f'{suffix}{sep}0{i}')
            data_frame[suffix] = data_frame[col]
            data_frame.drop([col], inplace=True, axis=1)
            
        if (i == 6 or i ==9):       
            col = (f'{suffix}{sep}0{i}')
            data_frame[suffix] += data_frame[col]
            data_frame.drop([col], inplace=True, axis=1)
            
        if i == 12:
            col = (f'{suffix}{sep}{i}')
            data_frame[suffix] += data_frame[col] 
            data_frame.drop([col], inplace=True, axis=1)
            
    return data_frame


def drop_selected_cols(data_frame, suffix, sep):
    """
    Remove columns with overlaping information.
   Add columns 3, 6, 9, and 12.
   """
    for i in range(1, 13):
       
        drop_list_1 = [1, 2, 4, 5, 7, 8]
        drop_list_2 = [10, 11]
       
        if i in drop_list_1:
            col = (f'{suffix}{sep}0{i}')
            data_frame.drop([col], inplace=True, axis=1)
       
        if i in drop_list_2:
            col = (f'{suffix}{sep}{i}')
            data_frame.drop([col], inplace=True, axis=1)
                 
    return data_frame


    # defining evaluation metric
def compute_metrics(y_test, y_pred, model_name):
    """
    Compute MSE, RMSE and MAE between predictions from a model
    and the actual values of the target variable.
    """
    mse = np.round(mean_squared_error(y_test, y_pred), 2)
    rmse = np.round(sqrt(mse), 2)
    mae = np.round(mean_absolute_error(y_test, y_pred), 2)
    
    # rounding to 2 decimal places
    print(f"MODEL: {model_name}")
    print(20*"-=")
    print(f"MSE  = {mse:>6}")
    print(f"RMSE = {rmse:>6}")
    print(f"MAE  = {mae:>6}")
    
    return 