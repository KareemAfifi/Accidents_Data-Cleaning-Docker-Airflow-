from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

import pandas as pd
import numpy as np
# For Label Encoding
from sklearn import preprocessing


import dash
import dash_core_components as dcc
import dash_html_components as html
from sqlalchemy import create_engine

import plotly.express as px
from dash import Dash, dcc, html, Input, Output

#My Imports:
#import seaborn as sns
#import matplotlib.pyplot as plt
import datetime

from sklearn.preprocessing import MinMaxScaler
import csv
#%matplotlib inline
from scipy import stats
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

dataset = '2000_Accidents_UK.csv'

def preprocessing_dataset(filename):
    df = pd.read_csv(filename,index_col='accident_index')
    df_observe = df.copy()
    df_observe.loc[df_observe.first_road_class == 'C', 'first_road_number'] = 0.0
    df_observe.loc[df_observe.first_road_class == 'Unclassified', 'first_road_number'] = 0.0
    df_observe.loc[df_observe.second_road_class == 'C', 'second_road_number'] = 0.0
    df_observe.loc[df_observe.second_road_class == 'Unclassified', 'second_road_number'] = 0.0
    df_observe.loc[df_observe.second_road_class == '-1', 'second_road_number'] =  -1
    df_observe.loc[df_observe.second_road_class == '-1', 'second_road_class'] = 'Not Present'
    df_observe.loc[df_observe.lsoa_of_accident_location == -1, 'lsoa_of_accident_location'] = np.nan
    df_observe.loc[df_observe.lsoa_of_accident_location == '-1', 'lsoa_of_accident_location'] = np.nan
    df_observe_junction = df_observe.copy()
    df_observe_junction.junction_control = df_observe_junction.junction_control.apply(lambda row: '-1' if ('missing' in row) else row)
    df_observe_junction.first_road_number.astype(np.float64)
    df_observe_junction.second_road_number.astype(np.float64)
    df_observe_missing = df_observe_junction.copy()
    for col in df_observe_missing:
        if df_observe_missing[col].dtype==object:
            df_observe_missing.loc[df_observe_missing[col].str.contains('missing',na=False), col] = np.nan
    df_missing_osgr = df_observe_missing.copy()
    df_missing_osgr.dropna(axis='index',how='all' ,subset=['location_easting_osgr'],inplace=True)
    df_missing_surface = df_missing_osgr.copy()
    df_missing_surface.dropna(axis='index',how='all' ,subset=['road_surface_conditions'],inplace=True)
    df_missing_police = df_missing_surface.copy()
    df_missing_police.did_police_officer_attend_scene_of_accident=df_missing_police.did_police_officer_attend_scene_of_accident.fillna('No')
    df_missing_junction = df_missing_police.copy()
    df_missing_junction.junction_detail = df_missing_junction.junction_detail.fillna(df_missing_junction[df_missing_junction['second_road_number'] == -1].junction_detail.mode()[0])
    df_missing_road_type = df_missing_junction.copy()
    df_missing_road_type_grouped = df_missing_road_type.groupby(['first_road_class'])['road_type'].agg(pd.Series.mode)
    dictionary = dict(df_missing_road_type_grouped)

    df_missing_road_type.road_type = df_missing_road_type.road_type.fillna(df_missing_road_type.first_road_class.map(dictionary))
    df_missing_trunk = df_missing_road_type.copy()
    
    grouping_of_speed_limit = df_missing_trunk.groupby(['speed_limit'])['trunk_road_flag'].agg(pd.Series.mode)
    dictionary = dict(grouping_of_speed_limit)

    df_missing_trunk.trunk_road_flag = df_missing_trunk.trunk_road_flag.fillna( df_missing_trunk.speed_limit.map(dictionary))
    
    df_missing_weather = df_missing_trunk.copy()
    
    grouping_of_weather_conditions= df_missing_weather.groupby(['road_surface_conditions'])['weather_conditions'].agg(pd.Series.mode)
    dictionary = dict(grouping_of_weather_conditions)

    df_missing_weather.weather_conditions = df_missing_weather.weather_conditions.fillna(df_missing_weather.road_surface_conditions.map(dictionary))

    df_missing_lsoa = df_missing_weather.copy()
    
    grouping_of_lsoa = df_missing_lsoa.groupby(['local_authority_district'])['lsoa_of_accident_location'].agg(pd.Series.mode)
    
    lsoa = pd.Series(grouping_of_lsoa)

    count = 0 
    for i,v in lsoa.items():
        if np.size(v) ==0:
            lsoa[count] = 'Not Available'
        if type(lsoa[count])==np.ndarray:
            lsoa[count] = lsoa[count] [0]
        count+=1
    dictionary = dict(lsoa)
    
    df_missing_lsoa.lsoa_of_accident_location = df_missing_lsoa.lsoa_of_accident_location.fillna(df_missing_lsoa.local_authority_district.map(dictionary))
    df_missing_pedestrian = df_missing_lsoa.copy()
    df_missing_pedestrian.pedestrian_crossing_human_control= df_missing_pedestrian.pedestrian_crossing_human_control.fillna(df_missing_pedestrian.pedestrian_crossing_human_control.mode()[0])
    
    grouping_of_physical_conditions= df_missing_pedestrian.groupby(['pedestrian_crossing_human_control'])['pedestrian_crossing_physical_facilities'].agg(pd.Series.mode)
    dictionary = dict(grouping_of_physical_conditions)

    df_missing_pedestrian.pedestrian_crossing_physical_facilities = df_missing_pedestrian.pedestrian_crossing_physical_facilities.fillna(df_missing_pedestrian.pedestrian_crossing_human_control.map(dictionary)  )
    
    df_missing_drop = df_missing_pedestrian.copy()
    
    df_missing_drop.dropna(axis='index', how='all', subset=['first_road_number'], inplace=True)
    df_missing_drop.dropna(axis='index', how='all', subset=['second_road_number'], inplace=True)
    df_missing_drop.dropna(axis='index', how='all', subset=['light_conditions'], inplace=True)
    df_missing_drop.dropna(axis='index', how='all', subset=['carriageway_hazards'], inplace=True)
    df_missing_drop.dropna(axis='index', how='all', subset=['carriageway_hazards'], inplace=True)
    df_missing_drop.dropna(axis='index', how='all', subset=['special_conditions_at_site'], inplace=True)
        
    clean_df = df_missing_drop.copy()
    outliers = clean_df.copy()
    Q1 = outliers.number_of_vehicles.quantile(0.25)
    Q3 = outliers.number_of_vehicles.quantile(0.75)
    IQR = Q3 - Q1
    cut_off = IQR * 1.5
    lower = Q1 - cut_off
    upper =  Q3 + cut_off
    df1 = outliers[outliers.number_of_vehicles > upper]
    df2 = outliers[outliers.number_of_vehicles < lower]
    
    Q1 = outliers.number_of_casualties.quantile(0.25)
    Q3 = outliers.number_of_casualties.quantile(0.75)

    IQR = Q3 - Q1
    cut_off = IQR * 1.5
    lower = Q1 - cut_off
    upper =  Q3 + cut_off

    df1 = outliers[outliers.number_of_casualties > upper]
    

    df2 = outliers[outliers.number_of_casualties < lower]
    
    median = outliers.number_of_vehicles.median()
    cutoff = outliers.number_of_vehicles.mean() + outliers.number_of_vehicles.std() * 3

    outliers.number_of_vehicles = np.where(outliers.number_of_vehicles>cutoff, median,outliers.number_of_vehicles)
    
    
    
    outliers.local_authority_district.replace('Edinburgh, City of' ,'City of Edinburgh' ,inplace=True)
    outliers.local_authority_district.replace('Rhondda, Cynon, Taff' ,'Rhondda Cynon Taf' ,inplace=True)
    outliers.local_authority_district.replace('Kingston upon Hull, City of' ,'Kingston upon Hull' ,inplace=True)
    outliers.local_authority_district.replace('Bristol, City of' ,'City of Bristol' ,inplace=True)
    outliers.local_authority_district.replace('Herefordshire, County of ' ,'County of Herefordshire' ,inplace=True)
    outliers.local_authority_district.replace('Southampton ' ,'Southampton' ,inplace=True)
    outliers.local_authority_district.replace('Rugby ' ,'Rugby' ,inplace=True)

    outliers.local_authority_ons_district.replace('Edinburgh, City of' ,'City of Edinburgh' ,inplace=True)
    outliers.local_authority_ons_district.replace('Rhondda, Cynon, Taff' ,'Rhondda Cynon Taf' ,inplace=True)
    outliers.local_authority_ons_district.replace('Kingston upon Hull, City of' ,'Kingston upon Hull' ,inplace=True)
    outliers.local_authority_ons_district.replace('Bristol, City of' ,'City of Bristol' ,inplace=True)
    outliers.local_authority_ons_district.replace('Herefordshire, County of ' ,'County of Herefordshire' ,inplace=True)
    outliers.local_authority_ons_district.replace('Southampton ' ,'Southampton' ,inplace=True)
    outliers.local_authority_ons_district.replace('Rugby ' ,'Rugby' ,inplace=True)

    outliers.local_authority_highway.replace('Edinburgh, City of' ,'City of Edinburgh' ,inplace=True)
    outliers.local_authority_highway.replace('Rhondda, Cynon, Taff' ,'Rhondda Cynon Taf' ,inplace=True)
    outliers.local_authority_highway.replace('Kingston upon Hull, City of' ,'Kingston upon Hull' ,inplace=True)
    outliers.local_authority_highway.replace('Bristol, City of' ,'City of Bristol' ,inplace=True)
    outliers.local_authority_highway.replace('Herefordshire, County of ' ,'County of Herefordshire' ,inplace=True)
    outliers.local_authority_highway.replace('Southampton ' ,'Southampton' ,inplace=True)
    outliers.local_authority_highway.replace('Rugby ' ,'Rugby' ,inplace=True)
    
    
    
    
    number_casualties_imputation = outliers.copy()
    floor = number_casualties_imputation['number_of_casualties'].quantile(0.10)
    cap = number_casualties_imputation['number_of_casualties'].quantile(0.90)
    
    #filling them with the quantile-based flooring and capping

    number_casualties_imputation['number_of_casualties'] = np.where(number_casualties_imputation['number_of_casualties'] < floor, floor, number_casualties_imputation['number_of_casualties'])
    number_casualties_imputation['number_of_casualties'] = np.where(number_casualties_imputation['number_of_casualties'] > cap, cap, number_casualties_imputation['number_of_casualties'])
    imputed_df = number_casualties_imputation.copy()
    
    local_authority_df = imputed_df.copy()
    local_authority_df['isEqual'] = local_authority_df.apply(lambda x: x['local_authority_ons_district']+' , '+x['local_authority_district'] if x['local_authority_ons_district'] !=
                                                   x['local_authority_district'] else True, axis=1)
   
   
   
   
    
    #CONTUNIATION OF MILESTONE 1
    
    
    
    df_merged_population= imputed_df.copy()
    df_merged_population.drop('accident_year',inplace=True,axis=1)
    transformation_df = df_merged_population.copy()
    adding_col_week_number = transformation_df.copy()
    #Changing the Type of the Date Column to DataTime
    adding_col_week_number.date= pd.to_datetime(adding_col_week_number.date, dayfirst=True) 
    adding_col_week_number['week_number'] = adding_col_week_number['date'].dt.isocalendar().week
    check_to_encoded = adding_col_week_number.copy()
    categorical_features = check_to_encoded.select_dtypes(exclude=["number","bool_"])
    df_test = check_to_encoded.copy()
    encoded_df = check_to_encoded.copy()
    #Encoding Variables Methods:
    to_be_label_encoded= ['police_force','day_of_week','local_authority_district','local_authority_ons_district','local_authority_highway','pedestrian_crossing_physical_facilities',
                         'light_conditions' , 'weather_conditions', 'road_surface_conditions', 'special_conditions_at_site',
                          'carriageway_hazards']

    to_be_hot_encoded = ['accident_severity','first_road_class','road_type','junction_control','second_road_class', 
                         'pedestrian_crossing_human_control','urban_or_rural_area']

    to_be_hot_encoded_most_frequent = ['junction_detail']

    #Binary Data
    to_be_binary_data = ['did_police_officer_attend_scene_of_accident','trunk_road_flag']
    
    #Label Encoding:
    for label in to_be_label_encoded:
        c = encoded_df[label+''].astype('category')
        d = dict(enumerate(c.cat.categories))
        encoded_df[label+'']=preprocessing.LabelEncoder().fit_transform(encoded_df[label+''])
        d = pd.DataFrame.from_dict(d,orient='index')
        d.rename(columns={0:''+label},inplace=True)
        d.to_csv('/opt/airflow/data/'+label+'.csv')
        #d.to_csv(label+'.csv')

    # One-Hot Frequent Encoding:
    for col in to_be_hot_encoded_most_frequent:
        top_x = calculate_top_categories(encoded_df,col,4)
        one_hot_encoded_frequency(encoded_df,col,top_x)
        encoded_df.drop(col+'',inplace=True,axis=1)


    for label in to_be_hot_encoded:
        one_hot_encode(encoded_df,label+'')
        encoded_df.drop(label+'',inplace=True,axis=1)
        
    #Binary Encoding:
    for label in to_be_binary_data:
        c = encoded_df[label+''].astype('category')
        d = dict(enumerate(c.cat.categories))
        encoded_df[label+''] = preprocessing.LabelEncoder().fit_transform(encoded_df[label+''])
        d = pd.DataFrame.from_dict(d,orient='index')
        d.rename(columns={0:''+label},inplace=True)
        d.to_csv('/opt/airflow/data/'+label+'.csv')
        #d.to_csv(label+'.csv')
    

    
    normalised_df = encoded_df.copy()
    df_normalization = normalised_df.copy()
    df_normalized_original_number_of_casualties = df_normalization.number_of_casualties
    df_normalization.number_of_casualties = MinMaxScaler().fit_transform(df_normalization[['number_of_casualties']])
    
    df_normalization_vehicles = normalised_df.copy()
    df_normalized_original_number_of_vehicles = df_normalization_vehicles.number_of_vehicles
    df_normalization_vehicles.number_of_vehicles = MinMaxScaler().fit_transform(df_normalization_vehicles[['number_of_vehicles']])
    
    df_normalization_speed = df_normalization_vehicles.copy()
    df_normalized_original_speed_limit = df_normalization_speed.speed_limit.copy()
    
    df_normalization_speed.speed_limit = MinMaxScaler().fit_transform(df_normalization_speed[['speed_limit']])
    
    
    
    is_weekend_df = df_normalization_speed.copy()
    
    is_weekend_df['isweekend'] = is_weekend_df.apply(lambda row: 1 if row.day_of_week in ['Sunday','Saturday'] else 0, axis =1)
    
    #Adding isMidnight column 12 AM->5 AM
    is_midnight_df = is_weekend_df.copy()
    starting_midnight = datetime.strptime('00:00', "%H:%M").time()
    ending_midnight = datetime.strptime('05:00', "%H:%M").time()

    is_midnight_df['ismidnight'] = is_midnight_df.time.apply(lambda row: 1 if (datetime.strptime(str(row), "%H:%M").time()>starting_midnight and datetime.strptime(str(row), "%H:%M").time()<ending_midnight ) else 0)
    
    is_duplicated_df = is_midnight_df.copy()
    
  
    #Observing Duplicated Value :
    trues = is_duplicated_df.duplicated(subset=is_duplicated_df.columns[3:],keep=False)
    number_of_duplicated_values = is_duplicated_df.duplicated(subset=is_duplicated_df.columns[3:])


    new_df = is_duplicated_df.copy()
    new_df=new_df[0:0]

    for i in range(len(trues)):
        if trues[i]==True:
            new_df  =new_df.append(is_duplicated_df.iloc[i].to_dict(),ignore_index=True)

    if number_of_duplicated_values.sum()!=0:
        is_duplicated_df.drop_duplicates( subset=is_duplicated_df.columns[3:],keep='first', inplace=True)# Here Inplace =TRUE

    binning_date_df = is_duplicated_df.copy()
    
    
    
    
    
    #BINNING THE DATE ATTRIBUTE
    binneddate, intervals = pd.qcut(binning_date_df.date,10, labels=None, retbins=True, precision=3, duplicates='raise')
    binneddata2= pd.DataFrame(binneddate)
    
    result = binning_date_df.merge(binneddata2, on=binning_date_df.index)
    
    result.rename(columns = {'date_x':'date', 'date_y':'binned_date','key_0':'accident_index'}, inplace = True)
    result.set_index('accident_index')
    print(result.info())
    print(result.head())
    
    
    #result.drop('accident_index',axis=1,inplace=True)
    #result.drop('index',axis=1, inplace=True)
    final_df = result.copy()
    
    
    final_df.to_csv('/opt/airflow/data/accidents_clean.csv',index=False)
    print('PreProcessing Part 1 Completed')
    
    
    
def data_integrations_m2(filename):
    s=requests.session()
    link_api = 'https://www.citypopulation.de/en/uk/admin/'
    req=s.post(link_api)
    sooup =BeautifulSoup(req.text)
    table_area = sooup.find(id="adminareas")
    table_itself = table_area.find(id="tl")
    content = []
    yellow_rows = table_itself.find_all(class_="admin1")
    white_rows = table_itself.find_all(class_="admin2")
    for i in white_rows:
        temp_array = i.find_all('tr')
        for j in temp_array:
            name = j.td.a.get_text()
            name=name.strip()
            if name == 'St Helens':
                name = "St. Helens"
            if name == 'St Albans':
                name = "St. Albans"
            if name == 'Stratford-on-Avon':
                name = "Stratford-upon-Avon"
            number = j.find(class_="rpop prio3").get_text()
            number=number.replace(',','')
            number=int(number)
            content.append([name,number])
     
    
    
    
    #Return as Previous
    population_df= pd.read_csv(filename)
    local_authority_lookup= pd.read_csv('/opt/airflow/data/local_authority_district.csv',index_col=0)
    population_df = pd.merge(population_df ,local_authority_lookup ,left_on= 'local_authority_district' , right_on = local_authority_lookup.index)
    population_df.rename(columns={'local_authority_district':'local_authority_district_encoded'},inplace=True)
    #population_df.rename(columns={'0':'local_authority_district'},inplace=True)
    population_df.rename(columns={'local_authority_district_y':'local_authority_district'},inplace=True)
    population_df.drop('local_authority_district_x',axis=1, inplace=True)
    #   ----------------

    
    
    districts_populations = pd.DataFrame(content, columns =['local_authority_district', 'population'])
    df_merged_population =population_df.reset_index().merge(districts_populations, on ='local_authority_district',how='left')
    for i,row in df_merged_population.iterrows():
        if(pd.isnull(row.population)):
            district_name = row.local_authority_district
            for m in content :
                a= m[0].replace('City of ','')
                a= a.replace('East ','')
                a= a.replace('West ','')
                a= a.replace('North ','')
                a= a.replace('South','')
                a= a.replace(' of ','')
                if(a in district_name):
                    df_merged_population.iloc[i, df_merged_population.columns.get_loc('population')]= m[1]
                    break


    
    df_merged_population.set_index(['accident_index'])
    mean_value_of_population=int(df_merged_population.population.mean())
    df_merged_population.population= df_merged_population.population.fillna(mean_value_of_population)
    
    
    df_merged_population.drop('index',axis=1,inplace=True)
    df_merged_population.drop('local_authority_district',axis=1,inplace=True)
    df_merged_population.rename(columns={'local_authority_district_encoded':'local_authority_district'},inplace=True)
    
    #Normalization for Population :::
    df_normalization_population = df_merged_population.copy()
    df_normalized_original_population = df_normalization_population.population.copy()
    df_normalization_population.population = MinMaxScaler().fit_transform(df_normalization_population[['population']])
    
    
    try:
        df_normalization_population.to_csv('/opt/airflow/data/UK_Accidents_2000.csv', mode='x',index=False)
        print('loaded after cleaning succesfully')
    except FileExistsError:
        print('File already exists')


    
    
 



def calculate_top_categories(df, variable, how_many):
    return [
        x for x in df[variable].value_counts().sort_values(
            ascending=False).head(how_many).index
    ]
def one_hot_encoded_frequency(df, variable, top_x_labels):
    for label in top_x_labels:
        df[variable + '_' + label] = np.where(
            df[variable] == label, 1, 0) 


def one_hot_encode(df, variable):
    for label in df[variable].unique():
        df[variable + '_' + label] = np.where(df[variable] == label, 1, 0)
#-------------------------------------------------------------------------------------------------
# Methods For DashBoards
def top_states_by_accidents(filename):
    df= pd.read_csv(filename,index_col='accident_index')
    df2 = pd.read_csv('/opt/airflow/data/local_authority_district.csv',index_col=0)
    df= pd.merge(df, df2,left_on='local_authority_district',right_on=df2.index)
    districts_by_accident = df.local_authority_district_y.value_counts()
    top_10_districts_by_accident = districts_by_accident[:10]
    #top_10_districts_by_accident[:10].plot(kind = 'barh',title="Top 10 States by Number of Accidents")
    fig_title="Top 10 States by Number of Accidents"
    fig = px.bar(top_10_districts_by_accident[:10],orientation='h')
    fig.update_layout(
    title=fig_title,
    xaxis_title="Number of Accidents",
    yaxis_title="City Name")
    fig.update_layout(showlegend=False)
    return fig
    
def time_accident_count(filename):
    df= pd.read_csv(filename,index_col='accident_index')
    accidents_by_hour_df = pd.to_datetime(df['time'].astype(str)).dt.hour
    accidents_by_hour_df = pd.to_datetime(df['time'].astype(str)).dt.hour
    #sns.histplot(accidents_by_hour_df, bins = 24,kde=True).set(title='Distribution of the percentage of accidents throughout the day')
    x = pd.DataFrame(accidents_by_hour_df)
    x= pd.DataFrame(x.groupby(['time']).time.count())
    fig_title="Distribution of the percentage of accidents throughout the day"
    fig = px.histogram(accidents_by_hour_df)
    fig.update_layout(
    title=fig_title,
    xaxis_title="Time of Day",
    yaxis_title="Number of Accidents")
    fig.update_layout(showlegend=False,bargap=0.1)
    return fig

def total_casualties_by_accident(filename):
    df= pd.read_csv(filename,index_col='accident_index')
    accidents_by_month_df = df.copy()
    accidents_by_month_df.date= pd.to_datetime(accidents_by_month_df.date, dayfirst=True)
    accidents_by_month_df.date=accidents_by_month_df.date.dt.month
    DataFrame = pd.DataFrame(accidents_by_month_df.groupby(['date'])['number_of_casualties'].sum())
    
    fig_title='Total Number of Casualties Relative in a given Month'
    fig = px.line(DataFrame)
    fig.update_layout(
    title=fig_title,
    xaxis_title="Time of Day",
    yaxis_title="Number of Accidents")
    fig.update_layout(showlegend=False,bargap=0.1)
    return fig

def casualties_by_road_type(filename):
    df= pd.read_csv(filename,index_col='accident_index')
    df2 = pd.read_csv('/opt/airflow/data/road_surface_conditions.csv',index_col=0)
    df= pd.merge(df, df2,left_on='road_surface_conditions',right_on=df2.index)
    
    casualties_by_road_type = df.copy().groupby(['road_surface_conditions_y'])['number_of_casualties'].sum()
    
    fig_title="Number of Casualties depending on Road Type"
    fig = px.bar(casualties_by_road_type,orientation='h')
    fig.update_layout(
    title=fig_title,
    xaxis_title="Number of Accidents",
    yaxis_title="Road Condition")
    fig.update_layout(showlegend=False)
    return fig

def accidents_dayofweek(filename):
    df= pd.read_csv(filename,index_col='accident_index')
    df2 = pd.read_csv('/opt/airflow/data/day_of_week.csv',index_col=0)
    df= pd.merge(df, df2,left_on='day_of_week',right_on=df2.index)
    accidents_by_week_df = df.groupby(['day_of_week_y'],as_index=False)['time'].count()
    accidents_by_week_df=accidents_by_week_df.sort_values(by=['day_of_week_y'])
    #return accidents_by_week_df
    fig_title="Number of Accidents relative to the Days of the Week"
    fig = px.histogram(accidents_by_week_df, x= 'day_of_week_y',y ='time',range_y=[20000,40000])
    fig.update_layout(
    title=fig_title,
    xaxis_title="Time of Day",
    yaxis_title="Number of Accidents")
    fig.update_layout(showlegend=False,bargap=0.1)
    return fig    





def create_dashboard(filename):
    df = pd.read_csv(filename)
    app = dash.Dash()
    app.layout = html.Div([
    html.H1("Milestone 3 Application Dashboards with Dash", style={'text-align': 'center'}),
    html.Br(),
    html.H1("Accident dataset", style={'text-align': 'center'}),
    html.Br(),
    html.Div(),
    html.H1("Top 10 States by Number of Accidents", style={'text-align': 'center'}),
    dcc.Graph(figure=top_states_by_accidents(filename)),
    html.Br(),
    html.Div(),
    
    html.H1("Distribution of the percentage of accidents throughout the day", style={'text-align': 'center'}),
    dcc.Graph(figure=time_accident_count(filename)),
    html.Br(),
    html.Div(),
    
    html.H1('Total Number of Casualties Relative in a given Month', style={'text-align': 'center'}),
    dcc.Graph(figure=total_casualties_by_accident(filename)),
    html.Br(),
    html.Div(),
    
    html.H1("Number of Casualties depending on Road Type", style={'text-align': 'center'}),
    dcc.Graph(figure=casualties_by_road_type(filename)),
    html.Br(),
    html.Div(),
    
    html.H1("Number of Accidents relative to the Days of the Week", style={'text-align': 'center'}),
    dcc.Graph(figure=accidents_dayofweek(filename)),
    html.Br(),
    html.Div(),
    
])
    app.run_server(host='0.0.0.0')
    print('dashboard is successful and running on port 8000')





# ###Dont forget saving to postgres and saving the lookk up tables as well
# #---------------------------------------------------------------------------------------
def load_to_postgres(filename): 
    df = pd.read_csv(filename)
    engine = create_engine('postgresql://postgres:postgres@pgdatabase:5432/accidents_etl')
    if(engine.connect()):
        print('connected succesfully')
    else:
        print('failed to connect')
    df.to_sql(name = 'UK_Accidents_2000',con = engine,if_exists='replace',index=False)
    
    #SAVING THE LOOKUP-Table
    print("SAVING THE LOOKUP-Table")
    master_df = pd.DataFrame()
    files_to_be_merged= ['day_of_week.csv','pedestrian_crossing_physical_facilities.csv',
                         'light_conditions.csv' , 'weather_conditions.csv', 'road_surface_conditions.csv', 'special_conditions_at_site.csv',
                          'carriageway_hazards.csv','did_police_officer_attend_scene_of_accident.csv','trunk_road_flag.csv','police_force.csv','local_authority_district.csv','local_authority_ons_district.csv','local_authority_highway.csv']
    
    for file in files_to_be_merged:
        master_df=master_df.append(pd.read_csv('/opt/airflow/data/'+file,index_col=0))
    
    master_df.to_csv('/opt/airflow/data/lookup_table.csv')
    master_df.to_sql(name = 'lookup_table.csv',con = engine,if_exists='replace')
    


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    'start_date': days_ago(2),
    "retries": 1,
}

dag = DAG(
    'accidents_etl_pipeline',
    default_args=default_args,
    description='accidents etl pipeline',
)
with DAG(
    dag_id = 'accidents_etl_pipeline',
    schedule_interval = '@once',
    default_args = default_args,
    tags = ['accident_pipeline'],
)as dag:
    preprocessing_task= PythonOperator(
        task_id = 'preprocessing',
        python_callable = preprocessing_dataset,
        op_kwargs={
            "filename": '/opt/airflow/data/2000_Accidents_UK.csv'
        },
    )
    data_integrations_m2_task= PythonOperator(
        task_id = 'data_integrations_m2',
        python_callable = data_integrations_m2,
        op_kwargs={
            "filename": "/opt/airflow/data/accidents_clean.csv"
        },
    )
    load_to_postgres_task=PythonOperator(
        task_id = 'load_to_postgres',
        python_callable = load_to_postgres,
        op_kwargs={
            "filename": "/opt/airflow/data/UK_Accidents_2000.csv"
        },
    )
    create_dashboard_task= PythonOperator(
        task_id = 'create_dashboard_task',
        python_callable = create_dashboard,
        op_kwargs={
            "filename": "/opt/airflow/data/UK_Accidents_2000.csv"
        },
    )
    

    
    preprocessing_task >> data_integrations_m2_task >> load_to_postgres_task >> create_dashboard_task

# DONT FORGET THE AUTHENTICATION FILE
    



