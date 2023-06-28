#!/usr/bin/env python
# coding: utf-8


# In[18]:


import pandas as pd
import re
import numpy as np
import seaborn as sns
import os
from datetime import datetime

path='output_all_students_Train_v10.xlsx'

def prepare_data(path):
    madlan_df = pd.read_excel(path)
    
    #הורדת שורות בהן אין מחיר
    madlan_df.dropna(subset = ['price'], inplace = True)
    madlan_df['price'].isnull().sum()
    
    #הפיכת עמודות מחיר ושטח למספרים בלבד
    madlan_df["price"] = madlan_df["price"].astype(str)  
    madlan_df["price"] = madlan_df["price"].apply(lambda x: re.sub(r'\D', '', x) if x else '') 
    madlan_df["price"] = pd.to_numeric(madlan_df["price"])
    madlan_df.dropna(subset = ['price'], inplace = True)

    madlan_df["Area"] = madlan_df["Area"].astype(str)  
    madlan_df["Area"] = madlan_df["Area"].apply(lambda x: re.sub(r'\D', '', x) if x else '') 
    madlan_df["Area"] = pd.to_numeric(madlan_df["Area"])
    
    #הורדת פסיקים או סימני פיסוק מיותרים מהטקסטים
    text_columns = ['Street', 'city_area','description ']
    madlan_df[text_columns] = madlan_df[text_columns].astype(str)  
    madlan_df[text_columns] = madlan_df[text_columns].apply(lambda x: x.str.replace(r'[^\w\s]', ''))
    
    #הוספת עמודות קומה , ומספר הקומות הקיימות
    madlan_df['floor'] = madlan_df['floor_out_of'].str.extract(r'קומה\s(\d+)')
    madlan_df['floor'] = madlan_df['floor'].fillna(0).astype(int)

    madlan_df['total_floors'] = madlan_df['floor_out_of'].str.extract(r'מתוך\s(\d+)')
    madlan_df['total_floors'] = madlan_df['total_floors'].fillna(0).astype(int)
    
    #עמודת entrance_date עדכנית
    madlan_df['entranceDate '] = madlan_df['entranceDate '].replace('גמיש', 'flexible')
    madlan_df['entranceDate '] = madlan_df['entranceDate '].replace('לא צויין', 'not_defined')
    madlan_df['entranceDate '] = madlan_df['entranceDate '].replace('מיידי', 'Less_than_6_months')

    valid_dates_mask = pd.to_datetime(madlan_df['entranceDate '], errors='coerce').notna()
    current_date = pd.to_datetime(datetime.now().date())
    madlan_df.loc[valid_dates_mask, 'time_difference'] = (current_date - pd.to_datetime(madlan_df.loc[valid_dates_mask, 'entranceDate '])).dt.days / 30
    bins = [-float('inf'), 6, 12, float('inf')]
    labels = ['Less_than_6_months', 'months_6_12', 'Above_year']
    madlan_df.loc[valid_dates_mask, 'entranceDate '] = pd.cut(madlan_df.loc[valid_dates_mask, 'time_difference'], bins=bins, labels=labels)
    madlan_df['entranceDate '] = madlan_df['entranceDate '].fillna('invalid_value')
    madlan_df = madlan_df.drop(['time_difference'], axis=1)
    
    #ייצוג כל השדות הבולאינים כ0 או 1
    boolean_columns = ['hasElevator ', 'hasParking ', 'hasBars ', 'hasStorage ', 'hasAirCondition ', 'hasBalcony ', 'hasMamad ', 'handicapFriendly ']
    madlan_df[boolean_columns].fillna(0, inplace = False )
    madlan_df[boolean_columns] = madlan_df[boolean_columns].astype(str)
    replacement_dict = {'יש': 1, 'יש ממ״ד': 1, 'יש מרפסת': 1, 'יש מיזוג אוויר': 1,'יש מיזוג אויר': 1, 'נגיש לנכים': 1,
                        'נגיש': 1,"לא נגיש":0 ,'yes': 1, 'TRUE': 1, 'True': 1, 'יש מחסן': 1, 'יש סורגים': 1,
                        'יש חנייה': 1,'יש חניה': 1, 'יש מעלית': 1, 'אין': 0, 'לא': 0, 'אין חניה': 0,
                        'אין ממ״ד': 0, 'אין מרפסת': 0, 'אין מחסן': 0, 'אין סורגים': 0,
                        'אין מעלית': 0, 'אין מיזוג אויר': 0, 'לא נגיש לנכים': 0, 'no': 0,'לא':0,
                        'FALSE': 0, 'False': 0,'כן':1, 'יש ממ״ד':1, 'יש ממ"ד':1, 'אין ממ"ד':0,"nan":0}
    madlan_df[boolean_columns] = madlan_df[boolean_columns].replace(replacement_dict)
    
    #הכנת עמודת מספר חדרים
    madlan_df["room_number"]=madlan_df["room_number"].apply(lambda x: str(x))
    madlan_df["room_number"]=madlan_df["room_number"].apply(lambda x: re.sub(r"[^0-9.]", "", x))
    madlan_df["room_number"]=madlan_df["room_number"].apply(lambda x: float(x)  if x != '' else None)
    madlan_df["floor"]=madlan_df["room_number"].astype(float)
    
    #עמודות שיכולות להיות רלוונטיות למודל ומילוי ערכים חסרים 
    new_data=madlan_df[["City","Area",'city_area',"type","room_number","furniture ","condition ","entranceDate ","hasElevator ", 'hasParking ', 'hasStorage ', 'hasAirCondition ', 'hasBalcony ', 'hasMamad ', 'handicapFriendly ',"price"]]
    new_data['Area'] = new_data['Area'].fillna(new_data['Area'].mean())
    new_data['room_number'] = new_data['room_number'].fillna(new_data['room_number'].mean())
    
    #ONE HOT ENCODING לעמודות למודל
    col_encode= ["City"]
    encoded_df= pd.get_dummies(new_data[col_encode],prefix=col_encode, prefix_sep='_', dtype=int,dummy_na=True)
    
    #העמודות הנוספות שבחרנו למודל 
    encoded_df["Area"]=new_data["Area"].values
    encoded_df["hasBalcony "]=new_data["hasBalcony "].values
    encoded_df['hasMamad ']=new_data['hasMamad '].values
    encoded_df["hasElevator "]=new_data["hasElevator "].values
    #encoded_df["room_number"]=new_data["room_number"].values
    encoded_df.columns
    
    #X וY למודל 
    y=new_data["price"].values
    x=encoded_df.values
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=13)
    
    #החזרת נתוני הדאטה למודל והטסט והטריין
    return encoded_df,x,y


encoded_df,x,y=prepare_data(path)


# In[ ]:





