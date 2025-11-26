"""
Page 2: Presentation of the data (volume, architecture, etc.)
Data analysis using DataVisualization figures
Description and justification of the pre-processing carried out
Todo by Kiki
"""
import numpy as np
import pandas as pd
# %matplotlib inline
import matplotlib.pyplot as plt

#import data
#MIT
df_mitbih_train = pd.read_csv('data/mitbih_train.csv', header = None)
df_mitbih_test = pd.read_csv('data/mitbih_test.csv', header = None)
#PTB
df_ptbdb_normal = pd.read_csv('data/ptbdb_normal.csv', header = None)
df_ptbdb_abnormal = pd.read_csv('data/ptbdb_abnormal.csv', header = None)
#combine MIT train and test sets
df_mitbih = pd.concat([df_mitbih_train, df_mitbih_test], axis=0, ignore_index=True)
#combine PTB normal and abnormal sets
df_ptbdb = pd.concat([df_ptbdb_normal, df_ptbdb_abnormal], axis=0, ignore_index=True)

#divide combined MIT dataframe in one dataframe for each class
#plot version: drop last column (because it contains class annotations and no ECG data)
df_mitbih_0 = df_mitbih.loc[df_mitbih[187]==0]
df_mitbih_0_plot = df_mitbih_0.drop(187, axis=1) 

df_mitbih_1 = df_mitbih.loc[df_mitbih[187]==1]
df_mitbih_1_plot = df_mitbih_1.drop(187, axis=1) 

df_mitbih_2 = df_mitbih.loc[df_mitbih[187]==2]
df_mitbih_2_plot = df_mitbih_2.drop(187, axis=1) 

df_mitbih_3 = df_mitbih.loc[df_mitbih[187]==3]
df_mitbih_3_plot = df_mitbih_3.drop(187, axis=1) 

df_mitbih_4 = df_mitbih.loc[df_mitbih[187]==4]
df_mitbih_4_plot = df_mitbih_4.drop(187, axis=1) 

#divide combined PTB dataframe in one dataframe for each class
#plot version: drop last column (because it contains class annotations and no ECG data)
df_ptbdb_0 = df_ptbdb.loc[df_ptbdb[187]==0]
df_ptbdb_0_plot = df_ptbdb_0.drop(187, axis=1) 

df_ptbdb_1 = df_ptbdb.loc[df_ptbdb[187]==1]
df_ptbdb_1_plot = df_ptbdb_1.drop(187, axis=1) 

import streamlit as st


def render():
    st.title("Data Overview")
    st.markdown("---")
    
    st.header("Dataset Information")
  
    st.markdown("""
      - **Problem Type:** Supervised Classification (labeled data) 
      - **Input Data:** Preprocessed ECG signals. Each sample represents a single heartbeat (centered R-peak).
      - **Classification problem:** Arrhythmia 5 classes, MI 2 classes
      - **Structure:** 188 columns per row.
        - Columns 0-186: 187 **time-series points** representing the ECG wave over time, approximately **1.2 heartbeats**.
        - Column 187: The **target label** column (Class ID) 
    """)
    st.write('MIT-BIH dataframes overview:')
    st.dataframe(df_mitbih.head())
    st.write('PTB dataframes overview:')
    st.dataframe(df_ptbdb_normal.head())

    st.divider()

    st.header("MIT-BIH Dataset")
    
    st.markdown("""
      - ECG recordings from 47 subjects 
    - **Classes:** 5 Categories
        - 0 - Normal (N)
        - 1 - Supraventricular/Atrial premature (S)
        - 2 - Premature ventricular contraction (V)
        - 3 - Fusion of ventricular and normal beat (F)
        - 4 - Unclassifiable / fusion of paced and normal (Q)
    - **Dataset Properties**
        - **109,446** heartbeat samples
        -	Numerical, normalized and preprocessed
        -	No missing values
        -	No duplicates
        -	Extremely **imbalanced class distribution**  
            - **0: 82.8%** 
            -	1: 2.5% 
            -	2: 6.6% 
            -	**3: 0.7%**
            -	4: 7.3% 
    -	**Key Challenge: Severe Class Imbalance**
    	  - Action required: Data augmentation (SMOTE) is necessary to prevent model bias.
    """)
    
    st.image("images/MIT_combined.png")

    mit_class_to_df = {
      0: df_mitbih_0_plot,
      1: df_mitbih_1_plot,
      2: df_mitbih_2_plot,
      3: df_mitbih_3_plot,
      4: df_mitbih_4_plot
    }
    st.subheader("MIT-BIH – Example ECG signals per class")
    selected_classes = st.multiselect("Select MIT-BIH classes to display", options=[0, 1, 2, 3, 4], default=[0])

    time_points = df_mitbih_0_plot.columns  # 0~186 time points, every class has same time points
    
    for cls in selected_classes:
        df_cls = mit_class_to_df[cls]
        
        examples = df_cls.sample(n=3, random_state=42)

        st.markdown(f"### MIT-BIH – Class {cls}")
        
        cols = st.columns(3)

        for i in range(3):
          row = examples.iloc[i]
          col = cols[i]   # Example i will be place in column i

          with col:
              fig, ax = plt.subplots()
              ax.plot(time_points, row.values)

              ax.set_title(f"Example {i+1}")
              ax.set_xlabel("Time point")
              ax.set_ylabel("Amplitude")

              st.pyplot(fig)

    st.divider()

    st.header("PTB Dataset")
    
    st.markdown("""
      - ECG recordings from 290 subjects
        - 148 Myocardial Infarction (MI) patients
        - 53 healthy controls
        - 90 patients diagnosed with other cardiac diseases
      - **Classes:** 2 Categories
        - 0 - Normal
        - 1 - Abnormal / Myocardial Infarction (MI)
      - **Dataset Properties**
        - **14,552** samples total
        - After **removing 7 duplicates** (6 in abnormal dataset, 1 in normal dataset) → **14,545** samples
        - Numerical, normalized, preprocessed
        - No missing values
        - Strong **imbalanced class distribution**
          - Normal: 27.8%
          - **MI: 72.2%**
      - **Key Challenge: Imbalance**
        - The "Normal" class is the minority, which requires careful handling during training.
    """)
    
    st.image("images/PTB_combined.png")

    ptb_class_to_df = {
      0: df_ptbdb_0_plot,
      1: df_ptbdb_1_plot  
    }
    st.subheader("PTB – Example ECG signals per class")
    selected_classes_ptb = st.multiselect("Select PTB classes to display", options=[0, 1], default=[0]) 
    time_points_ptb = df_ptbdb_0_plot.columns  # 0~186 time points, every class has same time points
    
    for cls in selected_classes_ptb:
        df_cls = mit_class_to_df[cls]
        
        examples = df_cls.sample(n=3, random_state=42)

        st.markdown(f"### PTB – Class {cls}")
        
        cols = st.columns(3)

        for i in range(3):
          row = examples.iloc[i]
          col = cols[i]   # Example i will be place in column i

          with col:
              fig, ax = plt.subplots()
              ax.plot(time_points, row.values)

              ax.set_title(f"Example {i+1}")
              ax.set_xlabel("Time point")
              ax.set_ylabel("Amplitude")

              st.pyplot(fig)
    

