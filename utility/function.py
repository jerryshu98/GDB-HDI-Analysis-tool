import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import os
# Function to create cause-measure table from the given file
def Create_cause_measure_table(file_path, location_name='Sub-Saharan Africa', age_name='All ages', year_start=1990, year_end=2021, save_path = None):
    
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Filter data by 'Rate', age group, and location
    df = df[df['metric_name'] == 'Rate']
    df = df[df['age_name'] == age_name]
    df = df[df['location_name'] == location_name]
    
    # Set rounding precision
    Round_number = 100

    # Pivot the table to create 'val', 'upper', 'lower' tables
    pivot_val = df.pivot_table(index=['cause_name', 'sex_name'],
                               columns=['year', 'measure_name'],
                               values='val',
                               aggfunc='mean',
                               fill_value=0)
    
    pivot_upper = df.pivot_table(index=['cause_name', 'sex_name'],
                                 columns=['year', 'measure_name'],
                                 values='upper',
                                 aggfunc='mean',
                                 fill_value=0)
    
    pivot_lower = df.pivot_table(index=['cause_name', 'sex_name'],
                                 columns=['year', 'measure_name'],
                                 values='lower',
                                 aggfunc='mean',
                                 fill_value=0)

    # Rename the measure name 'DALYs (Disability-Adjusted Life Years)' to 'DALYs' in all pivot tables
    pivot_val.rename(columns={'DALYs (Disability-Adjusted Life Years)': 'DALYs'}, level='measure_name', inplace=True)
    pivot_upper.rename(columns={'DALYs (Disability-Adjusted Life Years)': 'DALYs'}, level='measure_name', inplace=True)
    pivot_lower.rename(columns={'DALYs (Disability-Adjusted Life Years)': 'DALYs'}, level='measure_name', inplace=True)

    # Round down the values in the pivot tables to three decimal places
    pivot_val1 = pivot_val.map(lambda x: np.floor(x * Round_number) / Round_number)
    pivot_upper1 = pivot_upper.map(lambda x: np.floor(x * Round_number) / Round_number)
    pivot_lower1 = pivot_lower.map(lambda x: np.floor(x * Round_number) / Round_number)

    # Combine val, lower, and upper into a single format 'val (lower - upper)'
    combined_pivot = pivot_val1.astype(str) + " (" + pivot_lower1.astype(str) + " - " + pivot_upper1.astype(str) + ")"

    # Extract data for the start and end years (e.g., 1990 and 2021)
    df_start_year = combined_pivot.xs(year_start, level='year', axis=1)
    df_end_year = combined_pivot.xs(year_end, level='year', axis=1)

    # Calculate percentage change from start year to end year
    change_df = (pivot_val.xs(year_end, level='year', axis=1) - pivot_val.xs(year_start, level='year', axis=1)) / pivot_val.xs(year_start, level='year', axis=1) * 100
    change_upper = (pivot_upper.xs(year_end, level='year', axis=1) - pivot_upper.xs(year_start, level='year', axis=1)) / pivot_upper.xs(year_start, level='year', axis=1) * 100
    change_lower = (pivot_lower.xs(year_end, level='year', axis=1) - pivot_lower.xs(year_start, level='year', axis=1)) / pivot_lower.xs(year_start, level='year', axis=1) * 100

    # Round down the percentage change values to three decimal places
    change_df = change_df.map(lambda x: np.floor(x * Round_number) / Round_number)
    change_upper = change_upper.map(lambda x: np.floor(x * Round_number) / Round_number)
    change_lower = change_lower.map(lambda x: np.floor(x * Round_number) / Round_number)

    # Format the percentage change as 'change% (lower% - upper%)'
    change_df_formatted = change_df.astype(str) + " (" + change_lower.astype(str) + " - " + change_upper.astype(str) + ")"

    # Combine the data for start year, end year, and percentage change
    combined_df = pd.concat([df_start_year.add_suffix(f'_{year_start}'), df_end_year.add_suffix(f'_{year_end}'), change_df_formatted.add_suffix('_change_%')], axis=1)

    # Reorder the columns by name to group similar columns together
    cols = combined_df.columns.to_list()
    sorted_cols = sorted(cols, key=lambda x: x.split('_')[0])  # Sort by the prefix before '_year' and '_change_%'
    combined_df = combined_df[sorted_cols]
    
    if save_path != None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))  # Create the directory if it does not exist
        combined_df.to_csv(save_path)
    return combined_df


def Create_std_cause_measure_table(file_path, location_name='Sub-Saharan Africa', age_name='All ages', year_start=1990, year_end=2021, save_path=None):
    
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Filter by metric, location, and age group
    df = df[df['metric_name'] == 'Rate']
    # Uncomment if you need to filter by age_name
    # df = df[df['age_name'] == age_name]
    df = df[df['location_name'] == location_name]
    
    # Set rounding precision
    Round_number = 100

    # Age standardized weights dictionary
    age_standardized_weights = {
        '0-4 years': 0.088,
        '5-9 years': 0.086,
        '10-14 years': 0.086,
        '15-19 years': 0.084,
        '20-24 years': 0.082,
        '25-29 years': 0.080,
        '30-34 years': 0.077,
        '35-39 years': 0.074,
        '40-44 years': 0.071,
        '45-49 years': 0.067,
        '50-54 years': 0.062,
        '55-59 years': 0.057,
        '60-64 years': 0.052,
        '65-69 years': 0.047,
        '70-74 years': 0.040,
        '75-79 years': 0.031,
        '80+ years': 0.020
    }
    
    # Filter rows by age group
    df_filtered = df[df['age_name'].isin(age_standardized_weights.keys())]

    # Map age weights to the dataframe
    df['age_weight'] = df['age_name'].map(age_standardized_weights)

    # Calculate weighted 'val', 'upper', 'lower'
    df['weighted_val'] = df['val'] * df['age_weight']
    df['weighted_upper'] = df['upper'] * df['age_weight']
    df['weighted_lower'] = df['lower'] * df['age_weight']

    # Pivot table for weighted values, upper, and lower bounds
    pivot_weighted_val = df.pivot_table(index=['cause_name', 'sex_name'],
                                        columns=['year', 'measure_name'],
                                        values='weighted_val',
                                        aggfunc='sum',
                                        fill_value=0)  
    pivot_weighted_upper = df.pivot_table(index=['cause_name', 'sex_name'],
                                          columns=['year', 'measure_name'],
                                          values='weighted_upper',
                                          aggfunc='sum',
                                          fill_value=0) 
    pivot_weighted_lower = df.pivot_table(index=['cause_name', 'sex_name'],
                                          columns=['year', 'measure_name'],
                                          values='weighted_lower',
                                          aggfunc='sum',
                                          fill_value=0) 

    # Rename 'DALYs (Disability-Adjusted Life Years)' to 'DALYs' in all pivot tables
    pivot_weighted_val.rename(columns={'DALYs (Disability-Adjusted Life Years)': 'DALYs'}, level='measure_name', inplace=True)
    pivot_weighted_upper.rename(columns={'DALYs (Disability-Adjusted Life Years)': 'DALYs'}, level='measure_name', inplace=True)
    pivot_weighted_lower.rename(columns={'DALYs (Disability-Adjusted Life Years)': 'DALYs'}, level='measure_name', inplace=True)

    # Round down the values to three decimal places
    pivot_val1 = pivot_weighted_val.map(lambda x: np.floor(x * Round_number) / Round_number)
    pivot_upper1 = pivot_weighted_upper.map(lambda x: np.floor(x * Round_number) / Round_number)
    pivot_lower1 = pivot_weighted_lower.map(lambda x: np.floor(x * Round_number) / Round_number)

    # Combine val, upper, and lower into 'val (lower - upper)' format
    combined_pivot = pivot_val1.astype(str) + " (" + pivot_lower1.astype(str) + " - " + pivot_upper1.astype(str) + ")"

    # Extract data for the start and end years (e.g., year_start and year_end)
    df_start_year = combined_pivot.xs(year_start, level='year', axis=1)
    df_end_year = combined_pivot.xs(year_end, level='year', axis=1)

    # Calculate percentage change from start year to end year
    change_df = (pivot_weighted_val.xs(year_end, level='year', axis=1) - pivot_weighted_val.xs(year_start, level='year', axis=1)) / pivot_weighted_val.xs(year_start, level='year', axis=1) * 100
    change_upper = (pivot_weighted_upper.xs(year_end, level='year', axis=1) - pivot_weighted_upper.xs(year_start, level='year', axis=1)) / pivot_weighted_upper.xs(year_start, level='year', axis=1) * 100
    change_lower = (pivot_weighted_lower.xs(year_end, level='year', axis=1) - pivot_weighted_lower.xs(year_start, level='year', axis=1)) / pivot_weighted_lower.xs(year_start, level='year', axis=1) * 100

    # Round down the percentage change values to three decimal places
    change_df = change_df.map(lambda x: np.floor(x * Round_number) / Round_number)
    change_upper = change_upper.map(lambda x: np.floor(x * Round_number) / Round_number)
    change_lower = change_lower.map(lambda x: np.floor(x * Round_number) / Round_number)

    # Format percentage change as 'change% (lower% - upper%)'
    change_df_formatted = change_df.astype(str) + " (" + change_lower.astype(str) + " - " + change_upper.astype(str) + ")"

    # Combine data for start year, end year, and percentage change
    combined_df = pd.concat([df_start_year.add_suffix(f'_{year_start}'), df_end_year.add_suffix(f'_{year_end}'), change_df_formatted.add_suffix('_change_%')], axis=1)

    # Reorder columns to group similar types together
    cols = combined_df.columns.to_list()
    sorted_cols = sorted(cols, key=lambda x: x.split('_')[0])  # Sort by the name before '_year' or '_change_%'
    combined_df = combined_df[sorted_cols]

    # Optionally save the resulting dataframe
    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))  # Create the directory if it does not exist
        combined_df.to_csv(save_path)

    return combined_df


def Create_std_year_measure_figure(file_path, location_name='Sub-Saharan Africa',sex_name = 'Both', metric_name='Rate'):
    df = pd.read_csv(file_path)
    df = df[df['location_name'] == location_name]
    df = df[df['sex_name'] == sex_name]
    df = df[df['metric_name'] == metric_name]
    # 假設 age_standardized_weights 是一個字典，包含年齡組和對應的權重
    age_standardized_weights = {
        '0-4 years': 0.088,
        '5-9 years': 0.086,
        '10-14 years': 0.086,
        '15-19 years': 0.084,
        '20-24 years': 0.082,
        '25-29 years': 0.080,
        '30-34 years': 0.077,
        '35-39 years': 0.074,
        '40-44 years': 0.071,
        '45-49 years': 0.067,
        '50-54 years': 0.062,
        '55-59 years': 0.057,
        '60-64 years': 0.052,
        '65-69 years': 0.047,
        '70-74 years': 0.040,
        '75-79 years': 0.031,
        '80+ years': 0.020
    }

    # 添加權重欄位
    df['age_weight'] = df['age_name'].map(age_standardized_weights)

    # 計算加權 val, upper 和 lower
    df['weighted_val'] = df['val'] * df['age_weight']
    df['weighted_upper'] = df['upper'] * df['age_weight']
    df['weighted_lower'] = df['lower'] * df['age_weight']

    # 針對每個 'cause_name', 'measure_name', 'sex_name' 和 'year' 計算年齡標準化加權總和
    age_standardized_df = df.groupby(['cause_name', 'measure_name', 'sex_name', 'year'])[['weighted_val', 'weighted_upper', 'weighted_lower']].sum().reset_index()

    # 針對每個 'cause_name' 和 'measure_name' 繪製年齡標準化的趨勢圖
    unique_causes = age_standardized_df['cause_name'].unique()
    unique_measures = age_standardized_df['measure_name'].unique()

    # 計算子圖的行數和列數
    num_rows = len(unique_causes)
    num_cols = len(unique_measures)

    # 創建子圖，讓每個Y軸獨立
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), sharex=False, sharey=False)

    # 針對每個 cause_name 和 measure_name 繪製圖表
    for i, cause in enumerate(unique_causes):
        for j, measure in enumerate(unique_measures):
            # 選擇對應的數據
            df_subset = age_standardized_df[(age_standardized_df['cause_name'] == cause) & (age_standardized_df['measure_name'] == measure)]
            
            # 獲取當前的子圖軸
            ax = axes[i, j]
            
            # 在當前子圖軸中繪製趨勢圖
            for sex in df_subset['sex_name'].unique():
                df_sex = df_subset[df_subset['sex_name'] == sex]
                
                # 繪製數據點和線
                ax.plot(df_sex['year'], df_sex['weighted_val'], label=f'{sex}')
                
                # 添加上下介區間
                ax.fill_between(df_sex['year'], df_sex['weighted_lower'], df_sex['weighted_upper'], alpha=0.2)
            
            # 設定子圖標題和標籤
            if i == 0:
                ax.set_title(f'{measure}', fontsize=14)  # 只在第一行顯示 measure_name 標題
            if j == 0:
                ax.set_ylabel(f'{cause}', fontsize=12)  # 只在第一列顯示 cause_name 標籤
            
            # 設定年份的 X 軸標籤，每隔 5 年顯示一次
            years = df_subset['year'].unique()
            ax.set_xticks(range(int(years.min()), int(years.max()) + 1, 5))
            ax.tick_params(axis='x', rotation=45)

    # 添加圖例
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', title='Sex')
    # 設定整體佈局
    plt.tight_layout()
    plt.show()
    
    
    

# Function to calculate Pearson correlation, p-value, and 95% confidence intervals
def calculate_full_statistics(figure_1_path, hdi_path, location_name='Sub-Saharan Africa', sex_name='Both', metric_name='Rate', hdi_name = 'Human Development Index (value)',save_path = None):

    df = pd.read_csv(figure_1_path)
    df = df[(df['location_name'] == location_name) & (df['sex_name'] == sex_name) & (df['metric_name'] == metric_name)]
    
    hdi_df = pd.read_excel(hdi_path)
    hdi_df = hdi_df[hdi_df['indicator'] == hdi_name]
    age_standardized_weights = {
        '0-4 years': 0.088, '5-9 years': 0.086, '10-14 years': 0.086, '15-19 years': 0.084,
        '20-24 years': 0.082, '25-29 years': 0.080, '30-34 years': 0.077, '35-39 years': 0.074,
        '40-44 years': 0.071, '45-49 years': 0.067, '50-54 years': 0.062, '55-59 years': 0.057,
        '60-64 years': 0.052, '65-69 years': 0.047, '70-74 years': 0.040, '75-79 years': 0.031,
        '80+ years': 0.020
    }
    df['age_weight'] = df['age_name'].map(age_standardized_weights)
    
    # 計算加權的 val
    df['weighted_val'] = df['val'] * df['age_weight']
    
    # 年齡標準化
    age_standardized_df = df.groupby(['cause_name', 'measure_name', 'year'])[['weighted_val']].sum().reset_index()
    
    # Merge HDI data with the age-standardized data
    merged_df = pd.merge(age_standardized_df, hdi_df[['year', 'value']], on='year', suffixes=('', '_hdi'))

    # Initialize an empty list to store the results
    results = []

    # Calculate Pearson correlation, p-value, and 95% confidence intervals for each group
    for (cause, measure), group in merged_df.groupby(['cause_name', 'measure_name']):
        pearson_corr, p_value = stats.pearsonr(group['weighted_val'], group['value'])
        n = len(group)
        
        # Fisher's r-to-z transformation for confidence intervals
        z = np.arctanh(pearson_corr)  # Fisher's z
        se = 1 / np.sqrt(n - 3)  # Standard error
        z_conf_interval = stats.norm.ppf(0.975) * se  # 95% CI in z-space
        lower_z, upper_z = z - z_conf_interval, z + z_conf_interval
        lower_corr, upper_corr = np.tanh(lower_z), np.tanh(upper_z)  # Convert back to r-space
        
        # Append the result to the list
        results.append({
            'cause_name': cause,
            'measure_name': measure,
            'Pearson_Correlation': pearson_corr,
            'p_value': p_value,
            'Lower_CI_95': lower_corr,
            'Upper_CI_95': upper_corr
        })
    
    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))  # Create the directory if it does not exist
        results_df.to_csv(save_path)
    return results_df

