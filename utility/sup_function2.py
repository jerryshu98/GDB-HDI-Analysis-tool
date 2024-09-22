import pandas as pd
import numpy as np
import os  # Import os for path operations

def std_risk(file_path, location_name='Sub-Saharan Africa', age_name='All ages', year_start=1990, year_end=2021, save_path=None):
    df = pd.read_csv(file_path)
    # df = df[df['location_name'] == location_name]  # Filter by location if needed
    # df = df[df['age_name'] == age_name]  # Filter by age if needed
    Round_number = 100
    print(df['location_name'].unique())
    
    # Assume age_weights is a dictionary containing age groups and corresponding weights
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

    # Add age weight column
    df['age_weight'] = df['age_name'].map(age_standardized_weights)

    # Calculate weighted val, upper, and lower
    df['weighted_val'] = df['val'] * df['age_weight']
    df['weighted_upper'] = df['upper'] * df['age_weight']
    df['weighted_lower'] = df['lower'] * df['age_weight']

    # Generate tables for each cause_name
    cause_names = df['cause_name'].unique()
    location_tables = {}
    print(cause_names)
    
    for cause in cause_names:
        # Select data corresponding to the cause_name
        cause_df = df[df['cause_name'] == cause]
        
        # Generate weighted pivot tables, index by rei_name
        pivot_weighted_val = cause_df.pivot_table(index=['rei_name'],
                                                columns=['year', 'measure_name'],
                                                values='weighted_val',
                                                aggfunc='sum',
                                                fill_value=0)
        pivot_weighted_upper = cause_df.pivot_table(index=['rei_name'],
                                                    columns=['year', 'measure_name'],
                                                    values='weighted_upper',
                                                    aggfunc='sum',
                                                    fill_value=0)
        pivot_weighted_lower = cause_df.pivot_table(index=['rei_name'],
                                                    columns=['year', 'measure_name'],
                                                    values='weighted_lower',
                                                    aggfunc='sum',
                                                    fill_value=0)

        # Rename 'DALYs (Disability-Adjusted Life Years)' to 'DALYs' in measure_name level
        pivot_weighted_val.rename(columns={'DALYs (Disability-Adjusted Life Years)': 'DALYs'}, level='measure_name', inplace=True)
        pivot_weighted_upper.rename(columns={'DALYs (Disability-Adjusted Life Years)': 'DALYs'}, level='measure_name', inplace=True)
        pivot_weighted_lower.rename(columns={'DALYs (Disability-Adjusted Life Years)': 'DALYs'}, level='measure_name', inplace=True)

        # Floor values to three decimal places
        pivot_val1 = pivot_weighted_val.map(lambda x: np.floor(x * Round_number) / Round_number)
        pivot_upper1 = pivot_weighted_upper.map(lambda x: np.floor(x * Round_number) / Round_number)
        pivot_lower1 = pivot_weighted_lower.map(lambda x: np.floor(x * Round_number) / Round_number)

        # Combine val, upper, and lower into a new format 'val (lower - upper)'
        combined_pivot = pivot_val1.astype(str) + " (" + pivot_lower1.astype(str) + " - " + pivot_upper1.astype(str) + ")"

        # Extract data for 1990 and 2021
        df_1990 = combined_pivot.xs(year_start, level='year', axis=1)
        df_2021 = combined_pivot.xs(year_end, level='year', axis=1)

        # Calculate percentage change
        change_df = (pivot_weighted_val.xs(year_end, level='year', axis=1) - pivot_weighted_val.xs(year_start, level='year', axis=1)) / pivot_weighted_val.xs(year_start, level='year', axis=1) * 100
        change_upper = (pivot_weighted_upper.xs(year_end, level='year', axis=1) - pivot_weighted_upper.xs(year_start, level='year', axis=1)) / pivot_weighted_upper.xs(year_start, level='year', axis=1) * 100
        change_lower = (pivot_weighted_lower.xs(year_end, level='year', axis=1) - pivot_weighted_lower.xs(year_start, level='year', axis=1)) / pivot_weighted_lower.xs(year_start, level='year', axis=1) * 100

        # Floor percentage change values to three decimal places
        change_df = change_df.map(lambda x: np.floor(x * Round_number) / Round_number)
        change_upper = change_upper.map(lambda x: np.floor(x * Round_number) / Round_number)
        change_lower = change_lower.map(lambda x: np.floor(x * Round_number) / Round_number)

        # Format percentage change as 'change% (lower% - upper%)'
        change_df_formatted = change_df.astype(str) + " (" + change_lower.astype(str) + " - " + change_upper.astype(str) + ")"

        # Combine data for 1990, 2021, and change
        combined_df = pd.concat([df_1990.add_suffix(f'_{year_start}'), df_2021.add_suffix(f'_{year_end}'), change_df_formatted.add_suffix('_change_%')], axis=1)

        # Reorder columns, grouping similar types together
        cols = combined_df.columns.to_list()
        sorted_cols = sorted(cols, key=lambda x: x.split('_')[0])  # Sort based on names before '_1990', '_2021', '_change_%'
        combined_df = combined_df[sorted_cols]
        print(cause)
        
        # Save the result in the dictionary with key as cause_name
        location_tables[cause] = combined_df
    
    # Check if save_path is provided and ensure the directory exists
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))  # Create the directory if it does not exist
        for i in range(len(cause_names)):
            location_tables[list(location_tables.keys())[i]].to_csv(f'{save_path}_{list(location_tables.keys())[i]}.csv', index=True)
    
    return location_tables
    # Display an example table
    # location_tables[list(location_tables.keys())[0]]


def crude_risk(file_path, location_name='Sub-Saharan Africa', age_name='All ages', year_start=1990, year_end=2021, save_path=None):
    df = pd.read_csv(file_path)
    # df = df[df['location_name'] == location_name]  # Filter by location if needed
    # df = df[df['age_name'] == age_name]  # Filter by age if needed
    Round_number = 100

    # Assume age_weights is a dictionary containing age groups and corresponding weights
    age_standardized_weights = {
        'All ages': 1,
    }

    # Add age weight column
    df['age_weight'] = df['age_name'].map(age_standardized_weights)

    # Calculate weighted val, upper, and lower
    df['weighted_val'] = df['val'] * df['age_weight']
    df['weighted_upper'] = df['upper'] * df['age_weight']
    df['weighted_lower'] = df['lower'] * df['age_weight']

    # Generate tables for each cause_name
    cause_names = df['cause_name'].unique()
    location_tables = {}

    for cause in cause_names:
        # Select data corresponding to the cause_name
        cause_df = df[df['cause_name'] == cause]
        
        # Generate weighted pivot tables, index by rei_name
        pivot_weighted_val = cause_df.pivot_table(index=['rei_name'],
                                                columns=['year', 'measure_name'],
                                                values='weighted_val',
                                                aggfunc='sum',
                                                fill_value=0)
        pivot_weighted_upper = cause_df.pivot_table(index=['rei_name'],
                                                    columns=['year', 'measure_name'],
                                                    values='weighted_upper',
                                                    aggfunc='sum',
                                                    fill_value=0)
        pivot_weighted_lower = cause_df.pivot_table(index=['rei_name'],
                                                    columns=['year', 'measure_name'],
                                                    values='weighted_lower',
                                                    aggfunc='sum',
                                                    fill_value=0)

        # Rename 'DALYs (Disability-Adjusted Life Years)' to 'DALYs' in measure_name level
        pivot_weighted_val.rename(columns={'DALYs (Disability-Adjusted Life Years)': 'DALYs'}, level='measure_name', inplace=True)
        pivot_weighted_upper.rename(columns={'DALYs (Disability-Adjusted Life Years)': 'DALYs'}, level='measure_name', inplace=True)
        pivot_weighted_lower.rename(columns={'DALYs (Disability-Adjusted Life Years)': 'DALYs'}, level='measure_name', inplace=True)

        # Floor values to three decimal places
        pivot_val1 = pivot_weighted_val.map(lambda x: np.floor(x * Round_number) / Round_number)
        pivot_upper1 = pivot_weighted_upper.map(lambda x: np.floor(x * Round_number) / Round_number)
        pivot_lower1 = pivot_weighted_lower.map(lambda x: np.floor(x * Round_number) / Round_number)

        # Combine val, upper, and lower into a new format 'val (lower - upper)'
        combined_pivot = pivot_val1.astype(str) + " (" + pivot_lower1.astype(str) + " - " + pivot_upper1.astype(str) + ")"

        # Extract data for 1990 and 2021
        df_1990 = combined_pivot.xs(year_start, level='year', axis=1)
        df_2021 = combined_pivot.xs(year_end, level='year', axis=1)

        # Calculate percentage change
        change_df = (pivot_weighted_val.xs(year_end, level='year', axis=1) - pivot_weighted_val.xs(year_start, level='year', axis=1)) / pivot_weighted_val.xs(year_start, level='year', axis=1) * 100
        change_upper = (pivot_weighted_upper.xs(year_end, level='year', axis=1) - pivot_weighted_upper.xs(year_start, level='year', axis=1)) / pivot_weighted_upper.xs(year_start, level='year', axis=1) * 100
        change_lower = (pivot_weighted_lower.xs(year_end, level='year', axis=1) - pivot_weighted_lower.xs(year_start, level='year', axis=1)) / pivot_weighted_lower.xs(year_start, level='year', axis=1) * 100

        # Floor percentage change values to three decimal places
        change_df = change_df.map(lambda x: np.floor(x * Round_number) / Round_number)
        change_upper = change_upper.map(lambda x: np.floor(x * Round_number) / Round_number)
        change_lower = change_lower.map(lambda x: np.floor(x * Round_number) / Round_number)

        # Format percentage change as 'change% (lower% - upper%)'
        change_df_formatted = change_df.astype(str) + " (" + change_lower.astype(str) + " - " + change_upper.astype(str) + ")"

        # Combine data for 1990, 2021, and change
        combined_df = pd.concat([df_1990.add_suffix(f'_{year_start}'), df_2021.add_suffix(f'_{year_end}'), change_df_formatted.add_suffix('_change_%')], axis=1)

        # Reorder columns, grouping similar types together
        cols = combined_df.columns.to_list()
        sorted_cols = sorted(cols, key=lambda x: x.split('_')[0])  # Sort based on names before '_1990', '_2021', '_change_%'
        combined_df = combined_df[sorted_cols]
        print(cause)
        
        # Save the result in the dictionary with key as cause_name
        location_tables[cause] = combined_df
    
    # Check if save_path is provided and ensure the directory exists
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))  # Create the directory if it does not exist
        for i in range(len(cause_names)):
            location_tables[list(location_tables.keys())[i]].to_csv(f'{save_path}_{list(location_tables.keys())[i]}.csv', index=True)
    
    return location_tables
