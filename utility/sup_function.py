import pandas as pd
import numpy as np
import os
import geopandas as gpd
import matplotlib.pyplot as plt

def country_cause(file_path, sex_name='Both', metric_name='Rate', year_start=1990, year_end=2021, save_path=None, Country_list=None):
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Filter the dataframe by 'metric_name' and 'sex_name'
    df = df[df['metric_name'] == metric_name]
    df = df[df['sex_name'] == sex_name]

    # Rename specific locations for consistency
    df['location_name'] = df['location_name'].replace('United Republic of Tanzania', 'Tanzania')
    df['location_name'] = df['location_name'].replace('Congo', 'Republic of Congo')
    df['measure_name'] = df['measure_name'].replace('DALYs (Disability-Adjusted Life Years)', 'DALYs')
    
    # Ensure only records with 'Both' sexes are used
    df = df[df['sex_name'] == 'Both']
    
    # Filter data for specified countries
    if Country_list is not None:
        df = df[df['location_name'].isin(Country_list)]
    
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
    
    # Map age standardized weights to the dataframe
    df['age_weight'] = df['age_name'].map(age_standardized_weights)
    
    # Calculate weighted values for 'val', 'upper', and 'lower'
    df['weighted_val'] = df['val'] * df['age_weight']
    df['weighted_upper'] = df['upper'] * df['age_weight']
    df['weighted_lower'] = df['lower'] * df['age_weight']

    # Get unique cause names
    cause_names = df['cause_name'].unique()
    location_tables = {}

    for cause in cause_names:
        # Filter the dataframe for the specific cause
        cause_df = df[df['cause_name'] == cause]
        
        # Create pivot tables for weighted 'val', 'upper', and 'lower'
        pivot_weighted_val = cause_df.pivot_table(index=['location_name'],
                                                  columns=['year', 'measure_name'],
                                                  values='weighted_val',
                                                  aggfunc='sum',
                                                  fill_value=0)
        
        pivot_weighted_upper = cause_df.pivot_table(index=['location_name'],
                                                    columns=['year', 'measure_name'],
                                                    values='weighted_upper',
                                                    aggfunc='sum',
                                                    fill_value=0)
        
        pivot_weighted_lower = cause_df.pivot_table(index=['location_name'],
                                                    columns=['year', 'measure_name'],
                                                    values='weighted_lower',
                                                    aggfunc='sum',
                                                    fill_value=0)

        # Rename 'DALYs (Disability-Adjusted Life Years)' to 'DALYs'
        pivot_weighted_val.rename(columns={'DALYs (Disability-Adjusted Life Years)': 'DALYs'}, level='measure_name', inplace=True)
        pivot_weighted_upper.rename(columns={'DALYs (Disability-Adjusted Life Years)': 'DALYs'}, level='measure_name', inplace=True)
        pivot_weighted_lower.rename(columns={'DALYs (Disability-Adjusted Life Years)': 'DALYs'}, level='measure_name', inplace=True)

        # Round down to three decimal places
        pivot_val1 = pivot_weighted_val.map(lambda x: np.floor(x * Round_number) / Round_number)
        pivot_upper1 = pivot_weighted_upper.map(lambda x: np.floor(x * Round_number) / Round_number)
        pivot_lower1 = pivot_weighted_lower.map(lambda x: np.floor(x * Round_number) / Round_number)

        # Combine 'val', 'upper', and 'lower' into a single format 'val (lower - upper)'
        combined_pivot = pivot_val1.astype(str) + " (" + pivot_lower1.astype(str) + " - " + pivot_upper1.astype(str) + ")"

        # Extract data for the start and end years
        df_start_year = combined_pivot.xs(year_start, level='year', axis=1)
        df_end_year = combined_pivot.xs(year_end, level='year', axis=1)

        # Calculate percentage change between start and end years
        change_df = (pivot_weighted_val.xs(year_end, level='year', axis=1) - pivot_weighted_val.xs(year_start, level='year', axis=1)) / pivot_weighted_val.xs(year_start, level='year', axis=1) * 100
        change_upper = (pivot_weighted_upper.xs(year_end, level='year', axis=1) - pivot_weighted_upper.xs(year_start, level='year', axis=1)) / pivot_weighted_upper.xs(year_start, level='year', axis=1) * 100
        change_lower = (pivot_weighted_lower.xs(year_end, level='year', axis=1) - pivot_weighted_lower.xs(year_start, level='year', axis=1)) / pivot_weighted_lower.xs(year_start, level='year', axis=1) * 100

        # Round down the percentage change to three decimal places
        change_df = change_df.map(lambda x: np.floor(x * Round_number) / Round_number)
        change_upper = change_upper.map(lambda x: np.floor(x * Round_number) / Round_number)
        change_lower = change_lower.map(lambda x: np.floor(x * Round_number) / Round_number)

        # Format percentage change as 'change% (lower% - upper%)'
        change_df_formatted = change_df.astype(str) + " (" + change_lower.astype(str) + " - " + change_upper.astype(str) + ")"

        # Combine the data from start year, end year, and percentage change
        combined_df = pd.concat([df_start_year.add_suffix(f'_{year_start}'), df_end_year.add_suffix(f'_{year_end}'), change_df_formatted.add_suffix('_change_%')], axis=1)

        # Reorder the columns to group similar columns together
        cols = combined_df.columns.to_list()
        sorted_cols = sorted(cols, key=lambda x: x.split('_')[0])  # Sort by the prefix before '_year' and '_change_%'
        combined_df = combined_df[sorted_cols]

        # Store the result in a dictionary, with the key being the cause_name
        location_tables[cause] = combined_df
        
    # Check if save_path is provided and ensure the directory exists
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))  # Create the directory if it does not exist
        for i in range(len(cause_names)):
            location_tables[list(location_tables.keys())[i]].to_csv(f'{save_path}_{list(location_tables.keys())[i]}.csv', index=True)
    
    return location_tables


def std_age_cause(file_path, sex_name='Both', metric_name='Rate', location_name='Sub-Saharan Africa', year_start=1990, year_end=2021, save_path=None):
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Filter the dataframe by 'metric_name' and 'sex_name'
    df = df[df['metric_name'] == metric_name]
    df = df[df['sex_name'] == sex_name]
    df['location_name'] = df['location_name'].replace('United Republic of Tanzania', 'Tanzania')
    df['location_name'] = df['location_name'].replace('Congo', 'Republic of Congo')
    df['measure_name'] = df['measure_name'].replace('DALYs (Disability-Adjusted Life Years)', 'DALYs')
    df = df[df['location_name'] == location_name]

    Round_number = 100
    
    # Mapping for age groups
    age_group_mapping = {
        '0-4 years': '15-49',
        '5-9 years': '15-49',
        '10-14 years': '15-49',
        '15-19 years': '15-49',
        '20-24 years': '15-49',
        '25-29 years': '15-49',
        '30-34 years': '15-49',
        '35-39 years': '15-49',
        '40-44 years': '15-49',
        '45-49 years': '15-49',
        '50-54 years': '50-74',
        '55-59 years': '50-74',
        '60-64 years': '50-74',
        '65-69 years': '50-74',
        '70-74 years': '50-74',
        '75-79 years': '75 up',
        '80+ years': '75 up'
    }
    
    # Age standardized weights
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
    
    cause_names = df['cause_name'].unique()
    
    # Add age group based on age_name
    df['age_group'] = df['age_name'].map(age_group_mapping)

    # Perform age-standardization
    df['age_weight'] = df['age_name'].map(age_standardized_weights)

    # Calculate weighted values
    df['weighted_val'] = df['val'] * df['age_weight']
    df['weighted_upper'] = df['upper'] * df['age_weight']
    df['weighted_lower'] = df['lower'] * df['age_weight']

    # For each cause, create tables with rows as age_group, and then sum across age groups for 'all age'
    location_tables = {}

    for cause in cause_names:
        cause_df = df[df['cause_name'] == cause]
        
        # Aggregate the data by age_group (summing over all locations)
        pivot_weighted_val = cause_df.pivot_table(index=['age_group'],
                                                columns=['year', 'measure_name'],
                                                values='weighted_val',
                                                aggfunc='sum',
                                                fill_value=0)
        pivot_weighted_upper = cause_df.pivot_table(index=['age_group'],
                                                    columns=['year', 'measure_name'],
                                                    values='weighted_upper',
                                                    aggfunc='sum',
                                                    fill_value=0)
        pivot_weighted_lower = cause_df.pivot_table(index=['age_group'],
                                                    columns=['year', 'measure_name'],
                                                    values='weighted_lower',
                                                    aggfunc='sum',
                                                    fill_value=0)

        # Sum the values across all age groups for 'all age'
        all_age_val = pivot_weighted_val.sum(axis=0).to_frame().T
        all_age_val.index = ['all age']
        
        all_age_upper = pivot_weighted_upper.sum(axis=0).to_frame().T
        all_age_upper.index = ['all age']
        
        all_age_lower = pivot_weighted_lower.sum(axis=0).to_frame().T
        all_age_lower.index = ['all age']

        # Append 'all age' to the pivot tables
        pivot_weighted_val = pd.concat([pivot_weighted_val, all_age_val])
        pivot_weighted_upper = pd.concat([pivot_weighted_upper, all_age_upper])
        pivot_weighted_lower = pd.concat([pivot_weighted_lower, all_age_lower])

        # Format and clean the pivot tables
        pivot_val1 = pivot_weighted_val.map(lambda x: np.floor(x * Round_number) / Round_number)
        pivot_upper1 = pivot_weighted_upper.map(lambda x: np.floor(x * Round_number) / Round_number)
        pivot_lower1 = pivot_weighted_lower.map(lambda x: np.floor(x * Round_number) / Round_number)

        # Combine val, upper, and lower into one format
        combined_pivot = pivot_val1.astype(str) + " (" + pivot_lower1.astype(str) + " - " + pivot_upper1.astype(str) + ")"

        # Extract 1990 and 2021 data
        df_start_year = combined_pivot.xs(year_start, level='year', axis=1)
        df_end_year = combined_pivot.xs(year_end, level='year', axis=1)

        # Calculate percentage change between start and end years
        change_df = (pivot_weighted_val.xs(year_end, level='year', axis=1) - pivot_weighted_val.xs(year_start, level='year', axis=1)) / pivot_weighted_val.xs(year_start, level='year', axis=1) * 100
        change_upper = (pivot_weighted_upper.xs(year_end, level='year', axis=1) - pivot_weighted_upper.xs(year_start, level='year', axis=1)) / pivot_weighted_upper.xs(year_start, level='year', axis=1) * 100
        change_lower = (pivot_weighted_lower.xs(year_end, level='year', axis=1) - pivot_weighted_lower.xs(year_start, level='year', axis=1)) / pivot_weighted_lower.xs(year_start, level='year', axis=1) * 100

        # Round down the percentage change to three decimal places
        change_df = change_df.map(lambda x: np.floor(x * Round_number) / Round_number)
        change_upper = change_upper.map(lambda x: np.floor(x * Round_number) / Round_number)
        change_lower = change_lower.map(lambda x: np.floor(x * Round_number) / Round_number)

        # Format percentage change as 'change% (lower% - upper%)'
        change_df_formatted = change_df.astype(str) + " (" + change_lower.astype(str) + " - " + change_upper.astype(str) + ")"

        # Combine the data from start year, end year, and percentage change
        combined_df = pd.concat([df_start_year.add_suffix(f'_{year_start}'), df_end_year.add_suffix(f'_{year_end}'), change_df_formatted.add_suffix('_change_%')], axis=1)

        # Reorder the columns to group similar columns together
        cols = combined_df.columns.to_list()
        sorted_cols = sorted(cols, key=lambda x: x.split('_')[0])  # Sort by the prefix before '_year' and '_change_%'
        combined_df = combined_df[sorted_cols]

        # Store the result in a dictionary, with the key being the cause_name
        location_tables[cause] = combined_df
        
    # Check if save_path is provided and ensure the directory exists
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))  # Create the directory if it does not exist
        for i in range(len(cause_names)):
            location_tables[list(location_tables.keys())[i]].to_csv(f'{save_path}_{list(location_tables.keys())[i]}.csv', index=True)
    
    return location_tables


def std_location_measure(file_path, sex_name='Both', metric_name='Rate', year_start=1990, year_end=2021, save_path=None, Country_list=None):
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Filter the dataframe by 'metric_name' and 'sex_name'
    df = df[df['metric_name'] == metric_name]
    df = df[df['sex_name'] == sex_name]
    df['location_name'] = df['location_name'].replace('United Republic of Tanzania', 'Tanzania')
    df['location_name'] = df['location_name'].replace('Congo', 'Republic of Congo')
    df['measure_name'] = df['measure_name'].replace('DALYs (Disability-Adjusted Life Years)', 'DALYs')

    # Filter by the specified country list if provided
    if Country_list is not None:
        df = df[df['location_name'].isin(Country_list)]

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

    # Add age weight column
    df['age_weight'] = df['age_name'].map(age_standardized_weights)
    print(df['location_name'].unique())
    
    # Calculate weighted values for 'val', 'upper', and 'lower'
    df['weighted_val'] = df['val'] * df['age_weight']
    df['weighted_upper'] = df['upper'] * df['age_weight']
    df['weighted_lower'] = df['lower'] * df['age_weight']
    
    age_groups = {
        '14-49': ['15-19 years', '20-24 years', '25-29 years', '30-34 years', '35-39 years', '40-44 years', '45-49 years'],
        '50-74': ['50-54 years', '55-59 years', '60-64 years', '65-69 years', '70-74 years'],
        '75+': ['75-79 years', '80+ years']
    }

    # Function to sum up weighted values for each age group
    def sum_weighted_by_age_group(df, age_group):
        return df[df['age_name'].isin(age_groups[age_group])].groupby(['location_name', 'year', 'measure_name']).sum()

    # Dictionary to hold the final dataframes for each measure_name
    measure_tables = {}

    # Get unique measure names
    measure_names = df['measure_name'].unique()

    # Loop through each measure_name
    for measure in measure_names:
        # Filter the dataframe by measure_name
        measure_df = df[df['measure_name'] == measure]
        
        # Create a dictionary to store age group dataframes
        age_group_data = {}

        # Calculate weighted values for each age group
        for age_group in age_groups:
            age_group_data[age_group] = sum_weighted_by_age_group(measure_df, age_group)
        
        # Extract 1990 and 2021 data for each age group
        data_start = {}
        data_end = {}
        change_data = {}

        for age_group in age_groups:
            # Extract 1990 and 2021 data
            data_start[age_group] = age_group_data[age_group].xs(year_start, level='year', axis=0)['weighted_val']
            data_end[age_group] = age_group_data[age_group].xs(year_end, level='year', axis=0)['weighted_val']
            
            # Calculate percentage change between 1990 and 2021
            change_data[age_group] = (data_end[age_group] - data_start[age_group]) / data_start[age_group] * 100
        
        # Create a combined dataframe with columns for each age group (14-49, 50-74, 75+)
        combined_df = pd.DataFrame({
            '14-49_change_%': change_data['14-49'],
            '50-74_change_%': change_data['50-74'],
            '75+_change_%': change_data['75+']
        })
        
        # Save the result in the measure_tables dictionary
        measure_tables[measure] = combined_df
        
    # Check if save_path is provided and ensure the directory exists
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))  # Create the directory if it does not exist
        for i in range(len(measure_names)):
            measure_tables[list(measure_tables.keys())[i]].to_csv(f'{save_path}_{list(measure_tables.keys())[i]}.csv', index=True)
    
    return measure_tables


def plot_heap(file_path, shape_file_path, num):
    # Load the data from CSV and shapefile
    df = pd.read_csv(file_path)
    gdf = gpd.read_file(shape_file_path)
    
    # Replace specific country names for consistency
    df['location_name'] = df['location_name'].replace('United Republic of Tanzania', 'Tanzania')
    df['location_name'] = df['location_name'].replace('Congo', 'Republic of Congo')
    df['measure_name'] = df['measure_name'].replace('DALYs (Disability-Adjusted Life Years)', 'DALYs')

    # Filter the dataframe to include only Sub-Saharan African countries
    sub_saharan_countries = ['Angola', 'Burundi', 'Gabon', 'Republic of Congo', 'Central African Republic',
                            'Equatorial Guinea', 'Kenya', 'Djibouti', 'Malawi', 'Democratic Republic of the Congo',
                            'Ethiopia', 'Comoros', 'Eritrea', 'Mozambique', 'Uganda', 'Madagascar', 'Eswatini',
                            'Rwanda', 'Tanzania', 'Zambia', 'Lesotho', 'Somalia', 'Botswana', 'Burkina Faso', 'Gambia',
                            'Namibia', 'Zimbabwe', 'Benin', 'Cabo Verde', 'Guinea', 'Cameroon', "Côte d'Ivoire", 'Ghana',
                            'Niger', 'Chad', 'Liberia', 'Sao Tome and Principe', 'Mauritania', 'Guinea-Bissau',
                            'Sierra Leone', 'Mali', 'Senegal', 'Nigeria', 'Togo', 'South Sudan', 'South Africa']

    df = df[df['location_name'].isin(sub_saharan_countries)]

    # Rename the column in the shapefile to match the dataframe
    gdf = gdf.rename(columns={'NAME_0': 'location_name'})

    # Function to create age groups based on age_name
    def create_age_group(age_name):
        if age_name in ['15-19 years', '20-24 years', '25-29 years', '30-34 years', '35-39 years', '40-44 years', '45-49 years']:
            return '15-49'
        elif age_name in ['50-54 years', '55-59 years', '60-64 years', '65-69 years', '70-74 years']:
            return '50-74'
        elif age_name in ['75-79 years', '80+ years']:
            return '75+'
        else:
            return None  # For any age groups outside the desired ones

    # Apply the function to create a new 'age_group' column
    df['age_group'] = df['age_name'].apply(create_age_group)

    # Filter the DataFrame to only include the relevant age groups
    df = df[df['age_group'].notna()]

    age_groups = ['15-49', '50-74', '75+']
    df = df[df['age_group'].isin(age_groups)]

    # Pick the first cause for further analysis
    cause_to_plot = df['cause_name'].unique()[num]

    # Filter the dataframe for the selected cause
    df = df[df['cause_name'] == cause_to_plot]

    # Create pivot tables for each measure_name and each age group
    pivot_tables = {}
    for age_group in age_groups:
        for measure in df['measure_name'].unique():
            pivot_df = df[(df['age_group'] == age_group) & (df['measure_name'] == measure)].pivot_table(
                index='location_name', columns='year', values='val', aggfunc='sum', fill_value=0)

            # Calculate percentage change between 1990 and 2021
            if 1990 in pivot_df.columns and 2021 in pivot_df.columns:
                pivot_df['change_percent'] = (pivot_df[2021] - pivot_df[1990]) / pivot_df[1990] * 100
            
            pivot_tables[(measure, age_group)] = pivot_df

    # Function to plot maps from pivot tables
    def plot_maps_pivot_tables(pivot_tables, gdf, cause_name):
        """
        Plots the data from pivot_tables as maps in a grid format, 3 columns per row.
        Each map shows the incidence change for different age groups and measures.
    
        Parameters:
        pivot_tables (dict): Dictionary where keys are (measure_name, age_group) and values are pivot tables
        gdf (GeoDataFrame): GeoDataFrame containing the geographic shape data
        cause_name (str): The name of the cause to include in the title of each plot
        """
        num_plots = len(pivot_tables)
        num_cols = 3
        num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate number of rows needed
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 6 * num_rows))
        axes = axes.flatten()  # Flatten axes to easily access in a loop
        
        for idx, ((measure_name, age_group), pivot_df) in enumerate(pivot_tables.items()):
            ax = axes[idx]
            
            # Reset index to merge with geographic data
            change_data = pivot_df.reset_index()
            
            # Merge geographic data with the change data
            merged = gdf.set_index('location_name').join(change_data.set_index('location_name'), how='inner')
            
            # Check if 'change_percent' exists in the pivot table
            if 'change_percent' in merged.columns:
                # Plot the map for each measure and age group combination
                merged.plot(column='change_percent', cmap='coolwarm', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
                ax.set_title(f'{measure_name} Change for {cause_name} ({age_group}) (1990-2021)', fontsize=10)
        
        # Hide any remaining empty subplots
        for i in range(idx + 1, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.show()

    plot_maps_pivot_tables(pivot_tables, gdf, cause_to_plot)
