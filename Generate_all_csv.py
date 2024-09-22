from utility.function import *
from utility.sup_function import *
from utility.sup_function2 import *


file_path = './data/tabel_1_data_rate.csv'
Create_cause_measure_table(file_path, year_start=1990, year_end=2021,save_path='./results/basic/table1.csv')

file_path = './data/table_2_data_rate.csv'
Create_std_cause_measure_table(file_path, year_start=1990, year_end=2021,save_path='./results/basic/table2.csv')

figure_1_path = './data/figure_1.csv'
hdi_path = './data/Sub-Saharan_Africa_hdr-data.xlsx'
calculate_full_statistics(figure_1_path, hdi_path,save_path='./results/basic/GDB_HDI.csv')

# Filter the dataframe to include only Sub-Saharan African countries
sub_saharan_countries = ['Angola', 'Burundi', 'Gabon', 'Republic of Congo', 'Central African Republic',
                             'Equatorial Guinea', 'Kenya', 'Djibouti', 'Malawi', 'Democratic Republic of the Congo',
                             'Ethiopia', 'Comoros', 'Eritrea', 'Mozambique', 'Uganda', 'Madagascar', 'Eswatini',
                             'Rwanda', 'Tanzania', 'Zambia', 'Lesotho', 'Somalia', 'Botswana', 'Burkina Faso', 'Gambia',
                             'Namibia', 'Zimbabwe', 'Benin', 'Cabo Verde', 'Guinea', 'Cameroon', "Côte d'Ivoire", 'Ghana',
                             'Niger', 'Chad', 'Liberia', 'Sao Tome and Principe', 'Mauritania', 'Guinea-Bissau',
                             'Sierra Leone', 'Mali', 'Senegal', 'Nigeria', 'Togo', 'South Sudan', 'South Africa']

file_path = ('./data/all_country.csv')
tables= country_cause(file_path, Country_list= sub_saharan_countries,save_path='./results/etable/Country_to_measure_')

file_path = './data/table_2_data_rate.csv'
location_tables = std_age_cause(file_path,save_path='./results/etable/age_group_to_measure_')

file_path = ('./data/all_country.csv')
measure_tables = std_location_measure(file_path, Country_list= sub_saharan_countries,save_path='./results/etable/counrty_to_age_group_')

std_risk_df = std_risk('./data/risk_factor/risk_all.csv', save_path='results/risk/std_risk_')

crude_risk_df= crude_risk('./data/risk_factor/risk_all.csv', save_path='results/risk/crude_risk_')