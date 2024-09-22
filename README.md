# README

## Overview
This repository contains Python scripts that utilize various libraries for data analysis, visualization, and geographic data processing. 

## Required Packages
To run the scripts, you need to install the following Python packages:

- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical computing.
- `matplotlib`: For data visualization.
- `scipy`: For scientific and technical computing.
- `geopandas`: For working with geospatial data.

## Installation Instructions

### Using pip
You can install the required packages using `pip`. Open your terminal or command prompt and run:

```bash
pip install pandas numpy matplotlib scipy geopandas
```

### Using conda
If you are using Anaconda, you can create a new environment and install the packages with:

```bash
conda create -n myenv python=3.8
conda activate myenv
conda install pandas numpy matplotlib scipy geopandas
```

Replace `myenv` with your desired environment name.

## Usage in Jupyter Notebook
To use the code in Jupyter Notebook, follow these steps:

1. Ensure you have Jupyter Notebook installed. If not, install it using:

   ```bash
   pip install notebook
   ```

2. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

3. Create a new notebook or open an existing one, and import the necessary libraries at the beginning of your notebook:

   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import scipy.stats as stats
   import geopandas as gpd
   import os
   ```

4. You can now run the provided code snippets and begin your data analysis.

## Example Usage
This repository includes several Jupyter Notebooks and a Python script to facilitate data processing and visualization.

### Jupyter Notebooks
- **`etable_country.ipynb`**: This notebook generates tables based on country data and visualizes them.
- **`etable_risk.ipynb`**: This notebook focuses on risk assessment and provides adjustable charts for analysis.
- **`GDB_HDI_basic.ipynb`**: This notebook creates visualizations related to Human Development Index (HDI) data.

### Python Script
- **`Generate_all_csv.py`**: This script quickly generates all necessary CSV files for further analysis.

### How to Execute
1. **Run the Python Script**:
   Open your terminal or command prompt, navigate to the directory containing `Generate_all_csv.py`, and execute:

   ```bash
   python Generate_all_csv.py
   ```

2. **Run the Jupyter Notebooks**:
   After generating the CSV files, launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

   Then open the desired notebook (e.g., `etable_country.ipynb`, `etable_risk.ipynb`, or `GDB_HDI_basic.ipynb`) and run the cells to generate the tables and visualizations.

3. **Adjust Charts**:
   In the notebooks, you can modify parameters and settings to customize the charts as needed.
