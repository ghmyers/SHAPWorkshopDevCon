import os
import logging
import sys
from pathlib import Path

# Project directory structure
DATA_DIR = Path(Path.cwd().parent, 'data')
USGS_DATA_DIR = Path(DATA_DIR, 'raw', 'usgs_streamflow')
FORCING_DATA_DIR = Path(DATA_DIR, 'raw', 'basin_mean_forcing', 'daymet')
FORCING_DATA_DIR = Path(FORCING_DATA_DIR)
STATIC_DATA_DIR = Path(DATA_DIR, 'raw', 'basin_metadata')
FIGURE_DIR = Path(Path.cwd().parent, 'outputs', 'figures')
MODEL_DIR = Path(Path.cwd().parent, 'models')
os.chdir(Path.cwd().parent)


import pandas as pd

try:
    master_df = pd.read_csv(Path(DATA_DIR, 'processed', 'CAMELS_daymet.csv'), index_col=0)
    master_df.index = pd.to_datetime(master_df.index)
except:
    print("Preprocessed data not found. Running preprocessing script.")
    # Create dictionary for storing data
    data_dict = {}

    # Define column names
    columns = ['STAID', 'YEAR', 'MONTH', 'DAY', 'Q', 'QAQC']
    # Read in all data
    for huc in os.listdir(USGS_DATA_DIR):
        for basin_data in os.listdir(Path(USGS_DATA_DIR, huc)):
            # Read in fixed width .txt file as a dataframe
            basin_df = pd.read_fwf(Path(USGS_DATA_DIR, huc, basin_data), 
                                header=None, names=columns,
                                dtype={'STAID': str, 'QAQC': str})
            # Convert date columns to datetime and set as index
            basin_df['DATE'] = pd.to_datetime(basin_df[['YEAR', 'MONTH', 'DAY']])
            basin_df.set_index('DATE', inplace=True)
            # Drop unnecessary columns
            basin_df.drop(columns=['YEAR', 'MONTH', 'DAY', 'QAQC'], inplace=True)
            # Add dataframe to dictionary
            data_dict[basin_df['STAID'].iloc[0]] = basin_df
            
    import matplotlib.pyplot as plt
    import re

    # Read in forcing data
    COLS = ["Year", "Mnth", "Day", "Hr",
            "dayl(s)", "prcp(mm/day)", "srad(W/m2)",
            "swe(mm)", "tmax(C)", "tmin(C)", "vp(Pa)"]

    for huc in os.listdir(FORCING_DATA_DIR):
        for basin_data in os.listdir(Path(FORCING_DATA_DIR, huc)):
            # Get basin number
            basin_no = basin_data.split('_')[0]
            
            # Check to see if the first line looks like YYYY MM DD, if so there are no headers
            # Get the first non-empty line
            with open(Path(FORCING_DATA_DIR, huc, basin_data), 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:            # skip any empty lines just in case
                        break
                    
            has_header_row = not re.match(r"^\d{4}\s+\d{2}\s+\d{2}", line)
            
            if has_header_row:
                basin_df = pd.read_csv(Path(FORCING_DATA_DIR, huc, basin_data), skiprows=3, sep=r"\s+")
            else:
                basin_df = pd.read_csv(Path(FORCING_DATA_DIR, huc, basin_data), sep=r"\s+", header=None, names=COLS)
                
            basin_df.rename(columns={'Mnth': 'Month'}, inplace=True)

            # Convert date columns to datetime and set as index
            basin_df['DATE'] = pd.to_datetime(basin_df[['Year', 'Month', 'Day']])
            basin_df.set_index('DATE', inplace=True)
            # Drop unnecessary columns
            basin_df.drop(columns=['Year', 'Month', 'Day', 'Hr'], inplace=True)
            
            # Make sure all dtypes are numeric
            for dtype in basin_df.dtypes:
                assert dtype != object
                
            # Concatenate forcing dataframe with discharge dataframe
            try:
                data_dict[basin_no] = pd.concat([data_dict[basin_no], basin_df], axis=1)
            except Exception as e:
                print(f"STAID not found: {e}")
                
    # Read in annual average hydrometeorological data
    HMET_STATIC_COLS = ['HUC', 'STAID', 'Annual Runoff (mm d-1)', 'Annual Precip (mm d-1)', 'Annual PET (mm d-1)', 'Annual Temp (C)']
    STATIC_COLS = ['HUC', 'STAID', 'DA (km2)', 'Elevation (m)', 'Slope (m km-1)', 'Frac Forest (%)']
    hmet_static_df = pd.read_csv(Path(STATIC_DATA_DIR, 'basin_annual_hydrometeorology_characteristics_nldas.txt'), 
                                sep=r'\s+', names=HMET_STATIC_COLS, dtype={'STAID': str}, header=0)

    static_df = pd.read_csv(Path(STATIC_DATA_DIR, 'basin_physical_characteristics.txt'), sep=r'\s+',
                            names=STATIC_COLS, dtype={'STAID': str}, header=0)

    # Append static data to respective basin dataframe
    for basin, df in data_dict.items():
        row_hmet = hmet_static_df[hmet_static_df['STAID'] == basin]
        row = static_df[static_df['STAID'] == basin]
        row_hmet.name = basin
        if not row_hmet.empty:
            static_values_hmet = row_hmet.iloc[0]
            static_values = row.iloc[0]
            
        # Append HMET static variables
        data_dict[basin]['Annual Runoff (mm d-1)'] = static_values_hmet['Annual Runoff (mm d-1)']
        data_dict[basin]['Annual Precip (mm d-1)'] = static_values_hmet['Annual Precip (mm d-1)']
        data_dict[basin]['Annual PET (mm d-1)'] = static_values_hmet['Annual PET (mm d-1)']
        data_dict[basin]['Annual Temp (C)'] = static_values_hmet['Annual Temp (C)']
        
        # Append static variables
        data_dict[basin]['DA (km2)'] = static_values['DA (km2)']
        data_dict[basin]['Elevation (m)'] = static_values['Elevation (m)']
        data_dict[basin]['Slope (m km-1)'] = static_values['Slope (m km-1)']
        data_dict[basin]['Frac Forest (%)'] = static_values['Frac Forest (%)']
        
    # Turn data dict into master dataframe
    master_df = pd.concat(data_dict.values(), axis=0)
    master_df.dropna(inplace=True)

    master_df.head()
    master_df.to_csv(Path(DATA_DIR, 'processed', 'CAMELS_daymet.csv'), index=True)