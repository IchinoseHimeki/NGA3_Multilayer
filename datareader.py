"""
Copyright (c) 2025 by IchinoseHimeki(darwinlee1998@gmail.com)
Open source according to the terms of the GPLv3 license.
This script is a dataloader of main funciton.
"""
import os
import pandas as pd


def get_data(xlsxname):
    """Reads data from Excel files and performs length checks.

    Args:
        xlsxname (str): The name of the Excel file containing constraint data.

    Returns:
        tuple: A tuple containing the constraint data, pdata, wdata, ddata,
               edata, and croplen.  Returns None if there are length mismatches.

    Raises:
        ValueError: If Excel files are not found or sheets are missing.
    """

    try:
        conData = pd.read_excel(os.path.join(os.path.dirname(__file__), xlsxname), sheet_name='Con')
        pdata = pd.read_excel(os.path.join(os.path.dirname(__file__), "Cons.xlsx"), sheet_name='P').dropna()
        wdata = pd.read_excel(os.path.join(os.path.dirname(__file__), "Cons.xlsx"), sheet_name='W').dropna()
        ddata = pd.read_excel(os.path.join(os.path.dirname(__file__), "Cons.xlsx"), sheet_name='D').dropna()
        edata = pd.read_excel(os.path.join(os.path.dirname(__file__), "Cons.xlsx"), sheet_name='E').dropna()

    except FileNotFoundError:
        raise ValueError(f"Error: One or more Excel files not found.")
    except ValueError as e:  # Catch exceptions related to sheet names, etc.
        raise ValueError(f"Error reading Excel file: {e}")

    # Length checks using shape (more robust)
    if conData.shape[0] != pdata.shape[0] or conData.shape[0] != wdata.shape[0] or \
            conData.shape[0] != ddata.shape[0] or conData.shape[0] != edata.shape[0] or \
            (conData.shape[0] > 0 and (len(conData.columns) < len(['Region', 'landlimit', 'waterlimit']) or \
                                       conData.shape[1] != pdata.shape[1]-1 or \
                                       conData.shape[1] != wdata.shape[1]-1 or \
                                       conData.shape[1] != ddata.shape[1]-1)):
        print("Shape Mismatch Detected:")
        print("conData shape:", conData.shape)
        print("pdata shape:", pdata.shape)
        print("wdata shape:", wdata.shape)
        print("ddata shape:", ddata.shape)
        print("edata shape:", edata.shape)
        return None  # Or raise a more specific exception

    croplen = len(pdata.columns) - 1  # Assuming 'Region' is always present.  No need to subtract 1 if it's dropped
    return conData, pdata.drop('Region', axis=1), wdata.drop('Region', axis=1), ddata.drop('Region',
                                                                                           axis=1), edata.drop('Region',
                                                                                                               axis=1), croplen

def DataReader():
    data = get_data(xlsxname="Cons.xlsx")
    n_regions = len(data[0]['Region'])
    n_crops = data[5]
    land_limits = data[0]['landlimit'].to_numpy()
    water_limits = data[0]['waterlimit'].to_numpy()
    P = data[1].to_numpy()
    W = data[2].to_numpy()
    D = data[3].to_numpy()
    E = data[4].to_numpy().reshape(-1)
    return n_regions, n_crops, land_limits, water_limits, P, W, D, E
