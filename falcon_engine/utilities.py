import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import sys
import time 
import os


def load_clean_dataset(data_file_path):
    df = pd.read_csv(data_file_path)
    df = df.dropna(axis=1, how='all')
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    return df


def infer_raw_bounds_from_data(data, input_param_names, margin_fraction=0.0):
    if isinstance(data, (str, os.PathLike)):
        df = load_clean_dataset(data)
        source = str(data)
    else:
        df = data.copy()
        source = 'provided dataframe'

    missing_params = [param for param in input_param_names if param not in df.columns]
    if missing_params:
        raise KeyError(f"Input parameter columns missing from {source}: {missing_params}")

    bounds = {}
    for param in input_param_names:
        values = pd.to_numeric(df[param], errors='coerce').dropna()
        if values.empty:
            raise ValueError(f"Input parameter column '{param}' has no numeric values in {source}.")

        raw_min = float(values.min())
        raw_max = float(values.max())
        if raw_min == raw_max:
            raise ValueError(
                f"Input parameter column '{param}' has a constant value ({raw_min}) in {source}; "
                "optimization needs at least two distinct values or a manual bound."
            )

        margin = (raw_max - raw_min) * margin_fraction
        bounds[param] = (raw_min - margin, raw_max + margin)

    return bounds

def init_pipeline(pipeline_path, input_param_names, RUN_NAME, cell, param_type, data_file_path, prefix, output_floor, N_CV, model_list ):
    
    print_slowly('\n\n########## INITIALIZING MODEL TRAINING PIPELINE ##############\n\n')
    #Saving/Loading
    model_save_path           = f"output/{RUN_NAME}/{cell}/" # Where to save model, results, and training data 
    
    #initialize Pipeline Config and Data Storage Dictionary
    pipeline_dict = {'Cell' : cell,
                    'STEPS_COMPLETED':{
                        'Preprocessing': True,
                        'Model_Selection': True,
                        'Learning_Curve': True
                        },
                    'Saving':{
                        'RUN_NAME': RUN_NAME,
                        'Models': model_save_path,
                        },
                    'Data_preprocessing': {
                        'Data_Path': data_file_path,
                        'Formula_param_type': param_type,
                        'Input_Params': input_param_names,
                        'prefix' : prefix,
                        'output_floor':output_floor, 
                        'Scalers': {},
                        'Output_Scaler': None,
                        'X' : None, 
                        'y': None, 
                        'all_proc_data' : None,
                        'raw_data' : None
                        },
                    'Model_Selection': {
                        'Method': 'Nested CV',
                        'N_CV' : N_CV,
                        'Model_list': model_list,
                        'NESTED_CV': {},
                        'Best_Model':{
                            'Model_Name' : None, 
                            'Model': None, 
                            'Hyper_Params': None,
                            'Predictions' : None,
                            'MAE': None,
                            'Spearman':None,
                            'Pearson': None,
                            'HL_1': None
                            },
                            
                        },
                    'Learning_Curve':
                    {'NUM_ITER': None,
                        'num_splits': None,
                        'num_sizes': None,
                        'Train_Error': None,
                        'Valid_Error': None
                        }
                    }
  
    ####check save paths ########

    if not os.path.exists(model_save_path):
        # Create directory with execute bit so it is traversable/writeable.
        os.makedirs(model_save_path, mode=0o775, exist_ok=True)
        print(f"Directory '{model_save_path}' created.")
    else:
        print(f"Directory '{model_save_path}' already exists.")

    # Repair legacy directories created without execute permission (e.g., 0o666).
    try:
        os.chmod(model_save_path, 0o775)
    except OSError:
        pass

    with open(pipeline_path , 'wb') as file:
        pickle.dump(pipeline_dict, file)
        print(f"\n\n--- SAVED New {cell} Pipeline CONFIG  ---")
    return pipeline_dict    
def save_pipeline(pipeline, path, step):
    c = pipeline['Cell']
    with open(path , 'wb') as file:
            pickle.dump(pipeline, file)
    print(f"\n--- SAVED PIPELINE: {step} CONFIG AND RESULTS for {c}  ---")
def extract_training_data(pipeline):
    #Assign variables based on dictionary
    cell_type = pipeline['Cell']
    data_path = pipeline['Data_preprocessing']['Data_Path']
    input_params = pipeline['Data_preprocessing']['Input_Params']
    prefix = pipeline['Data_preprocessing']['prefix']
    output_floor = pipeline['Data_preprocessing']['output_floor']
    
    #Extract datafile
    df = load_clean_dataset(data_path)

    formula_col = next((col for col in ['Formula_label', 'Formula_Label'] if col in df.columns), None)
    if formula_col is None:
        raise KeyError("Dataset must include a formulation label column named 'Formula_label' or 'Formula_Label'.")

    target_col = prefix + cell_type
    if target_col not in df.columns and cell_type in df.columns:
        target_col = cell_type
    if target_col not in df.columns:
        raise KeyError(f"Target column '{prefix + cell_type}' was not found in {data_path}.")

    missing_params = [param for param in input_params if param not in df.columns]
    if missing_params:
        raise KeyError(f"Input parameter columns missing from {data_path}: {missing_params}")

    helper_col = next((col for col in ['Helper_lipid', 'Solvent'] if col in df.columns), None)
    batch_col = next((col for col in ['batch', 'Batch'] if col in df.columns), None)
    fraction_col = next((col for col in ['Fraction', 'fraction', 'Column1'] if col in df.columns), None)

    metadata_cols = [formula_col]
    if helper_col is not None:
        metadata_cols.append(helper_col)
    if batch_col is not None:
        metadata_cols.append(batch_col)
    if fraction_col is not None:
        metadata_cols.append(fraction_col)

    #Formatting Training Data
    raw_data = df[metadata_cols + input_params + [target_col]].copy()
    raw_data = raw_data.rename(columns={formula_col: 'Formula_label', target_col: prefix + cell_type})
    if helper_col is not None:
        raw_data = raw_data.rename(columns={helper_col: 'Helper_lipid'})
    if batch_col is not None:
        raw_data = raw_data.rename(columns={batch_col: 'Batch'})
    if fraction_col is not None:
        raw_data = raw_data.rename(columns={fraction_col: 'Fraction'})
    raw_data = raw_data.dropna() #Remove any NaN rows

    processed_data = raw_data.copy()

    #floor all output values below the noise
    processed_data.loc[processed_data[prefix + cell_type] < output_floor, prefix + cell_type] = output_floor 

    print("Input Parameters used:", input_params)
    print("Number of Datapoints used:", len(processed_data.index))

    scalers = {}

    for param in input_params:
        scaler = MinMaxScaler()
        processed_data[param] = scaler.fit_transform(processed_data[[param]])
        scalers[param] = scaler


    X = processed_data[input_params]                         
    Y = processed_data[prefix + cell_type].to_numpy()
    scaler = MinMaxScaler().fit(Y.reshape(-1,1))
    temp_Y = scaler.transform(Y.reshape(-1,1))
    Y = pd.DataFrame(temp_Y, columns = ["Scaled_" + prefix + cell_type])

    #Update Pipeline dictionary
    pipeline['Data_preprocessing']['Output_Scaler'] = scaler
    pipeline['Data_preprocessing']['Scalers'] = scalers
    pipeline['Data_preprocessing']['X'] = X
    pipeline['Data_preprocessing']['y'] = Y
    pipeline['Data_preprocessing']['all_proc_data'] = processed_data
    pipeline['Data_preprocessing']['raw_data'] = raw_data
    pipeline['STEPS_COMPLETED']['Preprocessing'] = True

    return pipeline, X,Y, processed_data
def print_slowly(text, delay=0.03):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()
def startup_banner():
    os.system('cls' if os.name == 'nt' else 'clear')  # Clear terminal
    banner = r"""

███████╗ █████╗ ██╗      ██████╗ ██████╗ ███╗   ██╗
██╔════╝██╔══██╗██║     ██╔════╝██╔═══██╗████╗  ██║
█████╗  ███████║██║     ██║     ██║   ██║██╔██╗ ██║
██╔══╝  ██╔══██║██║     ██║     ██║   ██║██║╚██╗██║
██║     ██║  ██║███████╗╚██████╗╚██████╔╝██║ ╚████║
╚═╝     ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝                                      
"""
    print_slowly(banner, delay=0.0015)

    meta_info = """
Author(s)    : Wu Han (Enoch) Toh, Leonardo Cheng et al.
Version      : FALCON v2.0
Description  : Machine Learning-Driven Multi-Objective Optimization Engine for Cell-Selective LNP Design
License      : MIT License
Repository   : https://github.com/MaoResearchGroup/falcon-lnp-optimization
Affiliation  : Mao Research Group, Institute for NanoBioTechnology (INBT), Johns Hopkins University  
------------------------------------------------------------
"""
    #print meta_info normally 
    print(meta_info)
    steps = [
        "Initializing FALCON Engine...",
        "Perching...",
        "Diving....",
        "FALCON is in flight!"
    ]

    for step in steps:
        print_slowly(step, delay=0.04)
        time.sleep(0.2)

    time.sleep(0.3)  # Pause before next action
