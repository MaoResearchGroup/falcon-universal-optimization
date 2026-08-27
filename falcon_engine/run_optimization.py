import numpy as np
import pandas as pd
import pickle
from falcon_engine.utilities import infer_raw_bounds_from_data, print_slowly
from .search_algorithms import OptimizationSearch
import shap

def run_optimization_pipeline(opt_methods, num_formulations, MAX_cell_targets, MIN_cell_targets, RUN_NAME, diversity_threshold, raw_bounds=None):
    """
    Run optimization for the given cell types and method.

    If raw_bounds is None, bounds are inferred from the min/max of each input
    parameter in the training dataset saved in the pipeline. If raw_bounds is a
    dict, any provided parameter bounds override the inferred dataset range.
    """
    not_initiliazed = True
    optimizer = None
    optimized_formulations = pd.DataFrame()

    for opt_method in opt_methods:
        print_slowly('\n######### DE NOVO FORMULATION SEARCH #####')
        print_slowly(f"Optimization Algorithm: {opt_method}")
        print_slowly(f"Number of Formulations: {num_formulations}")

        if opt_method in ['DA', 'BO']:
            print_slowly(f"Max Cell Target: {MAX_cell_targets[0]}")
        elif opt_method == 'NSGAII':
            print_slowly(f"Max Cell Targets: {MAX_cell_targets}")
            print_slowly(f"Min Cell Targets: {MIN_cell_targets}")
        else:   
            print_slowly(f"Minimizing variance across models for: {MAX_cell_targets + MIN_cell_targets}")

        # load the model and scalers 
        cell_type_list = MAX_cell_targets + MIN_cell_targets # we want to make single objective predictions for MAX_cell_targfets
        if not cell_type_list:
            raise ValueError("At least one output target must be listed in MAX_cell_targets or MIN_cell_targets.")

        models = {}
        input_scalars = {}
        output_scalars = {}
        training_data = {}
        scaled_bounds = {}  # final scaled bounds per cell_type
        pipeline = None
        input_param_names = None
        effective_raw_bounds = None
        search_scaled_bounds = None

        for cell_type in cell_type_list:
            model_path = f'output/{RUN_NAME}/{cell_type}/'
            with open(f'{model_path}Pipeline_dict.pkl', 'rb') as file:
                pipeline = pickle.load(file)

            models[cell_type] = pipeline['Model_Selection']['Best_Model']['Model']
            output_scalars[cell_type] = pipeline['Data_preprocessing']['Output_Scaler']
            input_scalars[cell_type] = pipeline['Data_preprocessing']['Scalers']
            training_data[cell_type] = pipeline['Data_preprocessing']['X']
            pipeline_input_params = pipeline['Data_preprocessing']['Input_Params']
            if input_param_names is None:
                input_param_names = pipeline_input_params
            elif input_param_names != pipeline_input_params:
                raise ValueError(
                    "All target pipelines must use the same input_param_names. "
                    f"Expected {input_param_names}, got {pipeline_input_params} for {cell_type}."
                )

            inferred_bounds = infer_raw_bounds_from_data(
                pipeline['Data_preprocessing']['raw_data'],
                input_param_names
            )
            effective_raw_bounds = inferred_bounds.copy()
            if raw_bounds is not None:
                effective_raw_bounds.update(raw_bounds)

            missing_bounds = [param for param in input_param_names if param not in effective_raw_bounds]
            if missing_bounds:
                raise KeyError(f"Missing optimization bounds for input parameters: {missing_bounds}")

            # Scale bounds for search algorithms
            scaler = input_scalars[cell_type]
            cell_scaled_bounds = []
            
            for param in input_param_names:
                raw_min, raw_max = effective_raw_bounds[param]
                scaled_min = scaler[param].transform([[raw_min]])[0][0]
                scaled_max = scaler[param].transform([[raw_max]])[0][0]
                cell_scaled_bounds.append((scaled_min, scaled_max))

            scaled_bounds[cell_type] = cell_scaled_bounds
            if search_scaled_bounds is None:
                search_scaled_bounds = cell_scaled_bounds

        print("\n--- Raw (Unscaled) Bounds for Each Input Parameter ---")
        for param in input_param_names:
            low, high = effective_raw_bounds[param]
            bound_source = "manual" if raw_bounds is not None and param in raw_bounds else "dataset"
            print(f"{param}: ({low:.4f}, {high:.4f}) [{bound_source}]")
        
        #max feature importance calculations for cell types 
        # max feature importance calculations for all cell types
        shap_importances = []
        for cell_type in cell_type_list:
            shap_imp = shap_analysis(
                models[cell_type],
                training_data[cell_type],
                input_param_names
            )
            shap_importances.append(shap_imp)
        max_feature_importance = pd.concat(shap_importances, axis=1).max(axis=1)

        #initialize search class
        if (not_initiliazed): 
            not_initiliazed = False
            optimizer = OptimizationSearch(models, 
                                        input_scalars, 
                                        output_scalars, 
                                        training_data, 
                                        input_param_names,
                                            cell_type_list, 
                                            max_feature_importance, 
                                            diversity_threshold, 
                                            opt_method, 
                                            search_scaled_bounds, 
                                            RUN_NAME)

        # Run optimization for each method
        if opt_method == "DA":
            optimizer.opt_method = opt_method
            optimizer.all_evaluations = []
            suggested_LNPs = optimizer.run_dual_annealing(num_formulations)
        elif opt_method =="BO":
            optimizer.opt_method = opt_method
            optimizer.all_evaluations = []
            suggested_LNPs = optimizer.run_bayesian(num_formulations)
        elif opt_method =="i-optimal":
            optimizer.opt_method = opt_method
            optimizer.all_evaluations = []
            suggested_LNPs = optimizer.run_i_optimal(num_formulations)
        elif opt_method =="NSGAII":
            optimizer.opt_method = opt_method
            optimizer.all_evaluations = []
            suggested_LNPs = optimizer.run_nsga2(num_formulations, 
                                                MAX_cell_targets, 
                                                MIN_cell_targets)
        elif opt_method == "i-optimal_minCT":
            optimizer.opt_method = opt_method
            suggested_LNPs = optimizer.run_i_optimal_minCT(num_formulations)
        else:
            raise KeyError
        
        #Save as .pkl and as excel for user
        suggested_LNPs.to_csv(f'output/{RUN_NAME}/{opt_method}_valid_suggestions.csv', index=False)
        optimized_formulations = pd.concat([optimized_formulations, suggested_LNPs], ignore_index=True)
        print(optimized_formulations)

        # Save full evaluation history
        all_eval_df = pd.DataFrame(optimizer.all_evaluations)
        all_eval_df.to_csv(
            f'output/{RUN_NAME}/{opt_method}_all_evaluations.csv',
            index=False
        )

    return optimized_formulations

def shap_analysis(model, train_data, input_param_names):

    explainer = shap.Explainer(model)
    X = pd.DataFrame(train_data, columns=input_param_names)
    shap_values = explainer(X)
    shap_matrix = shap_values.values 
    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    feature_importance = pd.Series(mean_abs_shap, index=input_param_names)

    return feature_importance
