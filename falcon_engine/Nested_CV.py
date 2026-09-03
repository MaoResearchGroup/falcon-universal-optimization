"""**Hyperparameter Optimization**"""

# import the necessary libraries to execute this code
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None
from sklearn.metrics import mean_absolute_error
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV as RSCV
import copy
import os

# import model frameworks
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from falcon_engine.utilities import print_slowly

class NESTED_CV:
  
    """
    These functions were adapted from our previously published pipeline (https://pubs.acs.org/doi/10.1021/acsnano.4c07615), 
      using procedures for hyperparameter optimization and nested cross-validation from the Aspuru-Guzik group's GitHub repository.
    NESTED_CV Class:
    - based on a pipeline developed for long acting injectible (LAI) drug delivey systems by the Aspuru-Guzik Group https://github.com/aspuru-guzik-group/long-acting-injectables
    - once model type is selected, NEST_CV will be conducted, data is split as follows:
          - Hold-out set - 15% of the total data will be randomly split stratified by helper lipid class for final hold-out validation
          - Outer loop (test) - 5 K fold split of the remaining 85% (68% training, 17% validation) for model training
          - inner loop (hyperparameter optimization) - Final K fold split of the remaining 68% training set for HP hoptimization
    - prints progress and reults at the end of each loop
    - configures a pandas dataframe with the reults of the NESTED_CV
    - fits and trains the best model on all the training data based on the results of the NESTED_CV
    """

    #Functions here
    def __init__(self, model_type = None):
        
        if model_type == 'MLR':
          self.user_defined_model = LinearRegression()
          self.p_grid = {'fit_intercept':[True, False],
                         'positive':[True, False]}
    
        elif model_type == 'lasso':
          self.user_defined_model = linear_model.Lasso()
          self.p_grid = {'alpha':[0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0],
                        'positive':[True, False]}

        elif model_type == 'ENet':
          self.user_defined_model = linear_model.ElasticNet(random_state=4)
          self.p_grid = {
                        'alpha':[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
                        'l1_ratio':[0.1, 0.3, 0.5, 0.7, 0.9],
                        'fit_intercept':[True, False],
                        'positive':[True, False]
                        }

        elif model_type == 'kNN':
          self.user_defined_model = KNeighborsRegressor()
          self.p_grid ={'n_neighbors':[2, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 50],
                        'weights': ["uniform", 'distance'],
                        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                        'leaf_size': [10, 30, 50, 75, 100],
                        'p':[1, 2],
                        'metric': ['minkowski']}

        elif model_type == 'PLS':
          self.user_defined_model = PLSRegression()
          self.p_grid ={'n_components':[2, 4, 6],
                        'max_iter': [250, 500, 750, 1000]}

        elif model_type == 'SVR':
          self.user_defined_model = SVR()
          self.p_grid ={'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
                        'degree':[2, 3, 4, 5, 6],
                        'gamma':['scale', 'auto'],
                        'C':[0.1, 0.5, 1, 2],
                        'epsilon':[0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2],
                        'shrinking': [True, False]}
        
        elif model_type == 'DT':
          self.user_defined_model = DecisionTreeRegressor(random_state=4)
          self.p_grid ={'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                        'splitter':['best', 'random'],
                        'max_depth':[None],
                        'min_samples_split':[2,4,6],
                        'min_samples_leaf':[1,2,4],
                        'max_features': [None, 1.0, 'sqrt','log2'],
                        'ccp_alpha': [0, 0.05, 0.1, 0.15]}  
        
        elif model_type == 'RF':
          self.user_defined_model = RandomForestRegressor(random_state=4)
          self.p_grid ={'n_estimators':[100,200,300,400],
                        'criterion':['squared_error', 'absolute_error'],
                        'max_depth': [2, 6, 10, 14, 18, 22, 26, 30],
                        'min_samples_split':[2,4,6,8],
                        'min_samples_leaf':[1,2,4],
                        'min_weight_fraction_leaf':[0.0],
                        'max_features': [None, 'sqrt'],
                        'max_leaf_nodes':[None],
                        'min_impurity_decrease': [0.0],
                        'bootstrap':[True],
                        'oob_score':[True],
                        'ccp_alpha': [0, 0.005, 0.01]}

        elif model_type == 'ET':
          self.user_defined_model = ExtraTreesRegressor(random_state=4)
          self.p_grid = {
                        'n_estimators':[100, 200, 300, 400],
                        'criterion':['squared_error', 'absolute_error'],
                        'max_depth':[None, 6, 10, 14, 18, 22],
                        'min_samples_split':[2, 4, 6, 8],
                        'min_samples_leaf':[1, 2, 4],
                        'max_features':[None, 'sqrt'],
                        'bootstrap':[False, True],
                        'ccp_alpha':[0, 0.005, 0.01]
                        }

        elif model_type == 'GBR':
          self.user_defined_model = GradientBoostingRegressor(random_state=4)
          self.p_grid = {
                        'loss':['squared_error', 'absolute_error', 'huber'],
                        'n_estimators':[100, 200, 300, 400],
                        'learning_rate':[0.3, 0.2, 0.1, 0.05, 0.01],
                        'subsample':[0.5, 0.7, 0.9, 1.0],
                        'criterion':['friedman_mse', 'squared_error'],
                        'max_depth':[2, 3, 4, 5, 6, 8],
                        'min_samples_split':[2, 4, 6, 8],
                        'min_samples_leaf':[1, 2, 4],
                        'max_features':[None, 'sqrt']
                        }

        elif model_type == 'HGBR':
          self.user_defined_model = HistGradientBoostingRegressor(random_state=4)
          self.p_grid = {
                        'loss':['squared_error', 'absolute_error'],
                        'learning_rate':[0.3, 0.2, 0.1, 0.05, 0.01],
                        'max_iter':[100, 200, 300, 400],
                        'max_leaf_nodes':[15, 31, 63, 127],
                        'max_depth':[None, 3, 5, 8, 12],
                        'min_samples_leaf':[10, 20, 40],
                        'l2_regularization':[0.0, 0.001, 0.01, 0.1]
                        }

        elif model_type == 'ADA':
          self.user_defined_model = AdaBoostRegressor(random_state=4)
          self.p_grid = {
                        'n_estimators':[50, 100, 200, 400],
                        'learning_rate':[1.0, 0.5, 0.2, 0.1, 0.05, 0.01],
                        'loss':['linear', 'square', 'exponential']
                        }
          
        elif model_type == 'LGBM':
          self.user_defined_model = LGBMRegressor(random_state=10, verbosity=-1, n_jobs=1)
          self.p_grid ={"n_estimators":[100,150,200,250,300,400,500,600],
                        'boosting_type': ['gbdt', 'dart', 'goss'],
                        'num_leaves':[16,32,64,128,256],
                        'max_bin': [10, 20, 30],
                        'max_depth': [2, 6, 10, 14, 18, 22, 26, 30],
                        'learning_rate':[0.1,0.01,0.001,0.0001],
                        'min_child_weight': [0.001,0.01,0.1,1.0,10.0],
                        'subsample': [0.4,0.6,0.8,1.0],
                        'min_child_samples':[2,10,20,40,100],
                        'reg_alpha': [0, 0.005, 0.01, 0.015],
                        'reg_lambda': [0, 0.005, 0.01, 0.015]}
        
        elif model_type == 'XGB':
          self.user_defined_model = XGBRegressor(objective ='reg:squarederror')
          self.p_grid ={#'booster': ['gbtree', 'gblinear', 'dart'],
                        "n_estimators":[100, 150, 300, 400],
                        'max_depth':[3, 4, 5, 6, 7, 8, 9, 10],
                        'gamma':[0, 2, 4, 6, 8, 10],
                        'learning_rate':[0.3, 0.2, 0.1, 0.05, 0.01],
                        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        'min_child_weight': [1.0, 2.0, 4.0, 5.0],
                        'max_delta_step':[1, 2, 4, 6, 8, 10],
                        'reg_alpha':[0.001, 0.01, 0.1],
                        'reg_lambda': [0.001, 0.01, 0.1]} 
        elif model_type == 'MLP':
          self.user_defined_model = MLPRegressor()
          self.p_grid ={
                        'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
                        'activation': ['tanh', 'relu', 'logistic', 'identity'],
                        'solver': ['sgd', 'adam'],
                        'alpha': [0.0001,0.01, 0.05],
                        'batch_size':['auto', 64, 128],
                        'max_iter': [100, 200, 300],
                        'learning_rate': ['constant','adaptive'],
                        'tol': [1e-4, 1e-5]
                        }                
        
        else:
          print("#######################\nSELECTION UNAVAILABLE!\n#######################\n\n")

    def filter_xgb_params(params):
      booster = params.get('booster', 'gbtree')
      allowed_params = {
        'gbtree': {'booster', 'n_estimators', 'max_depth', 'gamma', 'learning_rate', 'subsample',
                   'min_child_weight', 'max_delta_step', 'reg_alpha', 'reg_lambda'},
        'dart': {'booster', 'n_estimators', 'max_depth', 'gamma', 'learning_rate', 'subsample',
                 'min_child_weight', 'max_delta_step', 'reg_alpha', 'reg_lambda'},
        'gblinear': {'booster', 'n_estimators', 'learning_rate', 'reg_alpha', 'reg_lambda'}
      }
      return {k: v for k, v in params.items() if k in allowed_params[booster]}

    def _target_vector(self, y):
        return np.asarray(y).reshape(-1)

    def _make_stratify_labels(self, y, n_bins=5, require_repeated=True):
        y_values = pd.Series(self._target_vector(y)).astype(float)
        n_unique = y_values.nunique(dropna=True)
        if n_unique < 2:
            return None

        n_bins = min(n_bins, n_unique)
        try:
            labels = pd.qcut(y_values.rank(method='first'), q=n_bins, labels=False, duplicates='drop')
        except ValueError:
            return None

        if labels.nunique(dropna=True) < 2:
            return None
        if require_repeated and labels.value_counts().min() < 2:
            return None
        return labels.astype(int).to_numpy()

    def _score_holdout_distribution(self, y_all, y_test, bins):
        full_hist, _ = np.histogram(y_all, bins=bins)
        test_hist, _ = np.histogram(y_test, bins=bins)
        if full_hist.sum() == 0 or test_hist.sum() == 0:
            return np.inf

        full_profile = full_hist / full_hist.sum()
        test_profile = test_hist / test_hist.sum()
        empty_bins = (full_hist > 0) & (test_hist == 0)
        return np.mean(np.abs(full_profile - test_profile)) + 0.2 * empty_bins.mean()

    def _group_holdout_split(self, indices, y, groups, test_size=0.2, random_state=4):
        groups = pd.Series(groups).reset_index(drop=True)
        valid_groups = groups.dropna().unique()
        if len(valid_groups) < 2:
            return None

        y_values = self._target_vector(y)
        y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
        if y_min == y_max:
            bins = np.array([y_min - 0.5, y_max + 0.5])
        else:
            bins = np.linspace(y_min, y_max, min(5, len(y_values)) + 1)

        rng = np.random.default_rng(random_state)
        unique_groups = np.array(valid_groups)
        n_groups = len(unique_groups)
        n_total = len(groups)
        candidates = []

        if n_groups <= 18:
            for mask_int in range(1, (1 << n_groups) - 1):
                mask = np.array([(mask_int >> i) & 1 for i in range(n_groups)], dtype=bool)
                candidates.append(unique_groups[mask])
        else:
            for _ in range(5000):
                mask = rng.random(n_groups) < test_size
                if not mask.any():
                    mask[rng.integers(0, n_groups)] = True
                if mask.all():
                    mask[rng.integers(0, n_groups)] = False
                candidates.append(unique_groups[mask])

        best_score = np.inf
        best_test_mask = None
        for test_groups in candidates:
            test_mask = groups.isin(test_groups).to_numpy()
            if test_mask.all() or not test_mask.any():
                continue
            size_error = abs(test_mask.mean() - test_size)
            distribution_error = self._score_holdout_distribution(y_values, y_values[test_mask], bins)
            score = distribution_error + 2.0 * size_error
            if score < best_score:
                best_score = score
                best_test_mask = test_mask

        if best_test_mask is None:
            return None

        positional_indices = np.asarray(indices)
        test_idx = positional_indices[best_test_mask]
        cv_idx = positional_indices[~best_test_mask]
        return cv_idx, test_idx

    def _make_cv_splits(self, X, y, groups, n_splits, random_state, split_name):
        if groups is not None:
            groups = pd.Series(groups).reset_index(drop=True)
            n_groups = groups.nunique(dropna=True)
            if n_groups >= 2:
                effective_splits = min(n_splits, n_groups)
                y_bins = self._make_stratify_labels(y, n_bins=effective_splits, require_repeated=False)
                if StratifiedGroupKFold is not None and y_bins is not None:
                    try:
                        splitter = StratifiedGroupKFold(n_splits=effective_splits, shuffle=True, random_state=random_state)
                        return list(splitter.split(X, y_bins, groups))
                    except ValueError:
                        pass

                splitter = GroupKFold(n_splits=effective_splits)
                return list(splitter.split(X, self._target_vector(y), groups))

            print_slowly(
                f"Only {n_groups} fraction group available for {split_name} CV; "
                "falling back to row-wise KFold because grouped splitting is impossible."
            )

        effective_splits = min(n_splits, len(X))
        splitter = KFold(n_splits=effective_splits, random_state=random_state, shuffle=True)
        return list(splitter.split(X))

    def _print_final_split_summary(self):
        print_slowly('Final hold-out split summary:')
        print_slowly(f'  CV-loop rows: {len(self.X_cv_loop)}, final test rows: {len(self.X_final_test)}')
        if self.Split_Group is not None:
            cv_groups = set(self.Split_Group_cv_loop.dropna().astype(str))
            test_groups = set(self.Split_Group_final_test.dropna().astype(str))
            overlap = cv_groups.intersection(test_groups)
            print_slowly(f'  Final test fractions: {sorted(test_groups)}')
            print_slowly(f'  Fraction overlap between CV-loop and final test: {sorted(overlap)}')
        if self.Batch is not None:
            print_slowly(f'  Final test batches represented: {sorted(set(self.Batch_final_test.dropna().astype(str)))}')
        print_slowly(
            f"  Target mean CV/test: {float(np.mean(self._target_vector(self.y_cv_loop))):.3f} / "
            f"{float(np.mean(self._target_vector(self.y_final_test))):.3f}"
        )

    def input_target(self, X, y, data):
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        self.cell_data = data.reset_index(drop=True)
        self.F = self.cell_data['Formula_label'] #used to track formulations
        self.HL = self.cell_data['Helper_lipid'] if 'Helper_lipid' in self.cell_data.columns else None
        self.Batch = self.cell_data['Batch'] if 'Batch' in self.cell_data.columns else None
        self.Split_Group = self.cell_data['Fraction'] if 'Fraction' in self.cell_data.columns else None
    
    def cross_validation(self, input_value):
        if input_value == None:
            NUM_TRIALS = 10
        else: 
            NUM_TRIALS = input_value

        self.itr_number = [] # create new empty list for itr number 
        self.outer_MAE = []
        self.outer_spearman = []
        self.outer_pearson = []
        self.inner_results = []
        self.model_params = []
        self.y_test_list = []
        self.F_test_list = []
        self.pred_list = []


        # Split CV loop model development set and final hold-out test set.
        # When fractions are present, keep all rows for the same fraction
        # together and choose the final test fractions that best preserve the target histogram.
        indices = self.X.index

        grouped_split = None
        if self.Split_Group is not None:
            grouped_split = self._group_holdout_split(indices, self.y, self.Split_Group, test_size=0.2, random_state=4)

        if grouped_split is not None:
            cv_idx, test_idx = grouped_split
        else:
            stratify_labels = self._make_stratify_labels(self.y, n_bins=5, require_repeated=True)
            if stratify_labels is None and self.HL is not None and self.HL.value_counts().min() >= 2:
                stratify_labels = self.HL
            cv_idx, test_idx, _,_ = train_test_split(indices, self.y, test_size=0.2, random_state=4, stratify=stratify_labels, shuffle=True)

        self.X_cv_loop, self.X_final_test, = self.X.iloc[cv_idx], self.X.iloc[test_idx]
        self.y_cv_loop, self.y_final_test  = self.y.iloc[cv_idx], self.y.iloc[test_idx]
        self.F_cv_loop, self.F_final_test  = self.F.iloc[cv_idx], self.F.iloc[test_idx]
        self.Batch_cv_loop = self.Batch.iloc[cv_idx] if self.Batch is not None else None
        self.Batch_final_test = self.Batch.iloc[test_idx] if self.Batch is not None else None
        self.Split_Group_cv_loop = self.Split_Group.iloc[cv_idx] if self.Split_Group is not None else None
        self.Split_Group_final_test = self.Split_Group.iloc[test_idx] if self.Split_Group is not None else None
        self._print_final_split_summary()
        
        # Nested CV
        cv_outer = self._make_cv_splits(self.X_cv_loop, self.y_cv_loop, self.Split_Group_cv_loop, NUM_TRIALS, 42, 'outer')
        num_outer_folds = len(cv_outer)

        for i, (train_index, test_index) in enumerate(cv_outer):
          X_train = self.X_cv_loop.iloc[train_index].copy()
          X_test = self.X_cv_loop.iloc[test_index].copy()
          y_train = self.y_cv_loop.iloc[train_index].copy()
          y_test = self.y_cv_loop.iloc[test_index].copy()
          F_train = self.F_cv_loop.iloc[train_index].copy()
          F_test = self.F_cv_loop.iloc[test_index].copy()
          Split_Group_train = self.Split_Group_cv_loop.iloc[train_index].copy() if self.Split_Group_cv_loop is not None else None

          #store test set information
          F_test = np.array(F_test) #prevents index from being brought from dataframe
          self.F_test_list.append(F_test)
          y_test = np.array(y_test) #prevents index from being brought from dataframe
          self.y_test_list.append(y_test)
                
          # configure the cross-validation procedure - inner loop (validation set/HP optimization)
          cv_inner = self._make_cv_splits(X_train, y_train, Split_Group_train, 5, 0, 'inner')

          # define search space
          # Default to single-process CV to avoid joblib permission issues on restricted systems.
          # Override with: export FALCON_N_JOBS=<int>
          n_jobs = int(os.environ.get("FALCON_N_JOBS", "1"))
          search = RSCV(self.user_defined_model, self.p_grid, n_iter=100, verbose=0, scoring='neg_mean_absolute_error', cv=cv_inner,  n_jobs=n_jobs, refit=True, random_state= 42) #### RANDOM STATE WAS NOT SET
                  
          # execute search
          y_train = np.ravel(y_train)
          result = search.fit(X_train, y_train)
              
          # get the best performing model fit on the whole training set
          best_model = result.best_estimator_

          # get the score for the best performing model and store
          best_score = abs(result.best_score_)
          self.inner_results.append(best_score)
                  
          #### evaluate model on the hold out dataset
          yhat = best_model.predict(X_test)
          yhat_eval = np.asarray(yhat).reshape(-1)
          y_test_eval = np.asarray(y_test).reshape(-1)

          #Cell-type transfection predictions
          self.pred_list.append(yhat_eval)

          # evaluate the model accuracy using the hold out dataset Mean Absolute Error
          acc = mean_absolute_error(y_test_eval, yhat_eval)
          spearmans_rank = stats.spearmanr(y_test_eval, yhat_eval)
          pearsons_r = stats.pearsonr(y_test_eval, yhat_eval)

          # store the result
          self.itr_number.append(i+1)
          self.outer_MAE.append(acc)
          self.outer_spearman.append(spearmans_rank)
          self.outer_pearson.append(pearsons_r)
          self.model_params.append(result.best_params_)

          # report progress at end of each inner loop
          #Note: Test score = outerloop - hold out - dataset score
          # Best_Valid_Score = innerloop iteration score
          print('\n################################################################\n\nSTATUS REPORT:')
          print('Iteration '+str(i+1)+' of '+str(num_outer_folds)+' runs completed') 
          print('Best_Valid_Score: %.3f, Hold_Out_MAE: %.3f,  Hold_Out_Spearman_Rank: %.3f, Hold_Out_Pearsons_R: %.3f, \n\nBest_Model_Params: \n%s' % (best_score, acc, spearmans_rank[0], pearsons_r[0], result.best_params_))
          #print("\n################################################################\n ")
          
    def results(self):   
        #create dataframe with results of nested CV
        list_of_tuples = list(zip(self.itr_number, self.inner_results, self.outer_MAE, self.outer_spearman,self.outer_pearson, self.model_params, self.F_test_list, self.y_test_list, self.pred_list))
        CV_dataset = pd.DataFrame(list_of_tuples, columns = ['Iter (CV Fold)', 
                                                             'Valid Score (MAE)', 
                                                             'Test Score (MAE)', 
                                                             'Spearmans Rank',
                                                             'Pearsons Correlation',
                                                             'Model HyperParms', 
                                                             'Formulation_Index (Hold out set)', 
                                                             'Experimental_Transfection',
                                                             'Predicted_Transfection'])
        CV_dataset['Score_difference'] = abs(CV_dataset['Valid Score (MAE)'] - CV_dataset['Test Score (MAE)']) #Groupby dataframe model iterations that best fit the data (i.e., minimize different between validitaion and test)
        CV_dataset.sort_values(by=['Score_difference', 'Test Score (MAE)'], ascending=True, inplace=True) 
        CV_dataset = CV_dataset.reset_index(drop=True) # Reset index of dataframe
        print('\n\n################################################################\n\n')
        print('Cross Validation Results', CV_dataset)
        # save the results as a class object
        self.CV_dataset = CV_dataset

 ###### Retrain model using best parameters to evaluate the test set MAE
    def FINAL_TEST_MAE(self):
        # assign the best hyperparameters from NESTED CV
        self.best_model_params = self.CV_dataset.iloc[0,5]
        print('\nFinal_Best_Model_Params: \n%s' % self.best_model_params)

        # set params from the best model to a class object
        best_model = self.user_defined_model.set_params(**self.best_model_params)

        #Fit on all cv loop training data
        self.best_model = best_model.fit(self.X_cv_loop, np.ravel(self.y_cv_loop))

        #Predict test set
        yhat = self.best_model.predict(self.X_final_test)
        yhat_eval = np.asarray(yhat).reshape(-1)
        y_test_eval = np.asarray(self.y_final_test).reshape(-1)

        #Make Predictions dataframe for downstream plotting
        pred_df = self.F_final_test.reset_index(drop=True).rename('Formula label').to_frame()
        yhat_df = pd.DataFrame(yhat_eval, columns = ['Predicted_Transfection'])
        y_test_df = copy.copy(self.y_final_test)
        y_test_df.reset_index(inplace = True, drop = True)
        pred_df = pd.concat([pred_df, yhat_df, y_test_df], axis = 1, ignore_index=True)
        pred_df.columns = ['Formula label','Predicted_Transfection', 'Experimental_Transfection']
        if self.Split_Group_final_test is not None:
            fraction_df = self.Split_Group_final_test.reset_index(drop=True).rename('Fraction')
            pred_df = pd.concat([pred_df, fraction_df], axis=1)
        if self.Batch_final_test is not None:
            batch_df = self.Batch_final_test.reset_index(drop=True).rename('Batch')
            pred_df = pd.concat([pred_df, batch_df], axis=1)

        #calclate performance
        AE = abs(pred_df['Predicted_Transfection'] - pred_df['Experimental_Transfection'])


        acc = mean_absolute_error(y_test_eval, yhat_eval)
        spearmans_rank = stats.spearmanr(y_test_eval, yhat_eval)
        pearsons_r = stats.pearsonr(y_test_eval, yhat_eval)

        y_true_baseline = np.asarray(self.y_final_test).reshape(-1)
        y_cv_baseline = np.asarray(self.y_cv_loop).reshape(-1)
        mean_baseline_pred = np.full(y_true_baseline.shape, np.mean(y_cv_baseline))
        median_baseline_pred = np.full(y_true_baseline.shape, np.median(y_cv_baseline))

        mean_baseline_AE = np.abs(y_true_baseline - mean_baseline_pred)
        median_baseline_AE = np.abs(y_true_baseline - median_baseline_pred)
        mean_baseline_MAE = float(np.mean(mean_baseline_AE))
        median_baseline_MAE = float(np.mean(median_baseline_AE))

        n = len(AE)
        model_AE_se = float(AE.std(ddof=1)) / np.sqrt(n)
        mean_baseline_se = float(np.std(mean_baseline_AE, ddof=1)) / np.sqrt(n)
        median_baseline_se = float(np.std(median_baseline_AE, ddof=1)) / np.sqrt(n)
      
        print_slowly('\n################################################################\n\n BEST MODEL FINAL HOLD_OUT PERFORMANCE:')
        print_slowly(f'FINAL_Hold_Out_MAE: {acc:.3f} ± {model_AE_se:.3f}')
        print_slowly(f'FINAL_Hold_Out_Spearman_Rank: {spearmans_rank[0]:.3f}, FINAL_Hold_Out_Pearsons_R: {pearsons_r[0]:.3f}')
        print_slowly(f'Mean Baseline MAE: {mean_baseline_MAE:.3f} ± {mean_baseline_se:.3f}')
        print_slowly(f'Median Baseline MAE: {median_baseline_MAE:.3f} ± {median_baseline_se:.3f}\n')

        return AE, acc, model_AE_se, spearmans_rank, pearsons_r, pred_df, mean_baseline_MAE, mean_baseline_se, median_baseline_MAE, median_baseline_se

    def best_model_refit(self):
        # assign the best model hyperparameters
        self.best_model_params = self.CV_dataset.iloc[0,5]
        # print('\nFinal_Best_Model_Params: \n%s' % self.best_model_params)
        # set params from the best model to a class object
        best_model = self.user_defined_model.set_params(**self.best_model_params)
        y_train = self.y.copy()
        y_train = np.ravel(y_train) #reformat
        self.best_model = best_model.fit(self.X, y_train) #Fit hyperparameter optimized model using all data as training set.
