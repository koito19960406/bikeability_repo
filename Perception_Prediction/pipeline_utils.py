import itertools
import numpy as np
import pandas as pd
import tqdm
import os
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from lightgbm import LGBMRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, mean_absolute_percentage_error, \
    mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler, QuantileTransformer, \
    PowerTransformer
from sklearn.svm import SVR
from scipy.stats import uniform as sp_uniform
import joblib


def print_results(names, results, test_scores):
    print()
    print("#" * 30 + "Results" + "#" * 30)
    counter = 0

    class Color:
        PURPLE = '\033[95m'
        CYAN = '\033[96m'
        DARKCYAN = '\033[36m'
        BLUE = '\033[94m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        END = '\033[0m'

    # Get max row
    clf_names = set([name.split("_")[1] for name in names])
    max_mean = {name: 0 for name in clf_names}
    max_mean_counter = {name: 0 for name in clf_names}
    for name, result in zip(names, results):
        counter += 1
        clf_name = name.split("_")[1]
        if result.mean() > max_mean[clf_name]:
            max_mean_counter[clf_name] = counter
            max_mean[clf_name] = result.mean()

    # print max row in BOLD
    counter = 0
    prev_clf_name = names[0].split("_")[1]
    for name, result, score in zip(names, results, test_scores):
        counter += 1
        clf_name = name.split("_")[1]
        if prev_clf_name != clf_name:
            print()
            prev_clf_name = clf_name
        msg = "%s: %f (%f) [test_score:%.3f]" % (name, result.mean(), result.std(), score)
        if counter == max_mean_counter[clf_name]:
            print(Color.BOLD + msg)
        else:
            print(Color.END + msg)


def create_pipelines(seed, verbose=1):
    """
         Creates a list of pipelines with preprocessing(PCA), models and scalers.

    :param seed: Random seed for models who needs it
    :return:
    """
    # put regression models
    models = [
#         ('RFR', RandomForestRegressor()),
#               ('LR', LinearRegression()),
              ('LGBMR', LGBMRegressor(random_state=seed))
#               ('SVR', SVR()),
#               ('LS', Lasso(random_state=seed)),
#               ('MLP', MLPRegressor(random_state=seed))
              ]
    scalers = [('StandardScaler', StandardScaler())
            #    ('MinMaxScaler', MinMaxScaler()),
            #    ('MaxAbsScaler', MaxAbsScaler()),
            #    ('RobustScaler', RobustScaler()),
            #    ('QuantileTransformer-Normal', QuantileTransformer(output_distribution='normal')),
            #    ('QuantileTransformer-Uniform', QuantileTransformer(output_distribution='uniform')),
            #    ('PowerTransformer-Yeo-Johnson', PowerTransformer(method='yeo-johnson')),
            #    ('Normalizer', Normalizer())
               ]
    additions = [('PCA', PCA(0.9))]# n_components=10
    # Create pipelines
    pipelines = []
    for model in models:
#         # Append only model
#         model_name = "_" + model[0]
#         pipelines.append((model_name, Pipeline([model])))

#         # Append model+scaler
#         for scalar in scalers:
#             model_name = scalar[0] + "_" + model[0]
#             pipelines.append((model_name, Pipeline([scalar, model])))

#         # To easier distinguish between with and without Additions (i.e: PCA)
#         # Append model+addition
#         for addition in additions:
#             model_name = "_" + model[0] + "-" + addition[0]
#             pipelines.append((model_name, Pipeline([addition, model])))

        # Append model+scaler+addition
        for scalar in scalers:
            for addition in additions:
                model_name = scalar[0] + "_" + model[0] + "-" + addition[0]
                pipelines.append((model_name, Pipeline([scalar, addition, model])))

    if verbose:
        print("Created these pipelines:")
        for pipe in pipelines:
            print(pipe[0])

    return pipelines


def run_cv_and_test(X_train, y_train, X_test, y_test, pipelines, scoring, seed, num_folds,
                    dataset_name, n_jobs):
    """

        Iterate over the pipelines, calculate CV mean and std scores, fit on train and predict on test.
        Return the results in a dataframe

    """

    # List that contains the rows for a dataframe
    rows_list = []

    # Lists for the pipeline results
    results = []
    names = []
    test_scores = []
    prev_clf_name = pipelines[0][0].split("_")[1]
    print("First name is : ", prev_clf_name)

    for name, model in pipelines:
        kfold = model_selection.KFold(n_splits=num_folds, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, n_jobs=n_jobs, scoring=scoring)
        results.append(cv_results)
        names.append(name)

        # Print CV results of the best CV classier
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

        # fit on train and predict on test
        model.fit(X_train, y_train)
        if scoring == "accuracy":
            curr_test_score = model.score(X_test, y_test)
        elif scoring == "roc_auc":
            y_pred = model.predict_proba(X_test)[:, 1]
            curr_test_score = roc_auc_score(y_test, y_pred)

        test_scores.append(curr_test_score)

        # Add separation line if different classifier applied
        rows_list, prev_clf_name = check_seperation_line(name, prev_clf_name, rows_list)

        # Add for final dataframe
        results_dict = {"Dataset": dataset_name,
                        "Classifier_Name": name,
                        "CV_mean": cv_results.mean(),
                        "CV_std": cv_results.std(),
                        "Test_score": curr_test_score
                        }
        rows_list.append(results_dict)

    print_results(names, results, test_scores)

    df = pd.DataFrame(rows_list)
    return df[["Dataset", "Classifier_Name", "CV_mean", "CV_std", "Test_score"]]


def run_cv_and_test_hypertuned_params(X_train, y_train, X_test, y_test, pipelines, scoring, seed, num_folds,
                                      hypertuned_params, dataset_name, n_jobs, model_output_folder):
    """

        Iterate over the pipelines, calculate CV mean and std scores, fit on train and predict on test.
        Return the results in a dataframe

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param scoring:
    :param seed:
    :param num_folds:
    :param model_output_folder:
    :return:
    """

    # List that contains the rows for a dataframe
    rows_list = []

    # Lists for the pipeline results
    results = []
    names = []
    test_scores = []
    prev_clf_name = pipelines[0][0].split("_")[1]
    print("First name is : ", prev_clf_name)

    # To be used within GridSearch (5 in your case)
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    # To be used in outer CV (you asked for num_folds)
    outer_cv = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    for name, model in tqdm.tqdm(pipelines):

        # Get model's hyper parameters
        model_name = name.split("_")[1]
        if "-" in model_name:
            model_name = model_name.split("-")[0]

        if model_name in hypertuned_params.keys():
            random_grid = hypertuned_params[model_name]
            print(random_grid)
        else:
            continue

        # Train nested-CV
        clf = GridSearchCV(estimator=model, param_grid=random_grid, cv=inner_cv, scoring=scoring,
                           verbose=2, n_jobs=n_jobs, refit=True)
        cv_results = model_selection.cross_val_score(clf, X_train, y_train, cv=outer_cv, n_jobs=n_jobs, 
                                                     verbose=2, scoring=scoring)
        results.append(cv_results)
        names.append(name)

        # Print CV results of the best CV classier
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

        # fit on train and predict on test
        best_params=clf.fit(X_train, y_train).best_params_
        joblib.dump(clf, os.path.join(model_output_folder,dataset_name+"__"+name+'.joblib'))
        model.fit(X_train, y_train)
        n_pcs= model['PCA'].components_.shape[0]
        y_pred = model.predict(X_test)
        # get different performance metrics
        score_mean_absolute_error=mean_absolute_error(y_test, y_pred)
        score_mean_absolute_percentage_error = mean_absolute_percentage_error(y_test, y_pred)
        score_root_mean_squared_error=mean_squared_error(y_test, y_pred, squared=False)
        score_r2=r2_score(y_test, y_pred)
        
#         curr_test_score=clf.score(X_test,y_test)

        test_scores.append(score_mean_absolute_error)

        # Add separation line if different classifier applied
        rows_list, prev_clf_name = check_seperation_line(name, prev_clf_name, rows_list)

        # Add for final dataframe
        results_dict = {"Dataset": dataset_name,
                        "Classifier_Name": name,
                        "CV_mean": cv_results.mean(),
                        "CV_std": cv_results.std(),
                        "mean_absolute_error":score_mean_absolute_error,
                        "mean_absolute_percentage_error":score_mean_absolute_percentage_error,
                        "root_mean_squared_error":score_root_mean_squared_error,
                        "r2_score":score_r2,
                        "n_pcs":n_pcs,
                        "Best_params":best_params
                        }
        rows_list.append(results_dict)

    print_results(names, results, test_scores)

    df = pd.DataFrame(rows_list)
    df.to_csv(os.path.join(model_output_folder,dataset_name+"__"+name+'.csv'))
    return df[["Dataset", "Classifier_Name", "CV_mean", "CV_std", "mean_absolute_error",
               "mean_absolute_percentage_error","root_mean_squared_error",
               "r2_score", "n_pcs","Best_params"]]


def check_seperation_line(name, prev_clf_name, rows_list):
    """
        Add empty row if different classifier ending

    """

    clf_name = name.split("_")[1]
    if prev_clf_name != clf_name:
        empty_dict = {"Dataset": dataset_name,
                      "Classifier_Name": "",
                      "CV_mean": "",
                      "CV_std": "",
                      "mean_absolute_error":"",
                      "mean_absolute_percentage_error":"",
                      "root_mean_squared_error":"",
                      "r2_score":"",
                      "n_pcs":"",
                      "Best_params":""
                      }
        rows_list.append(empty_dict)
        prev_clf_name = clf_name
    return rows_list, prev_clf_name


def get_hypertune_params():
    """

        Create a dictionary with classifier name as a key and it's hyper parameters options as a value

    :return:
    """
#     # RFR PARAMS
#     n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 1)] # 10
#     max_features = ['auto'] #['auto', 'sqrt']
#     max_depth = [int(x) for x in np.linspace(10, 110, num = 1)] # 11
# #     max_depth.append(None)
#     min_samples_split = [2] #[2, 5, 10]
#     min_samples_leaf = [2] # [1, 2, 4]
#     bootstrap = [True] # [True, False] 
#     rfr_params = {'RFR__n_estimators': n_estimators,
#                 'RFR__max_features': max_features,
#                 'RFR__max_depth': max_depth,
#                 'RFR__min_samples_split': min_samples_split,
#                 'RFR__min_samples_leaf': min_samples_leaf,
#                 'RFR__bootstrap': bootstrap}

#     # Linear Regression PARAMS
#     fit_intercept = [True, False]
#     lr_params = {'LR__fit_intercept': fit_intercept}

    # LGBMR PARAMS
    num_leaves=[int(x) for x in np.linspace(start = 20, stop = 200, num =3)]# [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    max_depth=[int(x) for x in np.linspace(start = 10, stop = 100, num = 3)]
    min_child_samples=[int(x) for x in np.linspace(start = 500, stop = 2000, num = 3)] # [int(x) for x in np.linspace(start = 100, stop = 500, num = 10)]
    min_child_weight=[1e-11,1e-9,1e-7] # [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
    subsample=[x for x in np.linspace(start = 0, stop = 1, num = 3)] # [x for x in np.linspace(start = 0, stop = 1, num = 10)]
    colsample_bytree=[x for x in np.linspace(start = 0, stop = 1, num = 3)] # [x for x in np.linspace(start = 0, stop = 1, num = 10)]
    reg_alpha=[0, 1, 10, 100] #[0, 1e-1, 1, 2, 5, 7, 10, 50, 100]
    reg_lambda= [0, 1, 10, 100] # [0, 1e-1, 1, 5, 10, 20, 50, 100] 
    lgbmr_params ={
                'LGBMR__num_leaves': num_leaves, 
                'LGBMR__max_depth':max_depth,
                'LGBMR__min_child_samples': min_child_samples, 
                'LGBMR__min_child_weight': min_child_weight,
                'LGBMR__subsample': subsample, 
                'LGBMR__colsample_bytree': colsample_bytree,
#                 'LGBMR__reg_alpha': reg_alpha,
#                 'LGBMR__reg_lambda': reg_lambda
                  }

#     # SVR PARAMS
#     C = [x for x in np.arange(0.1, 2, 0.2)]
#     kernel = ["linear", "poly", "rbf", "sigmoid"]
#     svr_params = {'SVR__C': C,
#                   'SVR__kernel': kernel,
#                   }

#     # Lasso Regression Params
#     alpha = [x for x in np.arange(0, 1, 0.01)]
#     ls_params = {'LS__alpha': alpha
#                  }

#     # MLP PARAMS
#     hidden_layer_sizes = [(x, y) for x, y in itertools.product([x for x in range(1, 3)], [x for x in range(5, 120, 5)])]
#     activation = ["tanh", "relu"]
#     solver = ["lbfgs", "sgd", "adam"]
#     alpha = [0.1, 0.001, 0.0001]
#     learning_rate = ["constant", "invscaling", "adaptive"]
#     mlp_params = {'MLP__hidden_layer_sizes': hidden_layer_sizes,
#                   'MLP__activation': activation,
#                   'MLP__solver': solver,
#                   'MLP__alpha': alpha,
#                   'MLP__learning_rate': learning_rate,
#                   }

    hypertuned_params = {
#         "RFR": rfr_params,
#                          "LR": lr_params,
                         "LGBMR": lgbmr_params
#                          "MLP": mlp_params,
#                          "SVR": svr_params,
#                          "LR": lr_params,
                         }

    return hypertuned_params    