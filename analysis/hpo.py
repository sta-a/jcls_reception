# %%
# %load_ext autoreload
# %autoreload 2
from_commandline = True

import warnings
warnings.simplefilter('once', RuntimeWarning)
import argparse
import time
import os
import pandas as pd
import numpy as np
from hpo_helpers import get_data, CustomGroupKFold, get_task_params, run_gridsearch
from scipy.stats import pearsonr


start = time.time()
if from_commandline:
    parser = argparse.ArgumentParser()
    parser.add_argument('languages')
    parser.add_argument('data_dir')
    parser.add_argument('tasks', nargs="*", type=str)
    # If --testing flag is set, testing is set to bool(True). 
    # If --no-testing flag is set, testing is set to bool(False)
    parser.add_argument('--testing', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    languages = [args.languages]
    data_dir = args.data_dir
    tasks = args.tasks
    testing = args.testing
else:
    # Don't use defaults because VSC interactive mode can't handle command line arguments
    languages = ['eng', 'ger']
    data_dir = '../data'
    tasks = ['regression', 'binary', 'library', 'multiclass']
    testing = True
print(languages, data_dir, tasks, testing )
n_outer_folds = 5


for language in languages:

    # Use full features set for classification to avoid error with underrepresented classes
    sentiscores_dir = os.path.join(data_dir, 'sentiscores', language)
    metadata_dir = os.path.join(data_dir, 'metadata', language)
    canonscores_dir = os.path.join(data_dir, 'canonscores', language)
    features_dir = features_dir = os.path.join(data_dir, 'features_None', language)
    gridsearch_dir = os.path.join(data_dir, 'nested_gridsearch', language)
    if not os.path.exists(gridsearch_dir):
        os.makedirs(gridsearch_dir, exist_ok=True)

    columns_list = [
    ['average_sentence_embedding', 'doc2vec_chunk_embedding'],
    ['average_sentence_embedding', 'doc2vec_chunk_embedding', 'pos']]
    if language == 'eng':
        columns_list.extend([
            ['average_sentence_embedding', 'doc2vec_chunk_embedding', '->'], 
            ['average_sentence_embedding', 'doc2vec_chunk_embedding', '->', 'pos']])
    # columns_list.extend([['passthrough']])
    # if testing:
    #     columns_list = [columns_list[-1]]

    for task in tasks:
        task_params = get_task_params(task, testing)
        for label_type in task_params['labels']:
            for features in task_params['features']:
                print(f'Task: {task}, Label_type: {label_type}, Features: {features}\n')
                X, y = get_data(language, task, label_type, features, features_dir, canonscores_dir, sentiscores_dir, metadata_dir)

                # Run grid search for nested cv
                cv_outer = CustomGroupKFold(n_splits=5, stratified=task_params['stratified']).split(X, y.values.ravel())
                X_ = X.copy(deep=True)
                y_ = y.copy(deep=True)

                for outer_fold, (train_idx, test_idx) in enumerate(cv_outer):
                    X_train_outer, X_test_outer = X_.iloc[train_idx], X_.iloc[test_idx]
                    y_train_outer, y_test_outer = y_.iloc[train_idx], y_.iloc[test_idx]
                    print(f'\nX_train_outer{X_train_outer.shape}, X_test_outer{X_test_outer.shape},  y_train_outer{y_train_outer.shape}, y_test_outer{y_test_outer.shape}')
                    
                    gridsearch_object = run_gridsearch(
                        gridsearch_dir=gridsearch_dir, 
                        language=language, 
                        task=task, 
                        label_type=label_type, 
                        features=features, 
                        fold=outer_fold, 
                        columns_list=columns_list, 
                        task_params=task_params, 
                        X_train=X_train_outer,
                        y_train=y_train_outer)

                    estimator = gridsearch_object.best_estimator_

                    y_pred = estimator.predict(X_test_outer)
                    y_pred = pd.DataFrame(y_pred, columns=['y_pred'])
                    y_pred['fold'] = outer_fold
                    y_pred.to_csv(
                        os.path.join(gridsearch_dir, f'y-pred_{language}_{task}_{label_type}_{features}_fold-{outer_fold}.csv'), 
                        index=False)
                    y_true = pd.DataFrame(y_test_outer).rename({'y': 'y_true'}, axis=1).reset_index().rename({'index': 'file_name'})
                    y_true['fold'] = outer_fold              
                    y_true.to_csv(
                        os.path.join(gridsearch_dir, f'y-true_{language}_{task}_{label_type}_{features}_fold-{outer_fold}.csv'), 
                        index=False)

 # %%
