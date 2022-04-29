from copy import deepcopy
import pandas as pd
import numpy as np
import random
random.seed(2)
from collections import Counter
from scipy.stats import pearsonr
from math import sqrt
import heapq
import statistics
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.linear_model import Lasso
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import xgboost
#  from xgboost import XGBRegressor

class AuthorSplit():
    '''
    Distribute book names over splits.
    All works of an author are in the same split.
    Adapted from https://www.titanwolf.org/Network/q/b7ee732a-7c92-4416-bc80-a2bd2ed136f1/y
    '''
    def __init__(self, language, df, nr_splits, seed, stratified=False, return_indices=False):
        self.language = language
        self.df = df
        self.nr_splits = nr_splits
        self.stratified = stratified
        self.return_indices = return_indices
        self.file_names = df['file_name']
        self.author_bookname_mapping, self.works_per_author = self.get_author_books()
        random.seed(seed)

    def get_author_books(self):
        authors = []
        author_bookname_mapping = {}
        #Get books per authors
        for file_name in self.file_names:
            author = '_'.join(file_name.split('_')[:2])
            authors.append(author)
            if author in author_bookname_mapping:
                author_bookname_mapping[author].append(file_name)
            else:
                author_bookname_mapping[author] = []
                author_bookname_mapping[author].append(file_name)
                
        # Aggregate if author has collaborations with others
        if self.language == 'ger':
            agg_dict = {'Hoffmansthal_Hugo': ['Hoffmansthal_Hugo-von'], 
                        'Schlaf_Johannes': ['Holz-Schlaf_Arno-Johannes'],
                         'Arnim_Bettina': ['Arnim-Arnim_Bettina-Gisela']}
        else:
            agg_dict = {'Stevenson_Robert-Louis': ['Stevenson-Grift_Robert-Louis-Fanny-van-de', 
                                                   'Stevenson-Osbourne_Robert-Louis-Lloyde']}
            
        for author, aliases in agg_dict.items():
            if author in authors:
                for alias in aliases:
                    if alias in authors:
                        author_bookname_mapping[author].extend(author_bookname_mapping[alias]) 
                        del author_bookname_mapping[alias]
                        authors = [author for author in authors if author != alias]
        
        works_per_author = Counter(authors)
        return author_bookname_mapping, works_per_author
    
    def split(self):
        splits = [[] for _ in range(0,self.nr_splits)]

        if self.stratified == True:
            rare_labels = sorted(self.df['y'].unique().tolist())[1:]
            splits_counter = [0 for _ in range(0,self.nr_splits)]
            for rare_label in rare_labels:
                # If stratified, first distribute authors that have rarest label over split so that the author is assigned to the split with the smallest number or rarest labels
                counts = [(0,i) for i in range (0,self.nr_splits)]
                # heapify based on first element of tuple, inplace
                heapq.heapify(counts)
                for author in list(self.works_per_author.keys()):
                    rare_label_counter = 0
                    for curr_file_name in self.author_bookname_mapping[author]:
                        if self.df.loc[self.df['file_name'] == curr_file_name].squeeze().at['y'] == rare_label:
                            rare_label_counter += 1
                    if rare_label_counter != 0:
                        author_workcount = self.works_per_author.pop(author)
                        count, index = heapq.heappop(counts)
                        splits[index].append(author)
                        splits_counter[index] += author_workcount
                        heapq.heappush(counts, (count + rare_label_counter, index))
            totals = [(splits_counter[i],i) for i in range(0,len(splits_counter))]
        else:
            totals = [(0,i) for i in range (0, self.nr_splits)]
        # heapify based on first element of tuple, inplace
        heapq.heapify(totals)
        while bool(self.works_per_author):
            author = random.choice(list(self.works_per_author.keys()))
            author_workcount = self.works_per_author.pop(author)
            # find split with smallest number of books
            total, index = heapq.heappop(totals)
            splits[index].append(author)
            heapq.heappush(totals, (total + author_workcount, index))

        if not self.return_indices:
            # Return file_names in splits
            #Map author splits to book names
            map_splits = []
            for split in splits:
                new = []
                for author in split:
                    new.extend(self.author_bookname_mapping[author])
                map_splits.append(new)

            if self.stratified == True:
                for split in map_splits:
                    split_df = self.df[self.df['file_name'].isin(split)]
        else:
            # Return indices of file_names in split
            file_name_idx_mapping = dict((file_name, index) for index, file_name in enumerate(self.file_names))
            map_splits = []
            for split in splits:
                test_split = []
                for author in split:
                    # Get all indices from file_names from the same author
                    test_split.extend([file_name_idx_mapping[file_name] for file_name in  self.author_bookname_mapping[author]])
                # Indices of all file_names that are not in split
                train_split = list(set(file_name_idx_mapping.values()) - set(test_split))
                map_splits.append((train_split, test_split))
        return map_splits


class Regression():
    '''Predict sentiment scores.'''
    def __init__(self, results_dir, language, task_type, model, model_param, labels_string, labels, features_string, df, dimensionality_reduction, drop_columns, verbose=True):

        self.results_dir = results_dir
        self.language = language
        self.task_type = task_type
        self.model = model
        self.model_param = model_param
        self.labels_string = labels_string
        self.labels = labels
        self.features_string = features_string
        self.df = df
        self.dimensionality_reduction = dimensionality_reduction
        self.drop_columns = drop_columns
        self.verbose = True
        self.model_info_string = f'{self.language}_{self.task_type}_{self.model}_param-{self.model_param}_label-{self.labels_string}_feat-{self.features_string}_dimred-{self.dimensionality_reduction}_drop-{len(self.drop_columns)}'

        assert isinstance(self.drop_columns, list)
        for i in self.drop_columns:
            assert isinstance(i, str)
        assert (self.dimensionality_reduction in ['k_best_f_reg_0_10', 'k_best_mutual_info_0_10', 'ss_pca_0_95']) or (self.dimensionality_reduction is None)
        self._check_class_specific_assertions()

        self.df = self._drop_columns()

    def _check_class_specific_assertions(self):
        assert self.model in ['xgboost', 'svr', 'lasso']
        assert self.features_string in ['book', 'chunk', 'baac', 'cacb']
        assert self.labels_string in ['textblob', 'sentiart', 'combined']
    
    def _drop_columns(self):
        def _drop_column(column):
            for string in self.drop_columns:
                if string in column:
                    return True
            return False

        columns_before_drop = set(self.df.columns)
        if self.drop_columns:
            df = self.df[[column for column in self.df.columns if not _drop_column(column)]].reset_index(drop=True)
        columns_after_drop = set(self.df.columns)
        if self.verbose:
            print(f'Dropped {len(columns_before_drop - columns_after_drop)} columns.') 
        return df
    
    def _custom_pca(self, train_X):
        for i in range(5, train_X.shape[1], int((train_X.shape[1] - 5) / 10)):
            pca = PCA(n_components=i)
            new_train_X = pca.fit_transform(train_X)
            if pca.explained_variance_ratio_.sum() >= 0.95:
                break
        return new_train_X, pca

    def _reduce_dimensions(self, train_X, train_y, validation_X):
        if self.dimensionality_reduction == 'ss_pca_0_95':
            ss = StandardScaler()
            train_X = ss.fit_transform(train_X)
            validation_X = ss.transform(validation_X)
            train_X, pca = self._custom_pca(train_X)
            validation_X = pca.transform(validation_X)
        elif self.dimensionality_reduction == 'k_best_f_reg_0_10':
            k_best = SelectKBest(f_regression, k=np.minimum(int(0.10 * train_X.shape[0]), train_X.shape[1]))
            train_X = k_best.fit_transform(train_X, train_y)
            validation_X = k_best.transform(validation_X)
        elif self.dimensionality_reduction == 'k_best_mutual_info_0_10':
            k_best = SelectKBest(mutual_info_regression, k=np.minimum(int(0.10 * train_X.shape[0]), train_X.shape[1]))
            train_X = k_best.fit_transform(train_X, train_y)
            validation_X = k_best.transform(validation_X)
        elif self.dimensionality_reduction is None:
            pass
        return train_X, validation_X
    
    def _impute(self, train_X, validation_X):
        imputer = KNNImputer()
        train_X = imputer.fit_transform(train_X)
        validation_X = imputer.transform(validation_X)
        return train_X, validation_X
    
    def _get_model(self, model_param, train_X=None, train_y=None, train_file_names=None, task_type=None):
        if self.model == 'xgboost':
            if task_type == 'binary_classification':
                is_classification = True
                class_weights = dict(enumerate(compute_class_weight('balanced', classes=[0, 1], y=train_y.astype(int).tolist())))
            elif task_type == 'multiclass_classification':
                is_classification = True
                class_weights = dict(enumerate(compute_class_weight('balanced', classes=[0, 1, 2, 3], y=train_y.astype(int).tolist())))
            elif task_type == 'regression':
                is_classification = False
            else:
                raise Exception('Not a valid task_type')
            
            def feval(preds, train_data):
                labels = train_data.get_label()
                if is_classification:
                    labels = labels.astype(int)
                    preds = preds.argmax(axis=1).astype(int)
                    if task_type == 'binary_classification':
                        return 'acc', accuracy_score(labels, preds)
                    elif task_type == 'multiclass_classification':
                        return 'f1', f1_score(labels, preds, average='macro')
                else:
                    return 'rmse', np.sqrt(mean_squared_error(labels, preds))
            
            if is_classification:
                dtrain = xgboost.DMatrix(train_X, label=train_y.astype(int), weight=[class_weights[int(i)] for i in train_y])
            else:
                dtrain = xgboost.DMatrix(train_X, label=train_y)
            results = []
            df = np.hstack((train_file_names, train_X))
            df = pd.DataFrame(df, columns=['file_name'] + [f'col_{i}' for i in range(train_X.shape[1])])
            for max_depth in [2, 4, 6, 8]:
                for learning_rate in [None, 0.01, 0.033, 0.1]:
                    for colsample_bytree in [0.33, 0.60, 0.75]:
                        if task_type == 'multiclass_classification':
                            params = {'max_depth': max_depth, 'learning_rate': learning_rate, 'colsample_bytree': colsample_bytree, 'n_jobs': -1, 'objective': 'multi:softmax', 'num_class': 4, 'eval_metric': 'mlogloss'}
                        elif task_type == 'binary_classification':
                            params = {'max_depth': max_depth, 'learning_rate': learning_rate, 'colsample_bytree': colsample_bytree, 'n_jobs': -1, 'objective': 'multi:softmax', 'num_class': 2, 'eval_metric': 'mlogloss'}
                        elif task_type == 'regression':
                            params = {'max_depth': max_depth, 'learning_rate': learning_rate, 'colsample_bytree': colsample_bytree, 'n_jobs': -1}
                        else:
                            raise Exception('Not a valid task_type')
                        cv_results = xgboost.cv(
                                        params,
                                        dtrain,
                                        num_boost_round=99999,
                                        seed=42,
                                        nfold=5,
                                        folds=AuthorSplit(language=self.language, df=df, nr_splits=5, seed=8, return_indices=True).split(),
                                        feval=feval,
                                        maximize=is_classification, # if classification, maximize f1/acc score.
                                        early_stopping_rounds=10,
                                        verbose_eval=False)

                        if task_type == 'binary_classification':
                            nested_cv_score = cv_results.iloc[len(cv_results)-1]['test-acc-mean']
                        elif task_type == 'multiclass_classification':
                            nested_cv_score = cv_results.iloc[len(cv_results)-1]['test-f1-mean']
                        elif task_type == 'regression':
                            nested_cv_score = cv_results.iloc[len(cv_results)-1]['test-rmse-mean']
                        else:
                            raise Exception('Not a valid task_type')
                        num_boost_round = len(cv_results)
                        if task_type == 'multiclass_classification':
                            results.append({'max_depth': max_depth, 'learning_rate': learning_rate, 'colsample_bytree': colsample_bytree, 'num_boost_round': num_boost_round, 'nested_cv_score': nested_cv_score, 'objective': 'multi:softmax', 'num_class': 4, 'eval_metric': 'mlogloss'})
                        elif task_type == 'binary_classification':
                            results.append({'max_depth': max_depth, 'learning_rate': learning_rate, 'colsample_bytree': colsample_bytree, 'num_boost_round': num_boost_round, 'nested_cv_score': nested_cv_score, 'objective': 'multi:softmax', 'num_class': 2, 'eval_metric': 'mlogloss'})
                        elif task_type == 'regression':
                            results.append({'max_depth': max_depth, 'learning_rate': learning_rate, 'colsample_bytree': colsample_bytree, 'num_boost_round': num_boost_round, 'nested_cv_score': nested_cv_score})
                        else:
                            raise Exception('Not a valid task_type')
            best_parameters = sorted(results, key=lambda x: x['nested_cv_score'], reverse=is_classification)[0]
            return best_parameters
        elif self.model == 'svr':
            return SVR(C=model_param)
        elif self.model == 'lasso':
            return Lasso(alpha=model_param)
        elif self.model == 'svc':
            return SVC(C=model_param, class_weight='balanced')
        
    
    def _get_pvalue(self, validation_corr_pvalues):
        # Harmonic mean p-value
        denominator = sum([1/x for x in validation_corr_pvalues])
        mean_p_value = len(validation_corr_pvalues)/denominator
        return mean_p_value
    
    def _combine_df_labels(self, df):
        #Average of sentiscores per book
        df = df.merge(right=self.labels, on='file_name', how='inner', validate='many_to_one')
        return df

    def _prepare_dfs(self, split, df):
        # Prapare data
        train_df = df[~df['file_name'].isin(split)]
        train_X = train_df.drop(columns=['y', 'file_name'], inplace=False).values
        train_y = train_df['y'].values.ravel()

        validation_df = df[df['file_name'].isin(split)]
        validation_X = validation_df.drop(columns=['y', 'file_name'], inplace=False).values

        train_X, validation_X = self._impute(train_X, validation_X)
        #if self.verbose:
        #    print(f'train_X.shape before {self.dimensionality_reduction}: {train_X.shape}, validation_X.shape before {self.dimensionality_reduction}: {validation_X.shape}')
        train_X, validation_X = self._reduce_dimensions(train_X, train_y, validation_X)
        #if self.verbose:
        #    print(f'train_X.shape after {self.dimensionality_reduction}: {train_X.shape}, validation_X.shape after {self.dimensionality_reduction}: {validation_X.shape}')

        train_labels = deepcopy(train_df[['file_name', 'y']])
        validation_labels = deepcopy(validation_df[['file_name', 'y']])
        #if self.verbose:
            #print('Class distribution over train and validation set :', train_df['y'].value_counts()'\n', validation_df['y'].value_counts())
        return train_df, train_X, train_y, validation_X, train_labels, validation_labels
    
    def _concat_and_save_examples(self,all_validation_labels):
        all_validation_labels = pd.concat(all_validation_labels)
        all_validation_labels.to_csv(f'{self.results_dir}examples-{self.model_info_string}.csv', index=False)
        return all_validation_labels

    def run(self):
        # List of all predicted values/all true labels, for plotting
        all_predictions = []
        all_labels = []
        # List of dfs with file_name, y, yhat
        all_validation_labels = []

        train_mses = []
        train_maes = []
        train_r2s = []
        train_corrs = []
        
        validation_mses = []
        validation_maes = []
        validation_r2s = []
        validation_corrs = []
        validation_corr_pvalues = []

        df = self.df
        print('df', df)
        df = self._combine_df_labels(df)
        file_names_split = AuthorSplit(language=self.language, df=df, nr_splits=10, seed=1, return_indices=False).split()
        
        for index, split in enumerate(file_names_split):
            train_df, train_X, train_y, validation_X, train_labels, validation_labels = self._prepare_dfs(split, df)
            # Train model
            if self.model == 'xgboost':
                train_file_names = train_df['file_name'].values.reshape(-1, 1)
                best_parameters = self._get_model(self.model_param, train_X, train_y, train_file_names, task_type='regression')
                dtrain = xgboost.DMatrix(train_X, label=train_y)
                num_boost_round = best_parameters['num_boost_round']
                best_parameters.pop('nested_cv_score')
                best_parameters.pop('num_boost_round')
                model = xgboost.train(best_parameters,
                                      dtrain,
                                      num_boost_round=num_boost_round,
                                      verbose_eval=False)
            else:
                model = self._get_model(self.model_param)
                model.fit(train_X, train_y)
            
            # Predict           
            if self.model == 'xgboost':
                train_labels['yhat'] = model.predict(xgboost.DMatrix(train_X))
                validation_labels['yhat'] = model.predict(xgboost.DMatrix(validation_X))
            
            else:
                train_labels['yhat'] = model.predict(train_X)
                validation_labels['yhat'] = model.predict(validation_X)
            
            # Evaluate
            train_labels = train_labels.groupby('file_name').mean()
            validation_labels = validation_labels.groupby('file_name').mean()
            all_validation_labels.append(validation_labels.reset_index())
            
            train_y = train_labels['y'].tolist()
            train_yhat = train_labels['yhat'].tolist()
            validation_y = validation_labels['y'].tolist()
            validation_yhat = validation_labels['yhat'].tolist()
            
            # list of values
            all_labels.extend(validation_y)
            all_predictions.extend(validation_yhat)
            
            train_mse = mean_squared_error(train_y, train_yhat)
            train_mae = mean_absolute_error(train_y, train_yhat)
            train_r2 = r2_score(train_y, train_yhat)
            train_corr = pearsonr(train_y, train_yhat)[0]
            
            validation_mse = mean_squared_error(validation_y, validation_yhat)
            validation_mae = mean_absolute_error(validation_y, validation_yhat)
            validation_r2 = r2_score(validation_y, validation_yhat)
            validation_corr, p_value = pearsonr(validation_y, validation_yhat)
            
            train_mses.append(train_mse)
            train_maes.append(train_mae)
            train_r2s.append(train_r2)
            train_corrs.append(train_corr)
            
            validation_mses.append(validation_mse)
            validation_maes.append(validation_mae)
            validation_r2s.append(validation_r2)
            validation_corrs.append(validation_corr)
            validation_corr_pvalues.append(p_value)
            
            if self.verbose:
                print(f'Fold: {index+1}, TrainMSE: {np.round(train_mse, 3)}, TrainMAE: {np.round(train_mae, 3)}, ValMSE: {np.round(validation_mse, 3)}, ValMAE: {np.round(validation_mae, 3)}, ValR2: {np.round(validation_r2, 3)}, ValCorr: {np.round(validation_corr, 3)}, ValPValue: {np.round(p_value, 3)}')
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        
        # Save file_names with y and yhat for analyzing results for each book
        _ = self._concat_and_save_examples(all_validation_labels)
        
        mean_train_mse = np.mean(train_mses)
        mean_train_rmse = np.mean([sqrt(x) for x in train_mses])
        mean_train_mae = np.mean(train_maes)
        mean_train_r2 = np.mean(train_r2s)
        mean_train_corr = np.mean(train_corrs)
        
        mean_validation_mse = np.mean(validation_mses)
        mean_validation_rmse = np.mean([sqrt(x) for x in validation_mses])
        mean_validation_mae = np.mean(validation_maes)
        mean_validation_r2 = np.mean(validation_r2s)
        mean_validation_corr = np.mean(validation_corrs)
        mean_p_value = self._get_pvalue(validation_corr_pvalues)
        
        if self.verbose:
            print(f'''TrainMSE: {np.round(mean_train_mse, 3)}, 
                TrainRMSE: {np.round(mean_train_rmse, 3)}, 
                TrainMAE: {np.round(mean_train_mae, 3)}, 
                TrainR2: {np.round(mean_train_r2, 3)}, 
                TrainCorr: {np.round(mean_train_corr, 3)}, 
                ValMSE: {np.round(mean_validation_mse, 3)}, 
                ValRMSE: {np.round(mean_validation_rmse, 3)}, 
                ValMAE: {np.round(mean_validation_mae, 3)}, 
                ValR2: {np.round(mean_validation_r2, 3)}, 
                ValCorr: {np.round(mean_validation_corr, 3)}, 
                ValCorrPValue: {np.round(mean_p_value, 3)}''')
   
            if self.language == 'eng':
                color = 'm'
            else:
                color = 'teal'
            plt.figure(figsize=(4,4))
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            #plt.xlim([min(all_labels) - 0.01, max(all_labels) + 0.01])
            #plt.ylim([min(all_predictions) - 0.01, max(all_predictions) + 0.01])
            plt.xticks(np.arange(round(min(all_labels),2) - 0.01, round(max(all_labels),2) + 0.01, 0.03))
            plt.scatter(x=all_labels, y=all_predictions, s=6, c=color)
            plt.xlabel('True Scores', fontsize=15)
            plt.ylabel('Predicted Scores', fontsize=15)
            plt.savefig(f'{self.results_dir}{self.model_info_string}.png', dpi=400, bbox_inches='tight')
            plt.show();

        returned_values = [mean_train_mse, 
            mean_train_rmse, 
            mean_train_mae, 
            mean_train_r2, 
            mean_train_corr, 
            mean_validation_mse, 
            mean_validation_rmse, 
            mean_validation_mae, 
            mean_validation_r2, 
            mean_validation_corr, 
            mean_p_value]
        return [round(x, 3) for x in returned_values]


class TwoclassClassification(Regression):
    ''' Classify into reviewed/not reviewed.'''
    def __init__(self, results_dir, language, task_type, model, model_param, labels_string, labels, features_string, df, dimensionality_reduction, drop_columns, verbose=True):
        super().__init__(results_dir, language, task_type, model, model_param, labels_string, labels, features_string, df, dimensionality_reduction, drop_columns, verbose=True)

    def _check_class_specific_assertions(self):
        assert self.model in ['svc', 'xgboost']
        assert self.features_string in ['book', 'baac']
        assert self.labels_string in ['twoclass', 'library']
        
    def _combine_df_labels(self, df):
        #Reviews zum englischen Korpus beginnnen mit 1759 und decken alles bis 1914 ab
        df = df.merge(right=self.labels, on='file_name', how='left', validate='many_to_one')
        df['y'] = df['y'].fillna(value=0)
        #Select books written after year of first review)
        book_year = df['file_name'].str.replace('-', '_').str.split('_').str[-1].astype('int64')
        review_year = book_year[df['y'] != 0]
        # Keep only those texts that that were published after the first review had appeared
        df = df.loc[book_year>=min(review_year)]
        return df
    
    def _get_accuracy(self, df):
        return [accuracy_score(df['y'], df['yhat']), balanced_accuracy_score(df['y'], df['yhat'])] 

    def _make_crosstabs(self, all_validation_labels):
        crosstab = pd.crosstab(all_validation_labels['y'], all_validation_labels['yhat'], rownames=['True'], colnames=['Predicted'], margins=True)
        crosstab.to_csv(f'{self.results_dir}crosstab_{self.model_info_string}.csv', index=True)
        print('--------------------------\nCrosstab\n', crosstab, '\n--------------------------')
                    
    def run(self):
        train_accs = []
        train_balanced_accs = []
        validation_accs = []
        validation_balanced_accs = []
        df = self.df
        df = self._combine_df_labels(df)
        file_names_split_stratified = AuthorSplit(language=self.language, df=df, nr_splits=5, seed=1, stratified=True, return_indices=False).split()
        all_validation_labels = []

        for index, split in enumerate(file_names_split_stratified):
            train_df, train_X, train_y, validation_X, train_labels, validation_labels = self._prepare_dfs(split, df)
            
            # Train model
            if self.model == 'xgboost':
                train_file_names = train_df['file_name'].values.reshape(-1, 1)
                best_parameters = self._get_model(self.model_param, train_X, train_y, train_file_names, task_type='binary_classification')
                class_weights = dict(enumerate(compute_class_weight('balanced', classes=[0, 1], y=train_y.astype(int).tolist())))
                dtrain = xgboost.DMatrix(train_X, label=train_y.astype(int), weight=[class_weights[int(i)] for i in train_y])
                num_boost_round = best_parameters['num_boost_round']
                best_parameters.pop('nested_cv_score')
                best_parameters.pop('num_boost_round')
                model = xgboost.train(best_parameters,
                                      dtrain,
                                      num_boost_round=num_boost_round,
                                      verbose_eval=False)
            else:
                model = self._get_model(self.model_param)
                model.fit(train_X, train_y)
            
            # Predict           
            if self.model == 'xgboost':
                train_labels['yhat'] = model.predict(xgboost.DMatrix(train_X))
                validation_labels['yhat'] = model.predict(xgboost.DMatrix(validation_X))
            else:
                train_labels['yhat'] = model.predict(train_X)
                validation_labels['yhat'] = model.predict(validation_X)
            # list of dfs
            all_validation_labels.append(validation_labels)

            # Evaluate
            train_acc, train_balanced_acc = self._get_accuracy(train_labels)
            validation_acc, validation_balanced_acc = self._get_accuracy(validation_labels)
            
            train_accs.append(train_acc)
            train_balanced_accs.append(train_balanced_acc)
            validation_accs.append(validation_acc)
            validation_balanced_accs.append(validation_balanced_acc)
        
        # Save y and y_pred for examples
        all_validation_labels = self._concat_and_save_examples(all_validation_labels)

        self._make_crosstabs(all_validation_labels)

        mean_train_acc = statistics.mean(train_accs)
        mean_train_balanced_acc = statistics.mean(train_balanced_accs)
        mean_validation_acc = statistics.mean(validation_accs)
        mean_validation_balanced_acc = statistics.mean(validation_balanced_accs)

        return [round(mean_train_acc, 3), round(mean_train_balanced_acc, 3), round(mean_validation_acc, 3), round(mean_validation_balanced_acc, 3)]

class MulticlassClassification(TwoclassClassification):
    '''Classify into not reviewed/negative/not classified/positive.'''
    def __init__(self, results_dir, language, task_type, model, model_param, labels_string, labels, features_string, df, dimensionality_reduction, drop_columns, verbose=True):
        super().__init__(results_dir, language, task_type, model, model_param, labels_string, labels, features_string, df, dimensionality_reduction, drop_columns, verbose=True)

    def _check_class_specific_assertions(self):
        assert self.model in ['svc', 'xgboost']
        assert self.features_string in ['book', 'baac']
        assert self.labels_string in ['multiclass']
                
    def _get_f1_score(self, df):
        score = f1_score(df['y'], df['yhat'], average='macro')
        return score
        
    def run(self):
        train_f1s = []
        validation_f1s = []

        df = self.df
        df = self._combine_df_labels(df)
        file_names_split_stratified = AuthorSplit(language=self.language, df=df, nr_splits=5, seed=1, stratified=True, return_indices=False).split()
        all_validation_labels = []

        for index, split in enumerate(file_names_split_stratified):
            train_df, train_X, train_y, validation_X, train_labels, validation_labels = self._prepare_dfs(split, df)
    
            if self.model == 'xgboost':
                train_file_names = train_df['file_name'].values.reshape(-1, 1)
                best_parameters = self._get_model(self.model_param, train_X, train_y, train_file_names, task_type='multiclass_classification')
                class_weights = dict(enumerate(compute_class_weight('balanced', classes=[0, 1, 2, 3], y=train_y.astype(int).tolist())))
                dtrain = xgboost.DMatrix(train_X, label=train_y.astype(int), weight=[class_weights[int(i)] for i in train_y])
                num_boost_round = best_parameters['num_boost_round']
                best_parameters.pop('nested_cv_score')
                best_parameters.pop('num_boost_round')
                model = xgboost.train(best_parameters,
                                      dtrain,
                                      num_boost_round=num_boost_round,
                                      verbose_eval=False)
            else:
                model = self._get_model(self.model_param)
                model.fit(train_X, train_y)
            
            if self.model == 'xgboost':
                train_labels['yhat'] = model.predict(xgboost.DMatrix(train_X))
                validation_labels['yhat'] = model.predict(xgboost.DMatrix(validation_X))
            else:
                train_labels['yhat'] = model.predict(train_X)
                validation_labels['yhat'] = model.predict(validation_X)
            all_validation_labels.append(validation_labels)

            train_f1 = self._get_f1_score(train_labels)
            train_f1s.append(train_f1)
            validation_f1 = self._get_f1_score(validation_labels)           
            validation_f1s.append(validation_f1)
            if self.verbose:
                print(f'Fold: {index+1}, TrainF1: {np.round(train_f1, 3)}, ValF1: {np.round(validation_f1, 3)}')
        
        # Save y and yhat for examples
        all_validation_labels = self._concat_and_save_examples(all_validation_labels)

        self._make_crosstabs(all_validation_labels)

        mean_train_f1 = statistics.mean(train_f1s)
        mean_validation_f1 = statistics.mean(validation_f1s)
        
        if self.verbose:
            print(f'''TrainF1: {np.round(mean_train_f1, 3)}, ValidationF1: {np.round(mean_validation_f1, 3)}''')
            print('\n---------------------------------------------------\n')
        return [round(mean_train_f1, 3), round(mean_validation_f1, 3)]
