import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score


## Part 1: read/load data

def read_data(fn, filetype = "csv"):
    if filetype == "csv":
        return pd.read_csv(fn)
    if filetype == "excel":
        return pd.read_excel(fn)
    if filetype == "sql":
        return pd.read_sql(fn, con=conn)
    else:
        return print("I only have CSVs at the moment!")

## Part 2: explore data

def take_sample(df, fraction):
    return df.sample(frac = fraction)

def show_columns(df):
    return df.columns

def descrip_stats(df):
    return df.describe()

def counts_per_variable(df, x):
	return df.groupby(x).size()

def group_and_describe(df, x):
	return df.groupby(x).describe()

def ctab_percent(df, x, y):
	return pd.crosstab(df.loc[:, x], df.loc[:,y], normalize='index')

def ctab_raw(df, x, y):
	return pd.crosstab(df.loc[:, x], df.loc[:,y])

def basic_hist(df, x, title_text):
	sns.distplot(df[x]).set_title(title_text)
	plt.show()
	return

def basic_scatter(df, x, y, title_text):
	g = sns.lmplot(x, y, data= df)
	g = (g.set_axis_labels(x, y).set_title(title_text))
	plt.show()
	return

def correlation_heatmap(df, title_text):
	corrmat = df.corr()
	f, ax = plt.subplots(figsize=(12, 9))
	sns.heatmap(corrmat, vmax=.8, square=True).set_title(title_text)
	plt.show()
	return

def basic_boxplot(df, colname, title_text):
    sns.boxplot(y=df[colname]).set_title(title_text)
    plt.show()
    return

## Part III: Pre-processing data

def show_nulls(df):
	return df.isna().sum().sort_values(ascending=False)

def fill_whole_df_with_mean(df):
    num_cols = len(df.columns)
    for i in range(0, num_cols):
        df.iloc[:,i] = fill_col_with_mean(df.iloc[:,i])
    return

def fill_allNA_mode(df):
    num_col = len(df.columns.tolist())
    for i in range(0,num_col):
        df_feats.iloc[:,i] = df_feats.iloc[:,i].fillna(df_feats.iloc[:,i].mode()[0])
    return df

def fill_col_with_mean(df):
	return df.fillna(df.mean())

def fill_mean(df, colname):
    return df[colname].fillna(np.mean(df[colname]))

def left_merge(df_left, df_right, merge_column):
    return pd.merge(df_left, df_right, how = 'left', on = merge_column)

# generating features

def generate_dummy(df, colname, attach = False):
    # generate dummy variables from a categorical variable
    # if attach == True, then attach the dummy variables to the original dataframe
    if (attach == False):
        return pd.get_dummies(df[colname])
    else:
        return pd.concat([df, pd.get_dummies(df[colname])], axis = 1)

def discret_eqlbins(df, colname, bin_num):
    # cut continuous variable into bin_num bins
    return pd.cut(df[colname], bin_num)

def discret_quantiles(df, colname, quantile_num):
    # cut cont. variable into quantiles
    return pd.qcut(df[colname], quantile_num)

# feature-scaling

from sklearn import preprocessing
#min_max_scaler = preprocessing.MinMaxScaler()
#df_scaled = min_max_scaler.fit_transform(df)

# standardize data

# scaled_column = scale(df[['x','y']])
from sklearn.preprocessing import scale

def scale_df(df, features_list):
    temp_scaled = scale(df[features_list])
    #return a DF
    return pd.DataFrame(temp_scaled, columns= df.columns)

def scale_column(df, colname):
    scaled_col = scale(df[colname])
    return pd.DataFrame(scaled_col, columns = [colname])

# split data into training and test sets
from sklearn.model_selection import train_test_split
def split_traintest(df_features, df_target, test_size = 0.2):
    X_train, X_test, Y_train, Y_test = train_test_split(df_features, df_target, test_size = test_size)
    return X_train, X_test, Y_train, Y_test

# methods for training classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

def fit_randomforest(x_train, y_train, feature_number, num_trees, depth_num, criterion_choice):
    rf_clf = RandomForestClassifier(max_features = feature_number, n_estimators = num_trees, max_depth = depth_num, criterion = criterion_choice)
    rf_clf.fit(x_train,y_train)
    return rf_clf

def fit_svm(x_train, y_train, c_value, kern, rbf_gam):
    svm_clf = SVC(C = c_value, kernel = kern, gamma = rbf_gam, probability = True)
    svm_clf.fit(x_train, y_train)
    return svm_clf

def fit_naivebayes(x_train, y_train, alpha_value):
    nb_clf = MultinomialNB(alpha = alpha_value)
    nb_clf.fit(x_train,y_train)
    return nb_clf

def fit_knn(x_train, y_train, neighbor_num, distance_type, weight_type):
    knn_clf = KNeighborsClassifier(n_neighbors= neighbor_num, metric= distance_type, weights = weight_type)
    knn_clf.fit(x_train, y_train)
    return knn_clf

def fit_dtree(x_train, y_train, crit_par, split_par, maxdepth_par, minsplit_par,maxfeat_par, minleaf_par, maxleaf_par):
    dt_clf = DecisionTreeClassifier(criterion = crit_par, splitter = split_par, max_depth = maxdepth_par, min_samples_split = minsplit_par, max_features = maxfeat_par, min_samples_leaf = minleaf_par, max_leaf_nodes = maxleaf_par)
    dt_clf.fit(x_train, y_train)
    return dt_clf

def fit_logit(x_train, y_train, penalty_para, c_para):
    logit_clf = LogisticRegression(penalty = penalty_para, C = c_para)
    logit_clf.fit(x_train,y_train)
    return logit_clf

# grid methods

from sklearn.model_selection import GridSearchCV
def grid_cv(clf, param_grid, scoring, cv, x_train, y_train):
    # initialize the grid, scoring = a scoring metric or a dictionary of metrics, 
    # refit is necessary when u have a list of scoring metrics, it determines how the gridsearch algorithm decides the best estimator.
    grid = GridSearchCV(clf(), param_grid, scoring = scoring, cv= cv)
    grid.fit(x_train, y_train)

    # call the best classifier:
    grid.best_estimator_

    # see all performances:
    return grid

def grid_cv_mtp(clf, param_grid, scoring, cv = 5, refit_metric = 'roc'):
    # initialize the grid, scoring = a scoring metric or a dictionary of metrics, 
    # refit is necessary when u have a list of scoring metrics, it determines how the gridsearch algorithm decides the best estimator.
    grid = GridSearchCV(clf(), param_grid, scoring = scoring, cv= cv_num, refit = refit_metric)
    grid.fit(x_train, y_train)

    # call the best classifier:
    grid.best_estimator_

    # see all performances:
    return grid


def classifier_comparison(model_params, x_train, y_train, eva_metric, cv_num):
    comparison_results = {}
    for model, param_grid in model_params.items():
        # initialize gridsearch object
        grid = GridSearchCV(clf(), param_grid, scoring = eva_metric, cv= cv_num)
        grid.fit(x_train, y_train)
        comparison_results[model] ={}
        comparison_results[model]['cv_results'] = grid.cv_results_
        comparison_results[model]['best_estimator'] = grid.best_estimator_
        comparison_results[model]['best_score'] = grid.best_score_
        comparison_results[model]['best_params'] = grid.best_params_
    return comparison_results

## Part VI: Evaluating the classifier

#generate predictions according to a custom threshold
def make_predictions(clf, x_test, threshold = 0.7):
    # threshold = the probability threshold for something to be a 0.
    # generate array with predicted probabilities
    pred_array = clf.predict_proba(x_test)

    # initialize an empty array for the predictions
    pred_generated = np.array([])

    # predict the first entry
    if pred_array[0][0] >= threshold:
        pred_generated = np.hstack([pred_generated, 0])
    else:
        pred_generated = np.hstack([pred_generated, 1])

    # loops over the rest of the array
    for i in range(1,len(x_test)):
        if pred_array[i][0] >= threshold:
            pred_generated = np.vstack([pred_generated, 0])
        else:
            pred_generated = np.vstack([pred_generated, 1])

    # return an np.array
    return  pred_generated

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

def evaluateAccuracy(clf,predictDF, truthDF):
    correct_pred = 0
    pred_x = clf.predict(predictDF)
    for i in range(0,len(predictDF)):
        if pred_x[i] == truthDF.iloc[i]:
            correct_pred +=1
    return (correct_pred/len(predictDF))

# temporal validation
from dateutil import parser
def create_datetime(df, colname):
    # creates a new column with datetimem objects
    return df[colname].apply(parser.parse)

def retrieve_year(df, date_column):
    return df[date_column].map(lambda x: x.year)

def retrieve_month(df, date_column):
    return df[date_column].map(lambda x: x.month)

def retrieve_day(df, date_column):
    return df[date_column].map(lambda x: x.day)

model_params ={
    RandomForestClassifier: {
    'max_features': ["auto", "sqrt", "log2", 0.2], 
    'n_estimators' : [50, 100,  500,1000, 10000], 
    "max_depth": [3,5,8], 
    "criterion": ["gini", "entropy"],
    'min_samples_split': [2,5,10]
    },
    SVC:{
    "C": [0.00001,0.0001,0.001,0.01,0.1,1,10],
    "kernel":["linear", "rbf"],
    "gamma": [10**i for i in np.arange(0, 1, 0.05)]
    },
    MultinomialNB:{
    "alpha": [1, 5, 10, 25, 100]
    },
    KNeighborsClassifier:{
    "n_neighbors":[1,5,10,25,50,100],
    "metric": ["euclidean", "manhattan", "chebyshev" ],
    "weights":["uniform", "distance"]
    },
    DecisionTreeClassifier:{
    "criterion": ["gini", "entropy"],
    "splitter": ["best", "random"],
    "max_features": [None, "auto", "sqrt", "log2", 5, 0.3 ],
    "min_samples_split": [2, 5, 7, 9 ,15 ,20],
    "max_depth": [1,5,10,20,50,100],
    "min_samples_leaf": [1,2,3,4,5], 
    "max_leaf_nodes": [None, 2, 3 ,4, 5]
    },
    LogisticRegression:{
    "penalty": ['l1', 'l2'],
    "C": [10**-5, 10**-2, 10**-1, 1, 10, 10**2, 10**5]
    },
    GradientBoostingClassifier:{
    'loss': ["deviance", "exponential"], 
    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3, 0.5], 
    'n_estimators': [1,10,100,1000,10000],
    'max_depth': [1,3,5,10,20,50,100]
    }
}

tiny_model_params ={
    RandomForestClassifier: {
    'max_features': ["auto"], 
    'n_estimators' : [50, 1000], 
    "max_depth": [3,8], 
    "criterion": ["gini"],
    'min_samples_split': [2,5]
    },
    SVC:{
    "C": [0.001,0.01],
    "kernel":["linear", "rbf"],
    "gamma": [0.1, 1, 10]
    },
    MultinomialNB:{
    "alpha": [1, 5]
    },
    KNeighborsClassifier:{
    "n_neighbors":[25,50],
    "metric": ["euclidean" ],
    "weights":["uniform", "distance"]
    },
    DecisionTreeClassifier:{
    "criterion": ["gini", "entropy"],
    "splitter": ["best", "random"],
    "max_features": ["auto"],
    "min_samples_split": [2, 5],
    "max_depth": [1,5,10,20],
    "min_samples_leaf": [4,5], 
    "max_leaf_nodes": [4, 5]
    },
    LogisticRegression:{
    "penalty": ['l1'],
    "C": [10**-1, 1, 10]
    },
    GradientBoostingClassifier:{
    'loss': ["deviance"], 
    'learning_rate': [0.1, 0.2], 
    'n_estimators': [1,10,100,],
    'max_depth': [10,20]
    }
}


import pdb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

def xy_cv_small(main_grid, xcv1_train, xcv1_test, xcv2_train, xcv2_test, xcv3_train, xcv3_test, ycv1_train, ycv1_test, ycv2_train, ycv2_test, ycv3_train, ycv3_test):
    # initialize results dictionary
    results = {}
    prob_thresholds = [0.5]
    for clf_i in main_grid:

        #initialize parameter grid
        param_grid =  main_grid[clf_i]
        # store results for each classifier type
        results[clf_i] = {}

        # Random Forests
        loop_num = 0
        if clf_i == RandomForestClassifier:
            for criterion_par in param_grid['criterion']:
                for depth_par in param_grid['max_depth']:
                    for max_feat_par in param_grid['max_features']:
                        for samp_split_par in param_grid['min_samples_split']:
                            for n_est_par in param_grid['n_estimators']:
                                for prob in prob_thresholds:
                                    # intialize the main dictionary key
                                    model_key = ("RF",criterion_par, depth_par, max_feat_par, samp_split_par, n_est_par, prob)
                                    results[clf_i][model_key] = {}
                                    sub_loops = 0
                                    # for CV1
                                    try:
                                        rf_clf = clf_i(max_features = max_feat_par, 
                                            min_samples_split = samp_split_par,
                                            n_estimators = n_est_par, 
                                            max_depth = depth_par, 
                                            criterion = criterion_par)
                                        rf_clf.fit(xcv1_train, ycv1_train)
                                        y_pred = make_predictions(rf_clf, threshold = prob, x_test = xcv1_test)
                                        # generate metrics
                                        prec1 = precision_score(ycv1_test, y_pred)
                                        rec1 = recall_score(ycv1_test, y_pred)
                                        acc1 = accuracy_score(ycv1_test, y_pred)
                                        roc1 = roc_auc_score(ycv1_test, y_pred)
                                        f1_score1 = f1_score(ycv1_test, y_pred)
                                        sub_loops += 1

                                    except:
                                        pdb.set_trace()

                                    # for CV2
                                    try:
                                        rf_clf2 = clf_i(max_features = max_feat_par, 
                                            min_samples_split = samp_split_par,
                                            n_estimators = n_est_par, 
                                            max_depth = depth_par, 
                                            criterion = criterion_par)
                                        rf_clf2.fit(xcv2_train, ycv2_train)
                                        y_pred = make_predictions(rf_clf2, threshold = prob, x_test = xcv2_test)
                                        # generate metrics
                                        prec2 = precision_score(ycv2_test, y_pred)
                                        rec2 = recall_score(ycv2_test, y_pred)
                                        acc2 = accuracy_score(ycv2_test, y_pred)
                                        roc2 = roc_auc_score(ycv2_test, y_pred)
                                        f1_score2 = f1_score(ycv2_test, y_pred)
                                        sub_loops += 1

                                    except:
                                        pdb.set_trace()

                                    # for CV3
                                    try:
                                        rf_clf3 = clf_i(max_features = max_feat_par, 
                                            min_samples_split = samp_split_par,
                                            n_estimators = n_est_par, 
                                            max_depth = depth_par, 
                                            criterion = criterion_par)
                                        rf_clf3.fit(xcv3_train, ycv3_train)
                                        y_pred = make_predictions(rf_clf3, threshold = prob, x_test = xcv3_test)
                                        # generate metrics
                                        prec3 = precision_score(ycv3_test, y_pred)
                                        rec3 = recall_score(ycv3_test, y_pred)
                                        acc3 = accuracy_score(ycv3_test, y_pred)
                                        roc3 = roc_auc_score(ycv3_test, y_pred)
                                        f1_score3 = f1_score(ycv3_test, y_pred)
                                        sub_loops += 1

                                    except:
                                        pdb.set_trace()

                                    results[clf_i]['Precision']= (prec1 + prec2 + prec3)/3
                                    results[clf_i]['Recall']= (rec1 + rec2 + rec3)/3
                                    results[clf_i]['AUC Score'] = (roc1 + roc2 + roc3)/3
                                    results[clf_i]['Accuracy'] = (acc1 + acc2 + acc3)/3
                                    results[clf_i]['F1 Score'] = (f1_score1 + f1_score2 + f1_score3)/3
                                    loop_num += 1

        if clf_i == SVC:
            for c_par in param_grid['C']:
                for kernel_par in param_grid['kernel']:
                    for gamma_par in param_grid['gamma']:
                        for prob in prob_thresholds:
                            # intialize the main dictionary key
                            model_key = ("SVM", c_par, kernel_par, gamma_par, prob)
                            results[clf_i][model_key] = {}
                            sub_loops = 0
                            # for CV1
                            try:
                                svm_clf = clf_i(C = c_par, 
                                    kernel = kernel_par,
                                    gamma = gamma_par, 
                                    probability = True)
                                svm_clf.fit(xcv1_train, ycv1_train)
                                y_pred = make_predictions(svm_clf, threshold = prob, x_test = xcv1_test)
                                # generate metrics
                                prec1 = precision_score(ycv1_test, y_pred)
                                rec1 = recall_score(ycv1_test, y_pred)
                                acc1 = accuracy_score(ycv1_test, y_pred)
                                roc1 = roc_auc_score(ycv1_test, y_pred)
                                f1_score1 = f1_score(ycv1_test, y_pred)
                                sub_loops += 1

                            except:
                                pdb.set_trace()

                            # for CV2
                            try:
                                svm_clf2 = clf_i(C = c_par, 
                                    kernel = kernel_par,
                                    gamma = gamma_par,
                                    probability = True)
                                svm_clf2.fit(xcv2_train, ycv2_train)
                                y_pred = make_predictions(svm_clf2, threshold = prob, x_test = xcv2_test)
                                # generate metrics
                                prec2 = precision_score(ycv2_test, y_pred)
                                rec2 = recall_score(ycv2_test, y_pred)
                                acc2 = accuracy_score(ycv2_test, y_pred)
                                roc2 = roc_auc_score(ycv2_test, y_pred)
                                f1_score2 = f1_score(ycv2_test, y_pred)
                                sub_loops += 1

                            except:
                                pdb.set_trace()

                            # for CV3
                            try:
                                svm_clf3 = clf_i(C = c_par, 
                                    kernel = kernel_par,
                                    gamma = gamma_par,
                                    probability = True)
                                svm_clf3.fit(xcv3_train, ycv3_train)
                                y_pred = make_predictions(svm_clf3, threshold = prob, x_test = xcv3_test)
                                # generate metrics
                                prec3 = precision_score(ycv3_test, y_pred)
                                rec3 = recall_score(ycv3_test, y_pred)
                                acc3 = accuracy_score(ycv3_test, y_pred)
                                roc3 = roc_auc_score(ycv3_test, y_pred)
                                f1_score3 = f1_score(ycv3_test, y_pred)
                                sub_loops += 1

                            except:
                                pdb.set_trace()

                            results[clf_i]['Precision']= (prec1 + prec2 + prec3)/3
                            results[clf_i]['Recall']= (rec1 + rec2 + rec3)/3
                            results[clf_i]['AUC Score'] = (roc1 + roc2 + roc3)/3
                            results[clf_i]['Accuracy'] = (acc1 + acc2 + acc3)/3
                            results[clf_i]['F1 Score'] = (f1_score1 + f1_score2 + f1_score3)/3
                            loop_num += 1

        if clf_i == MultinomialNB:
            for alpha_par in param_grid['alpha']:
                for prob in prob_thresholds:
                    # intialize the main dictionary key
                    model_key = ("NB", alpha_par, prob)
                    results[clf_i][model_key] = {}
                    sub_loops = 0
                    # for CV1
                    try:
                        nb_clf = clf_i(alpha= alpha_par)
                        nb_clf.fit(xcv1_train, ycv1_train)
                        y_pred = make_predictions(nb_clf, threshold = prob, x_test = xcv1_test)
                        # generate metrics
                        prec1 = precision_score(ycv1_test, y_pred)
                        rec1 = recall_score(ycv1_test, y_pred)
                        acc1 = accuracy_score(ycv1_test, y_pred)
                        roc1 = roc_auc_score(ycv1_test, y_pred)
                        f1_score1 = f1_score(ycv1_test, y_pred)
                        sub_loops += 1

                    except:
                        pdb.set_trace()

                    # for CV2
                    try:
                        nb_clf2 = clf_i(alpha= alpha_par)
                        nb_clf2.fit(xcv2_train, ycv2_train)
                        y_pred = make_predictions(nb_clf2, threshold = prob, x_test = xcv2_test)
                        # generate metrics
                        prec2 = precision_score(ycv2_test, y_pred)
                        rec2 = recall_score(ycv2_test, y_pred)
                        acc2 = accuracy_score(ycv2_test, y_pred)
                        roc2 = roc_auc_score(ycv2_test, y_pred)
                        f1_score2 = f1_score(ycv2_test, y_pred)
                        sub_loops += 1

                    except:
                        pdb.set_trace()

                    # for CV3
                    try:
                        nb_clf3 = clf_i(alpha= alpha_par)
                        nb_clf3.fit(xcv3_train, ycv3_train)
                        y_pred = make_predictions(nb_clf3, threshold = prob, x_test = xcv3_test)
                        # generate metrics
                        prec3 = precision_score(ycv3_test, y_pred)
                        rec3 = recall_score(ycv3_test, y_pred)
                        acc3 = accuracy_score(ycv3_test, y_pred)
                        roc3 = roc_auc_score(ycv3_test, y_pred)
                        f1_score3 = f1_score(ycv3_test, y_pred)
                        sub_loops += 1

                    except:
                        pdb.set_trace()

                    results[clf_i]['Precision']= (prec1 + prec2 + prec3)/3
                    results[clf_i]['Recall']= (rec1 + rec2 + rec3)/3
                    results[clf_i]['AUC Score'] = (roc1 + roc2 + roc3)/3
                    results[clf_i]['Accuracy'] = (acc1 + acc2 + acc3)/3
                    results[clf_i]['F1 Score'] = (f1_score1 + f1_score2 + f1_score3)/3
                    loop_num += 1

        if clf_i == KNeighborsClassifier:
            for n_par in param_grid['n_neighbors']:
                for metric_par in param_grid['metric']:
                    for weights_par in param_grid['weights']:
                        for prob in prob_thresholds:
                            # intialize the main dictionary key
                            model_key = ("KNN",n_par, metric_par, weights_par, prob)
                            results[clf_i][model_key] = {}
                            sub_loops = 0
                            # for CV1
                            try:
                                knn_clf = clf_i(n_neighbors = n_par, metric = metric_par, weights = weights_par)
                                knn_clf.fit(xcv1_train, ycv1_train)
                                y_pred = make_predictions(knn_clf, threshold = prob, x_test = xcv1_test)
                                # generate metrics
                                prec1 = precision_score(ycv1_test, y_pred)
                                rec1 = recall_score(ycv1_test, y_pred)
                                acc1 = accuracy_score(ycv1_test, y_pred)
                                roc1 = roc_auc_score(ycv1_test, y_pred)
                                f1_score1 = f1_score(ycv1_test, y_pred)
                                sub_loops += 1

                            except:
                                pdb.set_trace()

                            # for CV2
                            try:
                                knn_clf2 = clf_i(n_neighbors = n_par, metric = metric_par, weights = weights_par)
                                knn_clf2.fit(xcv2_train, ycv2_train)
                                y_pred = make_predictions(knn_clf2, threshold = prob, x_test = xcv2_test)
                                # generate metrics
                                prec2 = precision_score(ycv2_test, y_pred)
                                rec2 = recall_score(ycv2_test, y_pred)
                                acc2 = accuracy_score(ycv2_test, y_pred)
                                roc2 = roc_auc_score(ycv2_test, y_pred)
                                f1_score2 = f1_score(ycv2_test, y_pred)
                                sub_loops += 1

                            except:
                                pdb.set_trace()

                            # for CV3
                            try:
                                knn_clf3 = clf_i(n_neighbors = n_par, metric = metric_par, weights = weights_par)
                                knn_clf3.fit(xcv3_train, ycv3_train)
                                y_pred = make_predictions(knn_clf3, threshold = prob, x_test = xcv3_test)
                                # generate metrics
                                prec3 = precision_score(ycv3_test, y_pred)
                                rec3 = recall_score(ycv3_test, y_pred)
                                acc3 = accuracy_score(ycv3_test, y_pred)
                                roc3 = roc_auc_score(ycv3_test, y_pred)
                                f1_score3 = f1_score(ycv3_test, y_pred)
                                sub_loops += 1

                            except:
                                pdb.set_trace()

                            results[clf_i]['Precision']= (prec1 + prec2 + prec3)/3
                            results[clf_i]['Recall']= (rec1 + rec2 + rec3)/3
                            results[clf_i]['AUC Score'] = (roc1 + roc2 + roc3)/3
                            results[clf_i]['Accuracy'] = (acc1 + acc2 + acc3)/3
                            results[clf_i]['F1 Score'] = (f1_score1 + f1_score2 + f1_score3)/3
                            loop_num += 1


        if clf_i == DecisionTreeClassifier:
            for criterion_par in param_grid['criterion']:
                for depth_par in param_grid['max_depth']:
                    for max_feat_par in param_grid['max_features']:
                        for samp_split_par in param_grid['min_samples_split']:
                            for splitter_par in param_grid['splitter']:
                                for min_leaf_par in param_grid['min_samples_leaf']:
                                    for max_leaf_par in param_grid['max_leaf_nodes']:
                                        for prob in prob_thresholds:
                                            # intialize the main dictionary key
                                            model_key = ("DT", criterion_par, depth_par, max_feat_par, samp_split_par, splitter_par, min_leaf_par, max_leaf_par, prob)
                                            results[clf_i][model_key] = {}
                                            sub_loops = 0
                                            # for CV1
                                            try:
                                                dt_clf = clf_i(max_features = max_feat_par, 
                                                    min_samples_split = samp_split_par,
                                                    splitter = splitter_par,
                                                    max_depth = depth_par, 
                                                    criterion = criterion_par,
                                                    min_samples_leaf = min_leaf_par,
                                                    max_leaf_nodes =  max_leaf_par)
                                                dt_clf.fit(xcv1_train, ycv1_train)
                                                y_pred = make_predictions(dt_clf, threshold = prob, x_test = xcv1_test)
                                                # generate metrics
                                                prec1 = precision_score(ycv1_test, y_pred)
                                                rec1 = recall_score(ycv1_test, y_pred)
                                                acc1 = accuracy_score(ycv1_test, y_pred)
                                                roc1 = roc_auc_score(ycv1_test, y_pred)
                                                f1_score1 = f1_score(ycv1_test, y_pred)
                                                sub_loops += 1

                                            except:
                                                pdb.set_trace()

                                            # for CV2
                                            try:
                                                dt_clf2 = clf_i(max_features = max_feat_par, 
                                                    min_samples_split = samp_split_par,
                                                    splitter = splitter_par,
                                                    max_depth = depth_par, 
                                                    criterion = criterion_par,
                                                    min_samples_leaf = min_leaf_par,
                                                    max_leaf_nodes =  max_leaf_par)
                                                dt_clf2.fit(xcv2_train, ycv2_train)
                                                y_pred = make_predictions(dt_clf2, threshold = prob, x_test = xcv2_test)
                                                # generate metrics
                                                prec2 = precision_score(ycv2_test, y_pred)
                                                rec2 = recall_score(ycv2_test, y_pred)
                                                acc2 = accuracy_score(ycv2_test, y_pred)
                                                roc2 = roc_auc_score(ycv2_test, y_pred)
                                                f1_score2 = f1_score(ycv2_test, y_pred)
                                                sub_loops += 1

                                            except:
                                                pdb.set_trace()

                                            # for CV3
                                            try:
                                                dt_clf3 = clf_i(max_features = max_feat_par, 
                                                    min_samples_split = samp_split_par,
                                                    splitter = splitter_par,
                                                    max_depth = depth_par, 
                                                    criterion = criterion_par,
                                                    min_samples_leaf = min_leaf_par,
                                                    max_leaf_nodes =  max_leaf_par)
                                                dt_clf3.fit(xcv3_train, ycv3_train)
                                                y_pred = make_predictions(dt_clf3, threshold = prob, x_test = xcv3_test)
                                                # generate metrics
                                                prec3 = precision_score(ycv3_test, y_pred)
                                                rec3 = recall_score(ycv3_test, y_pred)
                                                acc3 = accuracy_score(ycv3_test, y_pred)
                                                roc3 = roc_auc_score(ycv3_test, y_pred)
                                                f1_score3 = f1_score(ycv3_test, y_pred)
                                                sub_loops += 1

                                            except:
                                                pdb.set_trace()

                                            results[clf_i]['Precision']= (prec1 + prec2 + prec3)/3
                                            results[clf_i]['Recall']= (rec1 + rec2 + rec3)/3
                                            results[clf_i]['AUC Score'] = (roc1 + roc2 + roc3)/3
                                            results[clf_i]['Accuracy'] = (acc1 + acc2 + acc3)/3
                                            results[clf_i]['F1 Score'] = (f1_score1 + f1_score2 + f1_score3)/3
                                            loop_num += 1

        if clf_i == LogisticRegression:
            for penalty_par in param_grid['penalty']:
                for c_par in param_grid['C']:
                    for prob in prob_thresholds:
                        # intialize the main dictionary key
                        model_key = ("LR", penalty_par, c_par, prob)
                        results[clf_i][model_key] = {}
                        sub_loops = 0
                        # for CV1
                        try:
                            lr_clf = clf_i(penalty = penalty_par, C = c_par)
                            lr_clf.fit(xcv1_train, ycv1_train)
                            y_pred = make_predictions(lr_clf, threshold = prob, x_test = xcv1_test)
                            # generate metrics
                            prec1 = precision_score(ycv1_test, y_pred)
                            rec1 = recall_score(ycv1_test, y_pred)
                            acc1 = accuracy_score(ycv1_test, y_pred)
                            roc1 = roc_auc_score(ycv1_test, y_pred)
                            f1_score1 = f1_score(ycv1_test, y_pred)
                            sub_loops += 1

                        except:
                            pdb.set_trace()

                        # for CV2
                        try:
                            lr_clf2 = clf_i(penalty = penalty_par, C = c_par)
                            lr_clf2.fit(xcv2_train, ycv2_train)
                            y_pred = make_predictions(lr_clf2, threshold = prob, x_test = xcv2_test)
                            # generate metrics
                            prec2 = precision_score(ycv2_test, y_pred)
                            rec2 = recall_score(ycv2_test, y_pred)
                            acc2 = accuracy_score(ycv2_test, y_pred)
                            roc2 = roc_auc_score(ycv2_test, y_pred)
                            f1_score2 = f1_score(ycv2_test, y_pred)
                            sub_loops += 1

                        except:
                            pdb.set_trace()

                        # for CV3
                        try:
                            lr_clf3 = clf_i(penalty = penalty_par, C = c_par)
                            lr_clf3.fit(xcv3_train, ycv3_train)
                            y_pred = make_predictions(lr_clf3, threshold = prob, x_test = xcv3_test)
                            # generate metrics
                            prec3 = precision_score(ycv3_test, y_pred)
                            rec3 = recall_score(ycv3_test, y_pred)
                            acc3 = accuracy_score(ycv3_test, y_pred)
                            roc3 = roc_auc_score(ycv3_test, y_pred)
                            f1_score3 = f1_score(ycv3_test, y_pred)
                            sub_loops += 1

                        except:
                            pdb.set_trace()

                        results[clf_i]['Precision']= (prec1 + prec2 + prec3)/3
                        results[clf_i]['Recall']= (rec1 + rec2 + rec3)/3
                        results[clf_i]['AUC Score'] = (roc1 + roc2 + roc3)/3
                        results[clf_i]['Accuracy'] = (acc1 + acc2 + acc3)/3
                        results[clf_i]['F1 Score'] = (f1_score1 + f1_score2 + f1_score3)/3
                        loop_num += 1
    return results


def xy_cv(main_grid, xcv1_train, xcv1_test, xcv2_train, xcv2_test, xcv3_train, xcv3_test, ycv1_train, ycv1_test, ycv2_train, ycv2_test, ycv3_train, ycv3_test):
    # initialize results dictionary
    results = {}
    prob_thresholds = [0.05, 0.1, 0.2, 0.25, 0.3, 0.5, 0.7, 0.9]
    for clf_i in main_grid:

        #initialize parameter grid
        param_grid =  main_grid[clf_i]
        # store results for each classifier type
        results[clf_i] = {}

        # Random Forests
        loop_num = 0
        if clf_i == RandomForestClassifier:
            for criterion_par in param_grid['criterion']:
                for depth_par in param_grid['max_depth']:
                    for max_feat_par in param_grid['max_features']:
                        for samp_split_par in param_grid['min_samples_split']:
                            for n_est_par in param_grid['n_estimators']:
                                for prob in prob_thresholds:
                                    # intialize the main dictionary key
                                    model_key = ("RF",criterion_par, depth_par, max_feat_par, samp_split_par, n_est_par, prob)
                                    results[clf_i][model_key] = {}
                                    sub_loops = 0
                                    # for CV1
                                    try:
                                        rf_clf = clf_i(max_features = max_feat_par, 
                                            min_samples_split = samp_split_par,
                                            n_estimators = n_est_par, 
                                            max_depth = depth_par, 
                                            criterion = criterion_par)
                                        rf_clf.fit(xcv1_train, ycv1_train)
                                        y_pred = make_predictions(rf_clf, threshold = prob, x_test = xcv1_test)
                                        # generate metrics
                                        prec1 = precision_score(ycv1_test, y_pred)
                                        rec1 = recall_score(ycv1_test, y_pred)
                                        acc1 = accuracy_score(ycv1_test, y_pred)
                                        roc1 = roc_auc_score(ycv1_test, y_pred)
                                        f1_score1 = f1_score(ycv1_test, y_pred)
                                        sub_loops += 1

                                    except:
                                        pdb.set_trace()

                                    # for CV2
                                    try:
                                        rf_clf2 = clf_i(max_features = max_feat_par, 
                                            min_samples_split = samp_split_par,
                                            n_estimators = n_est_par, 
                                            max_depth = depth_par, 
                                            criterion = criterion_par)
                                        rf_clf2.fit(xcv2_train, ycv2_train)
                                        y_pred = make_predictions(rf_clf2, threshold = prob, x_test = xcv2_test)
                                        # generate metrics
                                        prec2 = precision_score(ycv2_test, y_pred)
                                        rec2 = recall_score(ycv2_test, y_pred)
                                        acc2 = accuracy_score(ycv2_test, y_pred)
                                        roc2 = roc_auc_score(ycv2_test, y_pred)
                                        f1_score2 = f1_score(ycv2_test, y_pred)
                                        sub_loops += 1

                                    except:
                                        pdb.set_trace()

                                    # for CV3
                                    try:
                                        rf_clf3 = clf_i(max_features = max_feat_par, 
                                            min_samples_split = samp_split_par,
                                            n_estimators = n_est_par, 
                                            max_depth = depth_par, 
                                            criterion = criterion_par)
                                        rf_clf3.fit(xcv3_train, ycv3_train)
                                        y_pred = make_predictions(rf_clf3, threshold = prob, x_test = xcv3_test)
                                        # generate metrics
                                        prec3 = precision_score(ycv3_test, y_pred)
                                        rec3 = recall_score(ycv3_test, y_pred)
                                        acc3 = accuracy_score(ycv3_test, y_pred)
                                        roc3 = roc_auc_score(ycv3_test, y_pred)
                                        f1_score3 = f1_score(ycv3_test, y_pred)
                                        sub_loops += 1

                                    except:
                                        pdb.set_trace()

                                    results[clf_i]['Precision']= (prec1 + prec2 + prec3)/3
                                    results[clf_i]['Recall']= (rec1 + rec2 + rec3)/3
                                    results[clf_i]['AUC Score'] = (roc1 + roc2 + roc3)/3
                                    results[clf_i]['Accuracy'] = (acc1 + acc2 + acc3)/3
                                    results[clf_i]['F1 Score'] = (f1_score1 + f1_score2 + f1_score3)/3
                                    loop_num += 1

        if clf_i == SVC:
            for c_par in param_grid['C']:
                for kernel_par in param_grid['kernel']:
                    for gamma_par in param_grid['gamma']:
                        for prob in prob_thresholds:
                            # intialize the main dictionary key
                            model_key = ("SVM", c_par, kernel_par, gamma_par, prob)
                            results[clf_i][model_key] = {}
                            sub_loops = 0
                            # for CV1
                            try:
                                svm_clf = clf_i(C = c_par, 
                                    kernel = kernel_par,
                                    gamma = gamma_par,
                                    probability = True)
                                svm_clf.fit(xcv1_train, ycv1_train)
                                y_pred = make_predictions(svm_clf, threshold = prob, x_test = xcv1_test)
                                # generate metrics
                                prec1 = precision_score(ycv1_test, y_pred)
                                rec1 = recall_score(ycv1_test, y_pred)
                                acc1 = accuracy_score(ycv1_test, y_pred)
                                roc1 = roc_auc_score(ycv1_test, y_pred)
                                f1_score1 = f1_score(ycv1_test, y_pred)
                                sub_loops += 1

                            except:
                                pdb.set_trace()

                            # for CV2
                            try:
                                svm_clf2 = clf_i(C = c_par, 
                                    kernel = kernel_par,
                                    gamma = gamma_par,
                                    probability = True)
                                svm_clf2.fit(xcv2_train, ycv2_train)
                                y_pred = make_predictions(svm_clf2, threshold = prob, x_test = xcv2_test)
                                # generate metrics
                                prec2 = precision_score(ycv2_test, y_pred)
                                rec2 = recall_score(ycv2_test, y_pred)
                                acc2 = accuracy_score(ycv2_test, y_pred)
                                roc2 = roc_auc_score(ycv2_test, y_pred)
                                f1_score2 = f1_score(ycv2_test, y_pred)
                                sub_loops += 1

                            except:
                                pdb.set_trace()

                            # for CV3
                            try:
                                svm_clf3 = clf_i(C = c_par, 
                                    kernel = kernel_par,
                                    gamma = gamma_par,
                                    probability = True)
                                svm_clf3.fit(xcv3_train, ycv3_train)
                                y_pred = make_predictions(svm_clf3, threshold = prob, x_test = xcv3_test)
                                # generate metrics
                                prec3 = precision_score(ycv3_test, y_pred)
                                rec3 = recall_score(ycv3_test, y_pred)
                                acc3 = accuracy_score(ycv3_test, y_pred)
                                roc3 = roc_auc_score(ycv3_test, y_pred)
                                f1_score3 = f1_score(ycv3_test, y_pred)
                                sub_loops += 1

                            except:
                                pdb.set_trace()

                            results[clf_i]['Precision']= (prec1 + prec2 + prec3)/3
                            results[clf_i]['Recall']= (rec1 + rec2 + rec3)/3
                            results[clf_i]['AUC Score'] = (roc1 + roc2 + roc3)/3
                            results[clf_i]['Accuracy'] = (acc1 + acc2 + acc3)/3
                            results[clf_i]['F1 Score'] = (f1_score1 + f1_score2 + f1_score3)/3
                            loop_num += 1

        if clf_i == MultinomialNB:
            for alpha_par in param_grid['alpha']:
                for prob in prob_thresholds:
                    # intialize the main dictionary key
                    model_key = ("NB", alpha_par, prob)
                    results[clf_i][model_key] = {}
                    sub_loops = 0
                    # for CV1
                    try:
                        nb_clf = clf_i(alpha= alpha_par)
                        nb_clf.fit(xcv1_train, ycv1_train)
                        y_pred = make_predictions(nb_clf, threshold = prob, x_test = xcv1_test)
                        # generate metrics
                        prec1 = precision_score(ycv1_test, y_pred)
                        rec1 = recall_score(ycv1_test, y_pred)
                        acc1 = accuracy_score(ycv1_test, y_pred)
                        roc1 = roc_auc_score(ycv1_test, y_pred)
                        f1_score1 = f1_score(ycv1_test, y_pred)
                        sub_loops += 1

                    except:
                        pdb.set_trace()

                    # for CV2
                    try:
                        nb_clf2 = clf_i(alpha= alpha_par)
                        nb_clf2.fit(xcv2_train, ycv2_train)
                        y_pred = make_predictions(nb_clf2, threshold = prob, x_test = xcv2_test)
                        # generate metrics
                        prec2 = precision_score(ycv2_test, y_pred)
                        rec2 = recall_score(ycv2_test, y_pred)
                        acc2 = accuracy_score(ycv2_test, y_pred)
                        roc2 = roc_auc_score(ycv2_test, y_pred)
                        f1_score2 = f1_score(ycv2_test, y_pred)
                        sub_loops += 1

                    except:
                        pdb.set_trace()

                    # for CV3
                    try:
                        nb_clf3 = clf_i(alpha= alpha_par)
                        nb_clf3.fit(xcv3_train, ycv3_train)
                        y_pred = make_predictions(nb_clf3, threshold = prob, x_test = xcv3_test)
                        # generate metrics
                        prec3 = precision_score(ycv3_test, y_pred)
                        rec3 = recall_score(ycv3_test, y_pred)
                        acc3 = accuracy_score(ycv3_test, y_pred)
                        roc3 = roc_auc_score(ycv3_test, y_pred)
                        f1_score3 = f1_score(ycv3_test, y_pred)
                        sub_loops += 1

                    except:
                        pdb.set_trace()

                    results[clf_i]['Precision']= (prec1 + prec2 + prec3)/3
                    results[clf_i]['Recall']= (rec1 + rec2 + rec3)/3
                    results[clf_i]['AUC Score'] = (roc1 + roc2 + roc3)/3
                    results[clf_i]['Accuracy'] = (acc1 + acc2 + acc3)/3
                    results[clf_i]['F1 Score'] = (f1_score1 + f1_score2 + f1_score3)/3
                    loop_num += 1

        if clf_i == KNeighborsClassifier:
            for n_par in param_grid['n_neighbors']:
                for metric_par in param_grid['metric']:
                    for weights_par in param_grid['weights']:
                        for prob in prob_thresholds:
                            # intialize the main dictionary key
                            model_key = ("KNN",n_par, metric_par, weights_par, prob)
                            results[clf_i][model_key] = {}
                            sub_loops = 0
                            # for CV1
                            try:
                                knn_clf = clf_i(n_neighbors = n_par, metric = metric_par, weights = weights_par)
                                knn_clf.fit(xcv1_train, ycv1_train)
                                y_pred = make_predictions(knn_clf, threshold = prob, x_test = xcv1_test)
                                # generate metrics
                                prec1 = precision_score(ycv1_test, y_pred)
                                rec1 = recall_score(ycv1_test, y_pred)
                                acc1 = accuracy_score(ycv1_test, y_pred)
                                roc1 = roc_auc_score(ycv1_test, y_pred)
                                f1_score1 = f1_score(ycv1_test, y_pred)
                                sub_loops += 1

                            except:
                                pdb.set_trace()

                            # for CV2
                            try:
                                knn_clf2 = clf_i(n_neighbors = n_par, metric = metric_par, weights = weights_par)
                                knn_clf2.fit(xcv2_train, ycv2_train)
                                y_pred = make_predictions(knn_clf2, threshold = prob, x_test = xcv2_test)
                                # generate metrics
                                prec2 = precision_score(ycv2_test, y_pred)
                                rec2 = recall_score(ycv2_test, y_pred)
                                acc2 = accuracy_score(ycv2_test, y_pred)
                                roc2 = roc_auc_score(ycv2_test, y_pred)
                                f1_score2 = f1_score(ycv2_test, y_pred)
                                sub_loops += 1

                            except:
                                pdb.set_trace()

                            # for CV3
                            try:
                                knn_clf3 = clf_i(n_neighbors = n_par, metric = metric_par, weights = weights_par)
                                knn_clf3.fit(xcv3_train, ycv3_train)
                                y_pred = make_predictions(knn_clf3, threshold = prob, x_test = xcv3_test)
                                # generate metrics
                                prec3 = precision_score(ycv3_test, y_pred)
                                rec3 = recall_score(ycv3_test, y_pred)
                                acc3 = accuracy_score(ycv3_test, y_pred)
                                roc3 = roc_auc_score(ycv3_test, y_pred)
                                f1_score3 = f1_score(ycv3_test, y_pred)
                                sub_loops += 1

                            except:
                                pdb.set_trace()

                            results[clf_i]['Precision']= (prec1 + prec2 + prec3)/3
                            results[clf_i]['Recall']= (rec1 + rec2 + rec3)/3
                            results[clf_i]['AUC Score'] = (roc1 + roc2 + roc3)/3
                            results[clf_i]['Accuracy'] = (acc1 + acc2 + acc3)/3
                            results[clf_i]['F1 Score'] = (f1_score1 + f1_score2 + f1_score3)/3
                            loop_num += 1


        if clf_i == DecisionTreeClassifier:
            for criterion_par in param_grid['criterion']:
                for depth_par in param_grid['max_depth']:
                    for max_feat_par in param_grid['max_features']:
                        for samp_split_par in param_grid['min_samples_split']:
                            for splitter_par in param_grid['splitter']:
                                for min_leaf_par in param_grid['min_samples_leaf']:
                                    for max_leaf_par in param_grid['max_leaf_nodes']:
                                        for prob in prob_thresholds:
                                            # intialize the main dictionary key
                                            model_key = ("DT", criterion_par, depth_par, max_feat_par, samp_split_par, splitter_par, min_leaf_par, max_leaf_par, prob)
                                            results[clf_i][model_key] = {}
                                            sub_loops = 0
                                            # for CV1
                                            try:
                                                dt_clf = clf_i(max_features = max_feat_par, 
                                                    min_samples_split = samp_split_par,
                                                    splitter = splitter_par,
                                                    max_depth = depth_par, 
                                                    criterion = criterion_par,
                                                    min_samples_leaf = min_leaf_par,
                                                    max_leaf_nodes =  max_leaf_par)
                                                dt_clf.fit(xcv1_train, ycv1_train)
                                                y_pred = make_predictions(dt_clf, threshold = prob, x_test = xcv1_test)
                                                # generate metrics
                                                prec1 = precision_score(ycv1_test, y_pred)
                                                rec1 = recall_score(ycv1_test, y_pred)
                                                acc1 = accuracy_score(ycv1_test, y_pred)
                                                roc1 = roc_auc_score(ycv1_test, y_pred)
                                                f1_score1 = f1_score(ycv1_test, y_pred)
                                                sub_loops += 1

                                            except:
                                                pdb.set_trace()

                                            # for CV2
                                            try:
                                                dt_clf2 = clf_i(max_features = max_feat_par, 
                                                    min_samples_split = samp_split_par,
                                                    splitter = splitter_par,
                                                    max_depth = depth_par, 
                                                    criterion = criterion_par,
                                                    min_samples_leaf = min_leaf_par,
                                                    max_leaf_nodes =  max_leaf_par)
                                                dt_clf2.fit(xcv2_train, ycv2_train)
                                                y_pred = make_predictions(dt_clf2, threshold = prob, x_test = xcv2_test)
                                                # generate metrics
                                                prec2 = precision_score(ycv2_test, y_pred)
                                                rec2 = recall_score(ycv2_test, y_pred)
                                                acc2 = accuracy_score(ycv2_test, y_pred)
                                                roc2 = roc_auc_score(ycv2_test, y_pred)
                                                f1_score2 = f1_score(ycv2_test, y_pred)
                                                sub_loops += 1

                                            except:
                                                pdb.set_trace()

                                            # for CV3
                                            try:
                                                dt_clf3 = clf_i(max_features = max_feat_par, 
                                                    min_samples_split = samp_split_par,
                                                    splitter = splitter_par,
                                                    max_depth = depth_par, 
                                                    criterion = criterion_par,
                                                    min_samples_leaf = min_leaf_par,
                                                    max_leaf_nodes =  max_leaf_par)
                                                dt_clf3.fit(xcv3_train, ycv3_train)
                                                y_pred = make_predictions(dt_clf3, threshold = prob, x_test = xcv3_test)
                                                # generate metrics
                                                prec3 = precision_score(ycv3_test, y_pred)
                                                rec3 = recall_score(ycv3_test, y_pred)
                                                acc3 = accuracy_score(ycv3_test, y_pred)
                                                roc3 = roc_auc_score(ycv3_test, y_pred)
                                                f1_score3 = f1_score(ycv3_test, y_pred)
                                                sub_loops += 1

                                            except:
                                                pdb.set_trace()

                                            results[clf_i]['Precision']= (prec1 + prec2 + prec3)/3
                                            results[clf_i]['Recall']= (rec1 + rec2 + rec3)/3
                                            results[clf_i]['AUC Score'] = (roc1 + roc2 + roc3)/3
                                            results[clf_i]['Accuracy'] = (acc1 + acc2 + acc3)/3
                                            results[clf_i]['F1 Score'] = (f1_score1 + f1_score2 + f1_score3)/3
                                            loop_num += 1

        if clf_i == LogisticRegression:
            for penalty_par in param_grid['penalty']:
                for c_par in param_grid['C']:
                    for prob in prob_thresholds:
                        # intialize the main dictionary key
                        model_key = ("LR", penalty_par, c_par, prob)
                        results[clf_i][model_key] = {}
                        sub_loops = 0
                        # for CV1
                        try:
                            lr_clf = clf_i(penalty = penalty_par, C = c_par)
                            lr_clf.fit(xcv1_train, ycv1_train)
                            y_pred = make_predictions(lr_clf, threshold = prob, x_test = xcv1_test)
                            # generate metrics
                            prec1 = precision_score(ycv1_test, y_pred)
                            rec1 = recall_score(ycv1_test, y_pred)
                            acc1 = accuracy_score(ycv1_test, y_pred)
                            roc1 = roc_auc_score(ycv1_test, y_pred)
                            f1_score1 = f1_score(ycv1_test, y_pred)
                            sub_loops += 1

                        except:
                            pdb.set_trace()

                        # for CV2
                        try:
                            lr_clf2 = clf_i(penalty = penalty_par, C = c_par)
                            lr_clf2.fit(xcv2_train, ycv2_train)
                            y_pred = make_predictions(lr_clf2, threshold = prob, x_test = xcv2_test)
                            # generate metrics
                            prec2 = precision_score(ycv2_test, y_pred)
                            rec2 = recall_score(ycv2_test, y_pred)
                            acc2 = accuracy_score(ycv2_test, y_pred)
                            roc2 = roc_auc_score(ycv2_test, y_pred)
                            f1_score2 = f1_score(ycv2_test, y_pred)
                            sub_loops += 1

                        except:
                            pdb.set_trace()

                        # for CV3
                        try:
                            lr_clf3 = clf_i(penalty = penalty_par, C = c_par)
                            lr_clf3.fit(xcv3_train, ycv3_train)
                            y_pred = make_predictions(lr_clf3, threshold = prob, x_test = xcv3_test)
                            # generate metrics
                            prec3 = precision_score(ycv3_test, y_pred)
                            rec3 = recall_score(ycv3_test, y_pred)
                            acc3 = accuracy_score(ycv3_test, y_pred)
                            roc3 = roc_auc_score(ycv3_test, y_pred)
                            f1_score3 = f1_score(ycv3_test, y_pred)
                            sub_loops += 1

                        except:
                            pdb.set_trace()

                        results[clf_i]['Precision']= (prec1 + prec2 + prec3)/3
                        results[clf_i]['Recall']= (rec1 + rec2 + rec3)/3
                        results[clf_i]['AUC Score'] = (roc1 + roc2 + roc3)/3
                        results[clf_i]['Accuracy'] = (acc1 + acc2 + acc3)/3
                        results[clf_i]['F1 Score'] = (f1_score1 + f1_score2 + f1_score3)/3
                        loop_num += 1
    return results
