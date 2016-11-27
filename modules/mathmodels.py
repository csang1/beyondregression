import uuid
import hashlib
import random
import math
import collections
from nlib import fit_least_squares, LINEAR, E
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy.stats as stats
from scipy.stats import ttest_ind
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
from sklearn.preprocessing import Imputer

class SmartTable(object):
    def __init__(self, *keys):
        self.keys = keys
        self.rows = []
    def append(self, item):
        self.rows.append([item[key] for key in self.keys])
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, k):
        return dict(zip(self.keys, self.rows[k]))
    def column(self, key, notnull=False):
        index = self.keys.index(key)
        column = [row[index] for row in self.rows]
        if notnull:
            column = filter(lambda x: x is not None, column)
        return column
    def next(self):
        for k in range(len(self.rows)):
            yield self[k]
    def purge_null(self, percent=0.10, filter_rows=True):
        new_table = SmartTable()
        n = (1.0-percent)*len(self)
        keys = [key for key in self.keys if len(self.column(key,notnull=True)) > n]
        indexes = map(self.keys.index, keys)
        new_rows = [[row[i] for i in indexes] for row in self.rows]
        new_table.keys = keys         
        if filter_rows:
            new_rows = filter(lambda row: not None in row, new_rows)
        new_table.rows = new_rows
        return new_table


def get_info(filename):
    df = pd.read_csv(filename)
    return list(df.columns.values), len(df)

def run_models(fullname, choose):
    folder, filename = os.path.split(fullname)
    folder = folder.replace('/uploads','/static')
    a = choose
    df = pd.read_csv(fullname)
    y_data = np.array(df.loc[:,a])
    X_picked = df.drop(a,1)
    imp = Imputer() 
    X_picked = imp.fit_transform(X_picked)
    y_data = imp.fit_transform(y_data)
    y_data = np.reshape(y_data, (len(df),))
    X_picked = preprocessing.minmax_scale(X_picked,feature_range=(0,1))
    y_data = preprocessing.minmax_scale(y_data,feature_range=(0,1))

    def add_layer(x, in_size, out_size, activation_function,layer,dropout):
        num_layer = 'layer%s' % layer
        with tf.name_scope(num_layer):
            global W
            tf.set_random_seed(666)
            W = tf.Variable(tf.random_uniform([in_size, out_size]))
        with tf.name_scope('y'):
            output = tf.matmul(x, W)
            if activation_function is 1:
                out = tf.nn.softsign(output)
            elif activation_function is 0:
                out = tf.nn.dropout(output,dropout)
            else:
                out = output
        return out

    # Create input variable
    with tf.name_scope('Input'):
        x = tf.placeholder(tf.float32,[None,9])
        y_ = tf.placeholder(tf.float32,[None,1])
        keep_prob = tf.placeholder(tf.float32) #dropout probability
        dropout = 0.85

    # Build up layers
    with tf.name_scope('Layer'):
         y= add_layer(x,9,1,2,1,dropout)

    # input
    header = list(df.columns.values)
    list_header = []
    for i in range(len(header)):
        if header[i] != a:
            list_header.append(header[i])

    # cost functiontensoflow
    with tf.name_scope('Cost'):
        cost = tf.reduce_mean(tf.square(y_-y))

    # training
    Training = tf.train.AdamOptimizer(0.01)
    train = Training.minimize(cost)

    with tf.Session() as sess:
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        feed = {}
        chi_sum = 0
        predict = []
        actual = []
        chi_plot = []
        i_count=[]
        counter = 0
        df1 = np.float32(df)
        for i in xrange(len(df.values)):
            #xs = np.array(np.float32([[df.iloc[i][headers] for headers in list_header]]))/1000
            #ys = np.array([[np.float32(df.iloc[i][a])]])/1000
            xs = np.array(np.float32([X_picked[i]]))
            ys = np.array(np.float32([[y_data[i]]]))
            feed={x:xs,y_:ys,keep_prob:dropout}
            prediction = sess.run(y,feed_dict=feed)
            sess.run(train,feed_dict=feed)
            chi = (np.square(ys-prediction))/(prediction)
            chi_sum += chi
            predict.append(float(prediction)*1000)
            actual.append(float(ys)*1000)
            chi_plot.append(float(chi))
            i_count.append(i)
    weight = np.array(sess.run(W)).flatten()
    r2_score_NN = r2_score(actual, predict)

    
    # scikit learn machine learning
    # Ridge Regression
    model1 = Ridge(alpha=1.0)
    model1.fit(X_picked, y_data)
    prediction_ridge = cross_val_predict(model1, X_picked, y_data, cv=10)
    r2_score_LR = r2_score(y_data, prediction_ridge)
    r_linear_ridge = stats.pearsonr(prediction_ridge,y_data)
    prediction_ridge = np.array(prediction_ridge)*1000
    ridge_coef = model1.coef_
    
    # Lasso Regression
    clf = linear_model.LassoLarsIC(criterion='aic')
    clf.fit(X_picked, y_data)
    prediction_lasso = cross_val_predict(clf, X_picked, y_data, cv=10)
    r2_score_lasso_aic = r2_score(y_data, prediction_lasso)
    r_linear_lassoAIC = stats.pearsonr(prediction_lasso,y_data)
    prediction_lasso = np.array(prediction_lasso)*1000
    lasso_coef = clf.coef_
    
    
    # Stats and Plot
    c= stats.pearsonr(actual, predict)
    plt.figure()
    plt.scatter(predict,actual,s=3)
    # plt.xlim((0,1000))
    # plt.ylim((0,1000))
    plt.xlabel('Prediction')
    plt.ylabel('Observation')
    plt.title('This is Neural Network')
    uuid = hashlib.md5(filename+'_NN:'+a).hexdigest()
    # uuid = str(uuid.uuid4())
    id_nn = uuid
    filename = os.path.join(folder, 'images/%s.png' % id_nn)
    plt.savefig(filename)
    plt.figure()
    plt.scatter(prediction_ridge,actual,s=3)
    # plt.xlim((0,1000))
    # plt.ylim((0,1000))
    plt.xlabel('Prediction')
    plt.ylabel('Observation')
    plt.title('This is Ridge Regression')
    uuid = hashlib.md5(filename+'_RIDGE:'+a).hexdigest()
    id_ridge = uuid
    filename1 = os.path.join(folder, 'images/%s.png' % id_ridge)
    plt.savefig(filename1)
    plt.figure()
    plt.scatter(prediction_lasso,actual,s=3)
    # plt.xlim((0,1000))
    # plt.ylim((0,1000))
    plt.xlabel('Prediction')
    plt.ylabel('Observation')
    plt.title('This is Lasso Regression')
    uuid = hashlib.md5(filename+'_LASSO:'+a).hexdigest()
    id_lasso = uuid
    filename2 = os.path.join(folder, 'images/%s.png' % id_lasso)
    plt.savefig(filename2)
    return dict(a=a, id_nn=id_nn, id_ridge=id_ridge, id_lasso=id_lasso, 
                ridge_coef=ridge_coef, lasso_coef=lasso_coef, 
                weight=weight, r_linear_lassoAIC=r_linear_lassoAIC, 
                r_linear_ridge=r_linear_ridge, r2_score_NN=r2_score_NN, uuid=uuid)

def make_plots(fullname, choose):
    folder, filename = os.path.split(fullname)
    folder = folder.replace('/uploads','/static')
    a = choose
    df = pd.read_csv(fullname)
    
    def add_layer(x, in_size, out_size, activation_function,layer,dropout):
        num_layer = 'layer%s' % layer
        with tf.name_scope(num_layer):
            global W
            tf.set_random_seed(666)
            W = tf.Variable(tf.random_uniform([in_size, out_size]))
        with tf.name_scope('y'):
            output = tf.matmul(x, W)
            if activation_function is 1:
                out = tf.nn.softsign(output)
            elif activation_function is 0:
                out = tf.nn.dropout(output,dropout)
            else:
                out = output
        return out

    # Create input variable
    with tf.name_scope('Input'):
        x = tf.placeholder(tf.float32,[None,9])
        y_ = tf.placeholder(tf.float32,[None,1])
        keep_prob = tf.placeholder(tf.float32) #dropout probability
        dropout = 0.85

    # Build up layers
    with tf.name_scope('Layer'):
         y = add_layer(x,9,1,2,1,dropout)

    # input
    header = list(df.columns.values)
    list_header = []
    for i in range(len(header)):
        if header[i] != a:
            list_header.append(header[i])

    # cost functiontensoflow
    with tf.name_scope('Cost'):
        cost = tf.reduce_mean(tf.square(y_-y))

    # training
    Training = tf.train.AdamOptimizer(0.01)
    train = Training.minimize(cost)

    with tf.Session() as sess:
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        feed = {}
        chi_sum = 0
        predict = []
        actual = []
        chi_plot = []
        i_count=[]
        counter = 0
        df1 = np.float32(df)
        for i in xrange(len(df.values)):
            xs = np.array(np.float32([[df.iloc[i][headers] for headers in list_header]]))/1000
            ys = np.array([[np.float32(df.iloc[i][a])]])/1000
            feed={x:xs,y_:ys,keep_prob:dropout}
            prediction = sess.run(y,feed_dict=feed)
            sess.run(train,feed_dict=feed)
            chi = (np.square(ys-prediction))/(prediction)
            chi_sum += chi
            predict.append(float(prediction)*1000)
            actual.append(float(ys)*1000)
            chi_plot.append(float(chi))
            i_count.append(i)
    b = stats.ttest_ind(actual, predict)
    c= stats.pearsonr(actual, predict)
    slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(actual,predict)
    r_square1 = r_value1**2
    plt.figure()
    T = np.arctan2(actual,predict) #for color generation
    plt.scatter(predict,actual,s=3,c=T)
    # plt.xlim((0,1000))
    # plt.ylim((0,1000))
    plt.xlabel('Prediction')
    plt.ylabel('Observation')
    plt.title('This is Neural Network')
    uuid = hashlib.md5(filename+'_NN:'+a).hexdigest()
    # uuid = str(uuid.uuid4())
    id_nn = uuid
    filename = os.path.join(folder, 'images/%s.png' % id_nn)
    plt.savefig(filename)

    table = SmartTable(*header)
    dict_index = {}
    for i in range(len(header)):
        dict_index[header[i]]=i
    for i in xrange(len(df.values)):
        temp = []
        for j in xrange(len(df.columns)):
            temp.append(df.values[i][j])
        table.rows.append([temp[dict_index[key]] for key in table.keys])
    
    def compute_importance(table, predicted_key, n):
        table = table.purge_null()
        selected = collections.OrderedDict()
        y_index = table.keys.index(predicted_key)
        points = [[row, row[y_index], 1.0] for row in table.rows]
        predictions = []
        while len(selected) < n:
            best_chi2, best_key = None, None        
            for key in table.keys:
                if not key == predicted_key and not key in selected:
                    pool_indexes = map(table.keys.index, selected.keys() + [key])
                    fs = [(lambda x,i=i: x[i]) for i in pool_indexes]
                    cs, chi2, f = fit_least_squares(points, fs)
                    if best_key is None or chi2 < best_chi2:
                        best_chi2, best_key = chi2, key
            selected[best_key] = best_chi2
        for i, point in enumerate(points):
            predictions.append((i, point[1], f(point[0])))
        return selected, predictions, f, cs
    selected, prediction, f, cs = compute_importance(table,a,9)
    predict = []
    actual = []
    for i in range(len(df.values)):
        predict.append(prediction[i][2])
        actual.append(prediction[i][1])
    plt.figure()
    T = np.arctan2(actual,predict)
    plt.scatter(predict,actual,s=3,c=T)
    # plt.xlim((0,1000))
    # plt.ylim((0,1000))
    plt.xlabel('Prediction')
    plt.ylabel('Observation')
    plt.title('This is LSF')
    uuid = hashlib.md5(filename+'_LSF:'+a).hexdigest()
    id_lsf = uuid
    filename = os.path.join(folder, 'images/%s.png' % id_lsf)
    plt.savefig(filename)
    d = stats.ttest_ind(actual, predict)
    e= stats.pearsonr(actual, predict)
    slope, intercept, r_value, p_value, std_err = stats.linregress(actual,predict)
    r_square = r_value**2
    return dict(a=a, id_nn=id_nn, id_lsf=id_lsf, b=b, c=c, d=d, e=e, 
                r_square1=r_square1, r_square=r_square, uuid=uuid)

