#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Explore-data" data-toc-modified-id="Explore-data-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Explore data</a></span><ul class="toc-item"><li><span><a href="#basic-info" data-toc-modified-id="basic-info-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>basic info</a></span></li><li><span><a href="#missing-values,-boxplot-for-data" data-toc-modified-id="missing-values,-boxplot-for-data-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>missing values, boxplot for data</a></span><ul class="toc-item"><li><span><a href="#five-feature-pairwise-plot" data-toc-modified-id="five-feature-pairwise-plot-1.2.1"><span class="toc-item-num">1.2.1&nbsp;&nbsp;</span>five feature pairwise plot</a></span></li><li><span><a href="#boxplot" data-toc-modified-id="boxplot-1.2.2"><span class="toc-item-num">1.2.2&nbsp;&nbsp;</span>boxplot</a></span></li></ul></li><li><span><a href="#pie-chart-for-class-distribution" data-toc-modified-id="pie-chart-for-class-distribution-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>pie chart for class distribution</a></span></li></ul></li><li><span><a href="#Model-comparison" data-toc-modified-id="Model-comparison-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Model comparison</a></span><ul class="toc-item"><li><span><a href="#setup-code" data-toc-modified-id="setup-code-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>setup code</a></span></li><li><span><a href="#basedline-score" data-toc-modified-id="basedline-score-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>basedline score</a></span></li><li><span><a href="#standarised-scores-(0-mean)" data-toc-modified-id="standarised-scores-(0-mean)-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>standarised scores (0 mean)</a></span></li><li><span><a href="#display-result-for-final-model-choices" data-toc-modified-id="display-result-for-final-model-choices-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>display result for final model choices</a></span></li></ul></li><li><span><a href="#Feature-selection" data-toc-modified-id="Feature-selection-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Feature selection</a></span><ul class="toc-item"><li><span><a href="#features-correlation" data-toc-modified-id="features-correlation-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>features correlation</a></span></li><li><span><a href="#correlated-with-y" data-toc-modified-id="correlated-with-y-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>correlated with y</a></span></li><li><span><a href="#Feature-Importance-using-extra-forest" data-toc-modified-id="Feature-Importance-using-extra-forest-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>Feature Importance using extra forest</a></span></li><li><span><a href="#Tree-based-feature-selection" data-toc-modified-id="Tree-based-feature-selection-3.4"><span class="toc-item-num">3.4&nbsp;&nbsp;</span>Tree-based feature selection</a></span></li><li><span><a href="#linear-SVM" data-toc-modified-id="linear-SVM-3.5"><span class="toc-item-num">3.5&nbsp;&nbsp;</span>linear SVM</a></span></li><li><span><a href="#PCA" data-toc-modified-id="PCA-3.6"><span class="toc-item-num">3.6&nbsp;&nbsp;</span>PCA</a></span><ul class="toc-item"><li><span><a href="#performance-comparison-using-PCA" data-toc-modified-id="performance-comparison-using-PCA-3.6.1"><span class="toc-item-num">3.6.1&nbsp;&nbsp;</span>performance comparison using PCA</a></span></li></ul></li><li><span><a href="#ANOVA-f-test" data-toc-modified-id="ANOVA-f-test-3.7"><span class="toc-item-num">3.7&nbsp;&nbsp;</span>ANOVA f-test</a></span></li><li><span><a href="#chi-square-test" data-toc-modified-id="chi-square-test-3.8"><span class="toc-item-num">3.8&nbsp;&nbsp;</span>chi-square test</a></span></li></ul></li><li><span><a href="#Remove-Outlier" data-toc-modified-id="Remove-Outlier-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Remove Outlier</a></span><ul class="toc-item"><li><span><a href="#Isolation-forest" data-toc-modified-id="Isolation-forest-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Isolation forest</a></span></li><li><span><a href="#DBSCAN" data-toc-modified-id="DBSCAN-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>DBSCAN</a></span></li><li><span><a href="#Interquatile-range" data-toc-modified-id="Interquatile-range-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Interquatile range</a></span><ul class="toc-item"><li><span><a href="#filter-using-interqutertile" data-toc-modified-id="filter-using-interqutertile-4.3.1"><span class="toc-item-num">4.3.1&nbsp;&nbsp;</span>filter using interqutertile</a></span></li><li><span><a href="#boxplot-after-outlier-removal" data-toc-modified-id="boxplot-after-outlier-removal-4.3.2"><span class="toc-item-num">4.3.2&nbsp;&nbsp;</span>boxplot after outlier removal</a></span></li></ul></li></ul></li><li><span><a href="#Parameter-tuning" data-toc-modified-id="Parameter-tuning-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Parameter tuning</a></span><ul class="toc-item"><li><span><a href="#KNN" data-toc-modified-id="KNN-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>KNN</a></span></li><li><span><a href="#extra-tree" data-toc-modified-id="extra-tree-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>extra tree</a></span></li><li><span><a href="#random-forest" data-toc-modified-id="random-forest-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>random forest</a></span></li><li><span><a href="#logistic-regression" data-toc-modified-id="logistic-regression-5.4"><span class="toc-item-num">5.4&nbsp;&nbsp;</span>logistic regression</a></span></li><li><span><a href="#MLP" data-toc-modified-id="MLP-5.5"><span class="toc-item-num">5.5&nbsp;&nbsp;</span>MLP</a></span></li></ul></li><li><span><a href="#Testing-on-validation-set" data-toc-modified-id="Testing-on-validation-set-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Testing on validation set</a></span><ul class="toc-item"><li><span><a href="#XGboost" data-toc-modified-id="XGboost-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>XGboost</a></span></li><li><span><a href="#extra-tree" data-toc-modified-id="extra-tree-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>extra tree</a></span><ul class="toc-item"><li><span><a href="#visulisation" data-toc-modified-id="visulisation-6.2.1"><span class="toc-item-num">6.2.1&nbsp;&nbsp;</span>visulisation</a></span></li></ul></li><li><span><a href="#random-forest" data-toc-modified-id="random-forest-6.3"><span class="toc-item-num">6.3&nbsp;&nbsp;</span>random forest</a></span></li><li><span><a href="#MLP" data-toc-modified-id="MLP-6.4"><span class="toc-item-num">6.4&nbsp;&nbsp;</span>MLP</a></span></li><li><span><a href="#KNN" data-toc-modified-id="KNN-6.5"><span class="toc-item-num">6.5&nbsp;&nbsp;</span>KNN</a></span></li><li><span><a href="#Logistic-Regression" data-toc-modified-id="Logistic-Regression-6.6"><span class="toc-item-num">6.6&nbsp;&nbsp;</span>Logistic Regression</a></span></li></ul></li><li><span><a href="#Voting-ensemble" data-toc-modified-id="Voting-ensemble-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Voting ensemble</a></span><ul class="toc-item"><li><span><a href="#KNN-+-MLP-+-RF-+-ET" data-toc-modified-id="KNN-+-MLP-+-RF-+-ET-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>KNN + MLP + RF + ET</a></span></li><li><span><a href="#RF-+-ET-+-MLP-(not-used)" data-toc-modified-id="RF-+-ET-+-MLP-(not-used)-7.2"><span class="toc-item-num">7.2&nbsp;&nbsp;</span>RF + ET + MLP (not used)</a></span></li></ul></li><li><span><a href="#Submit---model-used:-voting-ensemble-(KNN-+-MLP-+-RF-+-ET)" data-toc-modified-id="Submit---model-used:-voting-ensemble-(KNN-+-MLP-+-RF-+-ET)-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Submit - model used: voting ensemble (KNN + MLP + RF + ET)</a></span><ul class="toc-item"><li><span><a href="#compare-extra-tree-with-voting-ensemble" data-toc-modified-id="compare-extra-tree-with-voting-ensemble-8.1"><span class="toc-item-num">8.1&nbsp;&nbsp;</span>compare extra tree with voting ensemble</a></span></li><li><span><a href="#csv" data-toc-modified-id="csv-8.2"><span class="toc-item-num">8.2&nbsp;&nbsp;</span>csv</a></span></li></ul></li></ul></div>

# # Project
# 
# This is ML project done in COMP9417 where I practise the standard ML pipeline skill based on various sources mostly Kaggle.

# TOC(2) nbextension is suggested to install for below section  
# 
# External libraries are needed to run the below code:
# * plotly  
# 

# # Explore data

# In[1]:


#  set a seed for reproducibility
SEED = 23
np.random.seed(SEED) # needed

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as py

get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))


# ## basic info

# In[48]:


X_train = pd.read_csv('X_train.csv', header=None) # ignore first row
y_train = pd.read_csv('y_train.csv', header=None)


# In[3]:


X_train.info()


# In[4]:


X_train.head()


# In[5]:


X_train.describe()


# In[6]:


y_train.describe()


# ## missing values, boxplot for data

# In[7]:


# display n nan values
indices_nan = np.where(np.asanyarray(np.isnan(X_train)))[0]
print("List the indices of the removed data points are", list(indices_nan)) # if base csv, then this index + 2
# X_train = X_train.dropna() # remove rows if any conlumn contains nan


# ### five feature pairwise plot

# In[8]:


tmp = X_train.iloc[:, 0:5][:100]
sns.pairplot(tmp)#
plt.show()


# ### boxplot

# In[9]:


plt.figure(0, figsize=(25,10))
sns.boxplot(x="variable", y="value", data=pd.melt(X_train))

plt.show()


# ## pie chart for class distribution

# In[111]:


def PlotPie(df, idxFeature):

    trace=go.Pie(labels=[str(int(i)) for i in df.iloc[:, idxFeature].unique()],values=df.iloc[:, idxFeature].value_counts())
    py.iplot([trace])
    
PlotPie(y_train, 0)


# # Model comparison 
# basedline, cross validation, data is normalised

# ## setup code

# In[66]:


# Load libraries

from pandas import set_option
#from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import StandardScaler
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# In[128]:


import time
def BasedLine2(X_train, y_train,models, scoring='f1_weighted'): # marco bigger penalisation when  model does not perform well with the minority classes.
    # Test options and evaluation metric
    num_folds = 10

    results = []
    names = []
    for name, model in models:
        t0 = time.time()
        kfold = StratifiedKFold(n_splits=num_folds, random_state=SEED, shuffle=True)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        t1 = time.time()
        t = t1 - t0
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f) (%0.4f)s" % (name, cv_results.mean(), cv_results.std(), t)
        print(msg)
        
    return names, results


# In[21]:


class PlotBoxR(object):
    
    def __Trace(self,nameOfFeature,value): 
    
        trace = go.Box(
            y=value,
            name = nameOfFeature,
            marker = dict(
                color = 'rgb(0, 128, 128)',
            )
        )
        return trace

    def PlotResult(self,names,results):
        
        data = []

        for i in range(len(names)):
            data.append(self.__Trace(names[i],results[i]))


        py.iplot(data)


# In[85]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def GetBasedModel():
    
    pipelines = [
        ('LR'  , LogisticRegression()),
        ('LDA' , LinearDiscriminantAnalysis()),
        ('KNN' , KNeighborsClassifier()),
        ('CART', DecisionTreeClassifier()),
        ('NB'  , GaussianNB()),
        ('SVM' , SVC()),
        #('AB'  , AdaBoostClassifier()),
        ('DTR' , DecisionTreeRegressor()),
        ('RF'  , RandomForestClassifier()),
        ('ET'  , ExtraTreesClassifier()),
        ('NN'  , MLPClassifier())
    ]

    return pipelines 


# In[167]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def GetScaledModel(nameOfScaler, is_final=False, onemodel=[]):
    '''
    onemodel[0] - abbr of the model name
    [1] - the model
    '''
    if nameOfScaler == 'standard':
        scaler = StandardScaler()
    elif nameOfScaler =='minmax':
        scaler = MinMaxScaler()
    
    if onemodel:
        pipelines = [(nameOfScaler+onemodel[0]  , Pipeline([('Scaler', scaler),(onemodel[0]   , onemodel[1])])),]
            
        return pipelines 
    
    if not is_final:
        pipelines = [
            (nameOfScaler+'LR'  , Pipeline([('Scaler', scaler),('LR'  , LogisticRegression())])),
            (nameOfScaler+'LDA' , Pipeline([('Scaler', scaler),('LDA' , LinearDiscriminantAnalysis())])),
            (nameOfScaler+'KNN' , Pipeline([('Scaler', scaler),('KNN' , KNeighborsClassifier())])),
            (nameOfScaler+'CART', Pipeline([('Scaler', scaler),('CART', DecisionTreeClassifier(random_state=SEED))])),
            (nameOfScaler+'NB'  , Pipeline([('Scaler', scaler),('NB'  , GaussianNB())])),
            (nameOfScaler+'SVM' , Pipeline([('Scaler', scaler),('SVM' , SVC())])),
            # (nameOfScaler+'AB'  , Pipeline([('Scaler', scaler),('AB'  , AdaBoostClassifier())])),
            (nameOfScaler+'DTR' , Pipeline([('Scaler', scaler),('DTR' , DecisionTreeRegressor(random_state=SEED))])),
            (nameOfScaler+'RF'  , Pipeline([('Scaler', scaler),('RF'  , RandomForestClassifier(random_state=SEED))])),
            (nameOfScaler+'ET'  , Pipeline([('Scaler', scaler),('ET'  , ExtraTreesClassifier(random_state=SEED))])),
            (nameOfScaler+'NN'  , Pipeline([('Scaler', scaler),('NN'  , MLPClassifier())])) 
        ]
    
    else:
         pipelines = [
            (nameOfScaler+'LR'  , Pipeline([('Scaler', scaler),('LR'  , LogisticRegression())])),
            (nameOfScaler+'KNN' , Pipeline([('Scaler', scaler),('KNN' , KNeighborsClassifier())])),
            (nameOfScaler+'RF'  , Pipeline([('Scaler', scaler),('RF'  , RandomForestClassifier(random_state=SEED))])),
            (nameOfScaler+'ET'  , Pipeline([('Scaler', scaler),('ET'  , ExtraTreesClassifier(random_state=SEED))])),
            (nameOfScaler+'NN'  , Pipeline([('Scaler', scaler),('NN'  , MLPClassifier())])) 
         ]
        
    return pipelines 


# ## basedline score

# In[87]:


from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

models = GetBasedModel()
names,results = BasedLine2(X_train, y_train.values.ravel(),models)


# In[88]:


def ScoreDataFrame(names,results):
    def floatingDecimals(f_val, dec=3):
        prc = "{:."+str(dec)+"f}" 
    
        return float(prc.format(f_val))

    scores = []
    for r in results:
        scores.append(floatingDecimals(r.mean(),4))

    scoreDataFrame = pd.DataFrame({'Model':names, 'Score': scores})
    return scoreDataFrame
basedLineScore = ScoreDataFrame(names,results)
basedLineScore


# In[89]:


PlotBoxR().PlotResult(names,results)


# ## standarised scores (0 mean)

# In[90]:


models = GetScaledModel('standard')
names,results = BasedLine2(X_train, y_train.values.ravel(),models)
scaledScoreStandard = ScoreDataFrame(names,results)


# In[91]:


compareModels = pd.concat([basedLineScore,
                           scaledScoreStandard], axis=1)
compareModels


# In[92]:


PlotBoxR().PlotResult(names,results)


# ## display result for final model choices 

# In[134]:


models = GetScaledModel('standard', is_final=True)
names,results = BasedLine2(X_train, y_train.values.ravel(),models)
scaledScoreStandard = ScoreDataFrame(names,results)


# # Feature selection 
# - (Numerical Input, Categorical Output)

# ## features correlation

# Too many featrure and so heatmap not done yet

# In[241]:


# Create correlation matrix
corr_matrix = X_train.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

thresh = 0.99
# Find features with correlation greater than 0
X_train_corr_idx = [column for column in upper.columns if any(upper[column] > thresh)]
X_train_corr = X_train[X_train_corr_idx]
f = plt.figure(figsize=(19, 15))
plt.matshow(X_train_corr.corr(), fignum=f.number)
plt.xticks(range(X_train_corr.select_dtypes(['number']).shape[1]), X_train_corr.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(X_train_corr.select_dtypes(['number']).shape[1]), X_train_corr.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);


# ## correlated with y

# In[84]:


import seaborn as sns

data = X_train.copy()
data['y'] = y_train.copy() 
corr_mx = data.corr()

corr_target = abs(corr_mx['y'])
relevant_features = corr_target[corr_target > 0.5]
print(relevant_features)
print("feature 107 is highly correlated with the output variable y")


# ## Feature Importance using extra forest

# In[67]:


clf = ExtraTreesClassifier(random_state=SEED, n_estimators=300)  # https://scikit-learn.org/stable/modules/feature_selection.html
clf.fit(X_train, y_train.values.ravel())

# #############################################################################
# Plot feature importance
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + 1 # start from 1


# In[69]:


plt.figure(figsize=(60,15))
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X_train.columns[sorted_idx])#boston.feature_names[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


# In[76]:


plt.figure(figsize=(15, 7))
plt.bar(X_train.columns[sorted_idx], feature_importance[sorted_idx])

plt.show()


# In[558]:


plt.figure(figsize=(10,5))
plt.subplot(1, 2, 2)

pos_10 = pos[:10]
sorted_idx_10 = sorted_idx[128-10:]
plt.barh(pos_10, feature_importance[sorted_idx_10], align='center')
plt.yticks(pos_10, X_train.columns[sorted_idx_10]) #boston.feature_names[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Tree feature Importance')
plt.show()


# ## Tree-based feature selection 

# In[237]:


from sklearn.feature_selection import SelectFromModel

extra_tree = SelectFromModel(clf, prefit=True)
X_train_uncorrelated = extra_tree.transform(X_train)

print(X_train.shape)
print(X_train_uncorrelated.shape)


# In[138]:


models = GetScaledModel('standard', is_final=True)

names,results = BasedLine2(X_train_uncorrelated, y_train.values.ravel() ,models)
scaledScoreUncorrelated = ScoreDataFrame(names,results)


# In[ ]:


models = GetScaledModel('minmax')

names, results = BasedLine2(X_train_uncorrelated, y_train.values.ravel(), models)
scaledScoreUncorrelated = ScoreDataFrame(names,results)


# In[ ]:


# tree based
standardLR: 0.988388 (0.002939) (2.5770)s
standardLDA: 0.936282 (0.011338) (0.5000)s
standardKNN: 0.989694 (0.003474) (1.5740)s
standardCART: 0.969004 (0.007555) (3.0900)s
standardNB: 0.653904 (0.011444) (0.1730)s
standardSVM: 0.971762 (0.004988) (9.8190)s
standardDTR: 0.968384 (0.004151) (2.6700)s
standardRF: 0.993290 (0.002944) (23.5940)s
standardET: 0.995328 (0.002173) (5.4410)s
standardNN: 0.994249 (0.002719) (85.5400)s
# tree based
minmaxLR: 0.929561 (0.013392) (2.5490)s
minmaxLDA: 0.936282 (0.011338) (0.4920)s
minmaxKNN: 0.991497 (0.003797) (1.3530)s
minmaxCART: 0.968645 (0.007678) (2.9610)s
minmaxNB: 0.653746 (0.011244) (0.1310)s
minmaxSVM: 0.924977 (0.009184) (17.9440)s
minmaxDTR: 0.969581 (0.004976) (2.5860)s
minmaxRF: 0.992931 (0.003003) (23.4960)s
minmaxET: 0.995328 (0.002173) (5.4340)s
minmaxNN: 0.990418 (0.003632) (85.1250)s


# ## linear SVM 

# In[274]:


from sklearn.svm import LinearSVC
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train, y_train.values.ravel())
lsvc = SelectFromModel(lsvc, prefit=True)


# In[275]:


X_train_uncorrelated = lsvc.transform(X_train)
print(X_train.shape)
print(X_train_uncorrelated.shape)


# In[153]:


models = GetScaledModel('standard', is_final=True)

names,results = BasedLine2(X_train_uncorrelated, y_train.values.ravel() ,models)
scaledScoreUncorrelated = ScoreDataFrame(names,results)


# In[ ]:


models = GetScaledModel('minmax')

names, results = BasedLine2(X_train_uncorrelated, y_train.values.ravel(), models)
scaledScoreUncorrelated = ScoreDataFrame(names,results)


# In[ ]:


# linear svm
standardLR: 0.988271 (0.002957) (3.2990)s
standardLDA: 0.935625 (0.008560) (0.9490)s
standardKNN: 0.988488 (0.004889) (3.2280)s
standardCART: 0.967651 (0.005911) (4.7810)s
standardNB: 0.628579 (0.008900) (0.2740)s
standardSVM: 0.970788 (0.005237) (16.2470)s
standardDTR: 0.965663 (0.004823) (4.0070)s
standardRF: 0.994126 (0.002236) (30.4280)s
standardET: 0.994847 (0.002147) (6.7770)s
standardNN: 0.994368 (0.001699) (91.4530)s
# linear svm
minmaxLR: 0.921744 (0.013821) (3.5190)s
minmaxLDA: 0.935625 (0.008560) (0.7570)s
minmaxKNN: 0.989935 (0.004794) (2.5760)s
minmaxCART: 0.966932 (0.005766) (4.6940)s
minmaxNB: 0.628579 (0.008900) (0.2070)s
minmaxSVM: 0.856650 (0.013341) (37.7310)s
minmaxDTR: 0.965901 (0.004860) (4.0230)s
minmaxRF: 0.994126 (0.002236) (30.2350)s
minmaxET: 0.994847 (0.002147) (6.7730)s
minmaxNN: 0.989569 (0.002992) (88.8660)s


# ## PCA

# In[109]:


from sklearn.decomposition import PCA
pca = PCA(n_components=10)
pca.fit(X_train)


# In[110]:


print(pca.components_)
print(pca.explained_variance_)


# In[218]:


pca = PCA().fit(X_train)
plt.figure(figsize=(8,4))
plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('PCA - number of components')
plt.ylabel('cumulative explained variance')


# In[113]:


pca = PCA(n_components=10)
pca.fit(X_train)
X_pca = pca.transform(X_train)
print("original shape:   ", X_train.shape)
print("transformed shape:", X_pca.shape)


# ### performance comparison using PCA

# In[ ]:


for n_features in [10, 20, 30, 40, 50]:
    pca = PCA(n_components=n_features)
    pca.fit(X_train)
    X_pca = pca.transform(X_train)
    models = GetScaledModel('standard', is_final=True)
    print("n_features = " + str(n_features))
    names,results = BasedLine2(X_pca, y_train.values.ravel(), models)
    scaledScoreUncorrelated = ScoreDataFrame(names,results)


# In[ ]:


# result
n_features = 10
standardLR: 0.963734 (0.004937) (2.5460)s
standardKNN: 0.992207 (0.003687) (0.4720)s
standardRF: 0.994005 (0.002149) (11.1300)s
standardET: 0.995564 (0.002146) (3.4390)s
standardNN: 0.994612 (0.002526) (81.4890)s
n_features = 20
standardLR: 0.974999 (0.007364) (2.2830)s
standardKNN: 0.993167 (0.003261) (1.2290)s
standardRF: 0.995324 (0.002240) (14.6100)s
standardET: 0.996525 (0.001248) (4.4630)s
standardNN: 0.994970 (0.003067) (85.5540)s
n_features = 30
standardLR: 0.983006 (0.003419) (2.7440)s
standardKNN: 0.990292 (0.004061) (2.1570)s
standardRF: 0.994962 (0.001995) (18.2360)s
standardET: 0.996646 (0.001915) (5.3520)s
standardNN: 0.993415 (0.003085) (66.8380)s
n_features = 40
standardLR: 0.984557 (0.002779) (2.4510)s
standardKNN: 0.988021 (0.003556) (3.0570)s
standardRF: 0.995563 (0.002012) (21.5390)s
standardET: 0.996406 (0.001515) (6.0720)s
standardNN: 0.992576 (0.004990) (58.1020)s
n_features = 50
standardLR: 0.987072 (0.003370) (2.7980)s
standardKNN: 0.986588 (0.003155) (4.4080)s
standardRF: 0.994842 (0.002406) (25.0940)s
standardET: 0.996406 (0.001778) (6.8930)s
standardNN: 0.992110 (0.004276) (53.7570)s


# In[109]:


standardLR_lines = [0.963734, 0.974999, 0.983006, 0.984557, 0.987072]
standardKNN_lines = [0.992207, 0.993167, 0.990292, 0.988021, 0.986588]
standardRF_lines = [0.994005, 0.995324, 0.994962, 0.995563, 0.994842]
standardET_lines = [0.995564, 0.996525, 0.996646, 0.996406, 0.996406]
standardNN_lines = [0.994612, 0.994970, 0.993415, 0.992576, 0.992110]
plt.figure(figsize=(15, 7))
y = [i for i in range(10, 51, 10)]
# plot lines
plt.plot(y, standardLR_lines, label = "standardLR", marker='o')
plt.text(50, standardLR_lines[-1]+0.001, "standardLR")
plt.plot(y, standardKNN_lines, label = "standardKNN", marker='o')
plt.text(50, standardKNN_lines[-1]-0.001, "standardKNN")
plt.plot(y, standardRF_lines, label = "standardRF", marker='o')
plt.text(50, standardRF_lines[-1], "standardRF")
plt.plot(y, standardET_lines, label = "standardET", marker='o')
plt.text(50, standardET_lines[-1], "standardET")
plt.plot(y, standardNN_lines, label = "standardNN", marker='o')
plt.text(50, standardNN_lines[-1], "standardNN")
plt.title("Number of PCA component vs F1-scores on 10 folds")
plt.xlabel("Number of PCA component")
plt.ylabel("Weighted F1-score")
plt.legend()
plt.show()


# ## ANOVA f-test

# In[524]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

def select_features(X_train, y_train):
    # configure to select all features
    fs = SelectKBest(score_func=f_classif, k='all')
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    
    return X_train_fs, fs


# In[528]:


X_train_fs, fs = select_features(X_train, y_train)

plt.figure(figsize=(15, 7))
plt.bar([i for i in range(len(fs.scores_))], fs.scores_)

plt.show()


# In[557]:


pos = np.arange(sorted_idx.shape[0]) + 1 # start from 1
pos_10 = pos[:10]

# top 10 important feature
anova_idx = fs.scores_.argsort()[-10:]
anova_values = fs.scores_[anova_idx]

plt.figure(figsize=(10,5))
plt.subplot(1, 2, 2)

plt.barh(pos_10, anova_values, align='center')
plt.yticks(pos_10, X_train.columns[anova_idx])#boston.feature_names[sorted_idx])
plt.xlabel('Importance')
plt.title('ANOVA feature Importance')
plt.show()


# ## chi-square test 
# not applied as positive x value only 

# In[269]:


from sklearn.feature_selection import chi2

chi_scores = chi2(X_train,y_train)

p_values = pd.Series(chi_scores[1],index = X.columns)
p_values.sort_values(ascending = False , inplace = True)


# # Remove Outlier

# ## Isolation forest

# In[9]:


from sklearn.ensemble import IsolationForest
from sklearn.metrics import make_scorer, f1_score
from sklearn import model_selection
from sklearn.datasets import make_classification

iso = IsolationForest(random_state=SEED)
y_pred = iso.fit_predict(X_train)
X_train_cleaned = X_train[np.where(y_pred == 1, True, False)]
y_train_cleaned = y_train[np.where(y_pred == 1, True, False)]


# In[17]:


# summarize the shape of the updated training dataset
print(X_train_cleaned.shape, y_train_cleaned.shape)
cleaned,_ = X_train_cleaned.shape

total, _ = X_train.shape
print("number of outlier row removed: ", total - cleaned)
print("percentage of outliers removed: ", (total-cleaned) / total)


# In[6]:


plt.figure(0, figsize=(25,10))
sns.boxplot(x="variable", y="value", data=pd.melt(X_train_cleaned))
plt.show()


# ## DBSCAN

# In[82]:


from sklearn.cluster import DBSCAN

test = X_train[[107, 10]]
plt.scatter(test[107], test[10])
print(test)
db = DBSCAN(eps=0.4, min_samples=10).fit(test)
label = db.labels_
outliers = test[db.labels_ == -1]
# print(outliers)


# In[81]:


X_train_cleaned = X_train[db.labels_ != -1]
y_train_cleaned = y_train[db.labels_ != -1]
plt.figure(0, figsize=(25,10))
sns.boxplot(x="variable", y="value", data=pd.melt(X_train_cleaned))
plt.show()


# ## Interquatile range

# Numerical features preprocessing is different for tree and non tree model.
# 
# 1) Usually:
# - Tree based models does not depend on scaling
# - Non-tree based models hugely depend on scaling 
# 
# 2) Most Often used preprocening are:
# - MinMax scaler to [0,1]
# - Standard Scaler to mean = 0 and std =1
# - Rank (We do not work on it in this data set)
# - Using np.log(1+data),  np.sqrt(data) and stats.boxcox(data) (for exp dependency)
# 
# let's try some of them and see how our model prediction change by scalling
# 

# ### filter using interqutertile

# In[62]:


from numpy import quantile
# calculate interquartile range
# https://stackoverflow.com/questions/50461349/how-to-remove-outlier-from-dataframe-using-iqr
def remove_outlier_IQR(df):
    Q1=df.quantile(0.25)
    Q3=df.quantile(0.75)
    IQR=Q3-Q1
    thresh = 1.5
    df_final=df[~((df<(Q1-thresh*IQR)) | (df>(Q3+thresh*IQR))).any(axis=1)]
    return df_final                          
    
cleaned = remove_outlier_IQR(X_train)
n, _ = cleaned.shape
total, _ = X_train.shape
print("n removed:", total - n)
print("perc:", (total - n)/total)


# ### boxplot after outlier removal

# In[60]:


plt.figure(0, figsize=(25,10))
sns.boxplot(x="variable", y="value", data=pd.melt(cleaned))

plt.show()


# # Parameter tuning 
# The time consuming ones such as tree model and MLP are tuned on colab, so only results are copied and paste below

# In[44]:


from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)


# ## KNN

# In[189]:


KNN_model = KNeighborsClassifier()
param_grid = {
    "n_neighbors":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
}

KNN_GridSearch = GridSearchCV(KNN_model, 
                              param_grid, 
                              scoring="f1_weighted", 
                              cv=10
                              )
KNN_GridSearch.fit(X_train_scaled, y_train.values.ravel())
print(KNN_GridSearch.best_estimator_)
print(KNN_GridSearch.best_score_)


# ## extra tree

# In[73]:


from sklearn.metrics import make_scorer, f1_score
from sklearn import model_selection

clf = ExtraTreesClassifier(random_state=SEED)  # https://scikit-learn.org/stable/modules/feature_selection.html

param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}

grid_dt_estimator = model_selection.GridSearchCV(clf, 
                                                 param_grid,
                                                 scoring="f1_weighted", 
                                                 refit=True,
                                                 cv=10, 
                                                 return_train_score=True)
grid_dt_estimator.fit(X_train, y_train.values.ravel())

# ExtraTreesClassifier(criterion='gini', max_depth=None, max_features=10,
#                      min_samples_leaf=1, min_samples_split=2,n_estimators=300,random_state=SEED)
# 0.9955660187635302


# ## random forest

# In[ ]:


from sklearn.metrics import make_scorer, f1_score
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
SEED = 23

clf = RandomForestClassifier(random_state=SEED)  # https://scikit-learn.org/stable/modules/feature_selection.html

param_grid = {
              "max_features": [3, 10, 11, 12],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "n_estimators" :[100, 200, 300],
              "criterion": ["gini"]}

grid_dt_estimator = model_selection.GridSearchCV(clf, 
                                                 param_grid,
                                                 scoring="f1_weighted", 
                                                 refit=True,
                                                 cv=10, 
                                                 return_train_score=True)
grid_dt_estimator.fit(X_train, y_train.values.ravel())

# RandomForestClassifier(criterion='gini', max_features=11,
#                        min_samples_leaf=1, min_samples_split=2, n_estimators=200, random_state=SEED)

# 0.9950857839601017


# ## logistic regression

# In[194]:


LR_model = LogisticRegression()

param_grid = {
    "penalty": ['l2'],
    "C": [0.001, 0.01, 0.1, 1, 10],
    'max_iter': list(range(100,500,100)),
}

param_grid = {
    "penalty": ['l2'],
    "C": [1],
    'max_iter': [500]
}


LR_GridSearch = GridSearchCV(LR_model, 
                              param_grid, 
                              scoring="f1_weighted", 
                              cv=10
                              )

LR_GridSearch.fit(X_train_scaled, y_train.values.ravel())
print(LR_GridSearch.best_estimator_)
print(LR_GridSearch.best_score_)

# LogisticRegression(C=1, max_iter=200)
# 0.9914939568317609


# ## MLP

# In[ ]:


from sklearn.metrics import make_scorer, f1_score
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier

SEED = 23

clf = MLPClassifier(random_state=SEED)  # https://scikit-learn.org/stable/modules/feature_selection.html

param_grid = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

grid_dt_estimator = model_selection.GridSearchCV(clf, 
                                                 param_grid,
                                                 scoring="f1_weighted", 
                                                 refit=True,
                                                 cv=10, 
                                                 return_train_score=True)
grid_dt_estimator.fit(X_train, y_train.values.ravel())

# MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
#               beta_2=0.999, early_stopping=False, epsilon=1e-08,
#               hidden_layer_sizes=(50, 100, 50), learning_rate='adaptive',
#               learning_rate_init=0.001, max_fun=15000, max_iter=200,
#               momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
#               power_t=0.5, random_state=23, shuffle=True, solver='adam',
#               tol=0.0001, validation_fraction=0.1, verbose=False,
#               warm_start=False)
# 0.98490141953081


# # Testing on validation set

# In[441]:


X_val = pd.read_csv('X_val.csv', header=None) # ignore first row
y_val = pd.read_csv('y_val.csv', header=None)


# ## XGboost

# In[474]:


# https://www.datacamp.com/community/tutorials/xgboost-in-python
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
xgb_test = XGBClassifier(random_state=SEED, eval_metric='mlogloss')

xgb_test.fit(X_train, y_train.values.ravel())
y_pred = xgb_test.predict(X_val)
print(metrics.classification_report(y_val.values.ravel(), y_pred))
print(metrics.f1_score(y_val, y_pred, average='weighted'))


# In[457]:


kfold = StratifiedKFold(n_splits=10, random_state=SEED, shuffle=True)
results = cross_val_score(xgb_test, X_train,y_train.values.ravel(), cv=kfold, scoring="f1_weighted")
print("Train dataset: ", results.mean())


# In[ ]:


params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }


# ## extra tree

# In[351]:


extra_tree_test = ExtraTreesClassifier(criterion='gini', max_depth=None, min_samples_leaf=1, min_samples_split=2,n_estimators=300,random_state=SEED)

extra_tree_test.fit(X_train, y_train.values.ravel())
y_pred = extra_tree_test.predict(X_val)
print(metrics.classification_report(y_val.values.ravel(), y_pred))
print(metrics.f1_score(y_val, y_pred, average='weighted'))


# In[350]:


# ExtraTreesClassifier(criterion='gini', max_depth=None, min_samples_leaf=1, min_samples_split=2,n_estimators=300,random_state=SEED)
extra_tree_test = Pipeline([('Scaler', scaler),('Et' , 
                            ExtraTreesClassifier(criterion='gini', max_depth=None, min_samples_leaf=1, min_samples_split=2,n_estimators=300,random_state=SEED))])

extra_tree_test.fit(X_train, y_train.values.ravel())
y_pred = extra_tree_test.predict(X_val)
print(metrics.classification_report(y_val.values.ravel(), y_pred))
print(metrics.f1_score(y_val, y_pred, average='weighted'))


# In[348]:


kfold = StratifiedKFold(n_splits=10, random_state=SEED, shuffle=True)
results = cross_val_score(ensemble, X_train,y_train.values.ravel(), cv=kfold, scoring="f1_weighted")
print("Train dataset: ", results.mean())


# ### visulisation

# In[308]:


from sklearn.tree import export_graphviz
from sklearn import tree
import os
# Extract single tree
et_single = extra_tree_test.estimators_[1]

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(et_single,
               feature_names = X_train.columns, 
               #class_names=[i for i in range (1, 7)],
               filled = True)

tmp = fig.savefig('et_individualtree1.png')


# ## random forest

# In[425]:


random_forest_test = RandomForestClassifier(max_features=11, min_samples_leaf=1, min_samples_split=2, n_estimators=200, random_state=SEED) 

random_forest_test.fit(X_train, y_train.values.ravel())
y_pred = random_forest_test.predict(X_val)
print(metrics.classification_report(y_val.values.ravel(), y_pred))
print(metrics.f1_score(y_val, y_pred, average='weighted'))


# In[428]:


kfold = StratifiedKFold(n_splits=10, random_state=SEED, shuffle=True)
results = cross_val_score(random_forest_test, X_train,y_train.values.ravel(), cv=kfold, scoring="f1_weighted")
print("Train dataset: ", results.mean())


# In[289]:


# from sklearn.tree import export_graphviz
# from sklearn import tree
# import os
# # Extract single tree
# rf_single = random_forest_test.estimators_[0]

# fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
# tree.plot_tree(rf_single,
#                feature_names = X_train.columns, 
#                #class_names=[i for i in range (1, 7)],
#                filled = True)
# fig.savefig('rf_individualtree.png')


# ## MLP

# In[456]:


mlp_tmp = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=(50, 100, 50), learning_rate='adaptive',
              learning_rate_init=0.001, max_fun=15000, max_iter=200,
              momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
              power_t=0.5, random_state=23, shuffle=True, solver='adam',
              tol=0.0001, validation_fraction=0.1, verbose=False,
              warm_start=False)

MLP_test = Pipeline([('Scaler', scaler),('MLP' ,mlp_tmp)]) 
MLP_test.fit(X_train, y_train.values.ravel())

y_pred = MLP_test.predict(X_val)
print(metrics.classification_report(y_val.values.ravel(), y_pred))
print(metrics.f1_score(y_val, y_pred, average='weighted'))


# In[377]:


kfold = StratifiedKFold(n_splits=10, random_state=SEED, shuffle=True)
results = cross_val_score(MLP_test, X_train,y_train.values.ravel(), cv=kfold, scoring="f1_weighted")
print("Train dataset: ", results.mean())


# ## KNN

# In[427]:


KNN_test = Pipeline([('Scaler', scaler),('LDA' ,KNeighborsClassifier(n_neighbors=1))]) 
KNN_test.fit(X_train, y_train.values.ravel())

y_pred = KNN_test.predict(X_val)
print(metrics.classification_report(y_val.values.ravel(), y_pred))
print(metrics.f1_score(y_val, y_pred, average='weighted'))


# In[426]:


kfold = StratifiedKFold(n_splits=10, random_state=SEED, shuffle=True)
results = cross_val_score(KNN_test, X_train,y_train.values.ravel(), cv=kfold, scoring="f1_weighted")
print("Train dataset: ", results.mean())


# ## Logistic Regression

# In[367]:


LR_test = Pipeline([('Scaler', scaler),('LDA' , LogisticRegression(C=1, max_iter=500))]) # LogisticRegression(C=1, max_iter=500) # 1 prob
LR_test.fit(X_train, y_train.values.ravel())

y_pred = LR_test.predict(X_val)
print(metrics.classification_report(y_val.values.ravel(), y_pred))
print(metrics.f1_score(y_val, y_pred, average='weighted'))


# # Voting ensemble

# In[433]:


from sklearn.ensemble import VotingClassifier

param = {'n_neighbors': 1}
model2 = KNeighborsClassifier(**param)

model3 = Pipeline([('Scaler', scaler),('MLP' ,MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=(50, 100, 50), learning_rate='adaptive',
              learning_rate_init=0.001, max_fun=15000, max_iter=200,
              momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
              power_t=0.5, random_state=23, shuffle=True, solver='adam',
              tol=0.0001, validation_fraction=0.1, verbose=False,
              warm_start=False))]) 

model4 = RandomForestClassifier()

model5 = ExtraTreesClassifier()


# ## KNN + MLP + RF + ET

# In[434]:


estimators = [('KNN',model2), ('MLP',model3), ('RF',model4),  ('ET',model5)]


# In[435]:


# create the ensemble model
kfold = StratifiedKFold(n_splits=10, random_state=SEED, shuffle=True)
ensemble = VotingClassifier(estimators) # argument "soft" to the voting parameter to take into account the probability of each vote.
results = cross_val_score(ensemble, X_train,y_train.values.ravel(), cv=kfold, scoring="f1_weighted")
ensemble_model = ensemble.fit(X_train,y_train.values.ravel())


# In[479]:


y_pred = ensemble_model.predict(X_val)
print(metrics.classification_report(y_val.values.ravel(), y_pred))
print("Train dataset: ", "mean", results.mean(), "var", results.std())
print("Evalution dataset: ",metrics.f1_score(y_val, y_pred, average='weighted'))


# ## RF + ET + MLP (not used)

# In[ ]:


estimators = [('MLP',model3), ('RF',model4),  ('ET',model5)]


# In[ ]:


# create the ensemble model
kfold = StratifiedKFold(n_splits=10, random_state=SEED, shuffle=True)
ensemble = VotingClassifier(estimators) # argument "soft" to the voting parameter to take into account the probability of each vote.
results = cross_val_score(ensemble, X_train,y_train.values.ravel(), cv=kfold, scoring="f1_weighted")
ensemble_model = ensemble.fit(X_train,y_train.values.ravel())


# In[371]:


y_pred = ensemble_model.predict(X_val)
print(metrics.classification_report(y_val.values.ravel(), y_pred))
print("Train dataset: ", results.mean())
print("Evalution dataset: ",metrics.f1_score(y_val, y_pred, average='weighted'))


# # Submit - model used: voting ensemble (KNN + MLP + RF + ET)

# In[378]:


extra_tree_test = ExtraTreesClassifier(criterion='gini', max_depth=None,
                     min_samples_leaf=1, min_samples_split=2,n_estimators=300,random_state=SEED)

extra_tree_test.fit(X_train, y_train.values.ravel())

y_pred = extra_tree_test.predict(X_val)
print(metrics.classification_report(y_val.values.ravel(), y_pred))
print(metrics.f1_score(y_val, y_pred, average='weighted'))


# In[ ]:


X_test = pd.read_csv('X_test.csv', header=None) # ignore first row


# ## compare extra tree with voting ensemble

# In[ ]:


model1 = Pipeline([('Scaler', scaler),('MLP' ,MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=(50, 100, 50), learning_rate='adaptive',
              learning_rate_init=0.001, max_fun=15000, max_iter=200,
              momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
              power_t=0.5, random_state=23, shuffle=True, solver='adam',
              tol=0.0001, validation_fraction=0.1, verbose=False,
              warm_start=False))]) 

model2 = RandomForestClassifier()

model3 = ExtraTreesClassifier(criterion='gini', max_depth=None, min_samples_leaf=1, min_samples_split=2,n_estimators=300,random_state=SEED)
model4 = Pipeline([('Scaler', scaler),('LDA' ,KNeighborsClassifier(n_neighbors=1))]) 


# In[ ]:


estimators = [('MLP',model1), ('RF',model2),  ('ET',model3), ('KNN', model4)]


# In[480]:


y_pred_check = ensemble.predict(X_test)


# In[483]:


extra_tree_test.fit(X_train, y_train.values.ravel())

y_pred = extra_tree_test.predict(X_test)


# In[484]:


test_check = (y_pred == y_pred_check)


# In[485]:


r, _ = test_check[test_check==True].reshape(-1, 1).shape
print(r)
total = X_test.shape[0]
print(total)
diff_perc = r / total
print(diff_perc)


# ## csv

# In[513]:


prediction = pd.DataFrame(y_pred_check).to_csv('y_test.csv' , header=None, index=None)

