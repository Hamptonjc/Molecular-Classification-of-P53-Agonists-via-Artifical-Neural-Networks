#!/usr/bin/env python
# coding: utf-8

# # Molecular Classification of P53 Agonists via Artifical Neural Network Part 2
# 
# <br>
# <center><b>Author: Jonathan Hampton</b></center>
# 
# <br>
# <center><b>November 2019</b></center>

# In[60]:


# Imports
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib
import numpy as np
import deepchem as dc
import deepchem
from rdkit import Chem
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Load data
from numpy import load

X_train = load('data/X_train_OG.npy')
y_train = load('data/y_train_OG.npy')

X_test = load('data/X_test.npy')
y_test = load('data/y_test.npy')

X_train_resamp1 = load('data/X_train_resamp1.npy')
y_train_resamp1 = load('data/y_train_resamp1.npy')

X_train_resamp2 = load('data/X_train_resamp2.npy')
y_train_resamp2 = load('data/y_train_resamp2.npy')

X_train_resamp3 = load('data/X_train_resamp3.npy')
y_train_resamp3 = load('data/y_train_resamp3.npy')

X_train_resamp4 = load('data/X_train_resamp4.npy')
y_train_resamp4 = load('data/y_train_resamp4.npy')

X_train_resamp5 = load('data/X_train_resamp5.npy')
y_train_resamp5 = load('data/y_train_resamp5.npy')


# ## Building the Pytorch Neural Network

# In[3]:


# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

from skorch import NeuralNetClassifier


# In[4]:


class ToxModelV1(nn.Module):
    
    def __init__(self, in_sz=5000, out_sz=4, nonlin=F.relu):
        super().__init__()
        self.nonlin = nonlin
        self.fc1 = nn.Linear(in_sz, 250)
        self.fc2 = nn.Linear(250, 65)
        self.fc3 = nn.Linear(65, out_sz)
        
    def forward(self, X):
        X = self.nonlin((self.fc1(X)))
        X = self.nonlin((self.fc2(X)))
        X = F.softmax((self.fc3(X)), dim=1)
        X0 = torch.sqrt(X[:,0]*(1-X[:,1]))
        X1 = (torch.sqrt(X[:,2]*(1-X[:,3])))
        X = X0, X1
        return torch.stack(X).transpose(0,1)
        
    


# In[5]:


torch.manual_seed(314)
ToxModelV1()


# In[6]:


# a less complex model with only two layers...

class ToxModelV2(nn.Module):
    
    def __init__(self, in_sz=5000, out_sz=4, nonlin=F.relu):
        super().__init__()
        self.nonlin = nonlin
        self.fc1 = nn.Linear(in_sz, 100)
        self.fc2 = nn.Linear(100, out_sz)
        
    def forward(self, X):
        X = self.nonlin((self.fc1(X)))
        X = F.softmax((self.fc2(X)), dim=1)
        X0 = torch.sqrt(X[:,0]*(1-X[:,1]))
        X1 = (torch.sqrt(X[:,2]*(1-X[:,3])))
        X = X0, X1
        return torch.stack(X).transpose(0,1)


# In[7]:


torch.manual_seed(314)
ToxModelV2()


# In[8]:


model = NeuralNetClassifier(ToxModelV1, train_split=None, iterator_train__shuffle=False,
                            optimizer=torch.optim.Adam, criterion = nn.CrossEntropyLoss)


# ## Grid Search Cross Validation

# First with the original data

# In[9]:


# Sklearn Grid Search

from sklearn.model_selection import GridSearchCV


params = { 'lr': [0.001,0.0001],
         'max_epochs':[5,10],
         'module__nonlin':[F.relu, torch.sigmoid],
         }

gs = GridSearchCV(model, params, cv=4, refit=True, scoring='accuracy', n_jobs=3)

gs.fit(X_train,y_train)

print(gs.best_score_, gs.best_params_)


# Now with the resampled data. First up is original data with SMOTE to oversample to minority class.

# In[10]:


params = { 'lr': [0.001,0.0001],
         'max_epochs':[5,10],
         'module__nonlin':[F.relu, torch.sigmoid],
         }

gs1 = GridSearchCV(model, params, cv=4, refit=True, scoring='accuracy', n_jobs=3)

gs1.fit(X_train_resamp1,y_train_resamp1)

print(gs1.best_score_, gs1.best_params_)


# Now for the data with undersampling (n_neighbors = 30) applied to the majority class first, then SMOTE to balance the minority class.

# In[11]:


params = { 'lr': [0.001,0.0001],
         'max_epochs':[5,10],
         'module__nonlin':[F.relu, torch.sigmoid],
         }

gs2 = GridSearchCV(model, params, cv=4, refit=True, scoring='accuracy', n_jobs=3)

gs2.fit(X_train_resamp2,y_train_resamp2)

print(gs2.best_score_, gs2.best_params_)


# Next for the data with undersampling of n_neighbors = 20

# In[12]:


params = { 'lr': [0.001,0.0001],
         'max_epochs':[5,10],
         'module__nonlin':[F.relu, torch.sigmoid],
         }

gs3 = GridSearchCV(model, params, cv=4, refit=True, scoring='accuracy', n_jobs=3)

gs3.fit(X_train_resamp3,y_train_resamp3)

print(gs3.best_score_, gs3.best_params_)


# Next for the data with undersampling of n_neighbors = 15

# In[13]:


params = { 'lr': [0.001,0.0001],
         'max_epochs':[5,10],
         'module__nonlin':[F.relu, torch.sigmoid],
         }

gs4 = GridSearchCV(model, params, cv=4, refit=True, scoring='accuracy', n_jobs=3)

gs4.fit(X_train_resamp4,y_train_resamp4)

print(gs4.best_score_, gs4.best_params_)


# Lastly, the data with undersampling of n_neighbors = 10

# In[14]:


params = { 'lr': [0.001,0.0001],
         'max_epochs':[5,10],
         'module__nonlin':[F.relu, torch.sigmoid],
         }

gs5 = GridSearchCV(model, params, cv=4, refit=True, scoring='accuracy', n_jobs=3)

gs5.fit(X_train_resamp5,y_train_resamp5)

print(gs5.best_score_, gs5.best_params_)


# # Testing

# In[15]:


test_preds_OG = gs.predict(X_test)

test_preds_1 = gs1.predict(X_test)

test_preds_2 = gs2.predict(X_test)

test_preds_3 = gs3.predict(X_test)

test_preds_4 = gs4.predict(X_test)

test_preds_5 = gs5.predict(X_test)


# In[16]:


test_pred_probs_OG = gs.predict_proba(X_test)

test_pred_probs_1 = gs1.predict_proba(X_test)

test_pred_probs_2 = gs2.predict_proba(X_test)

test_pred_probs_3 = gs3.predict_proba(X_test)

test_pred_probs_4 = gs4.predict_proba(X_test)

test_pred_probs_5 = gs5.predict_proba(X_test)


# ## Evaluation

# Classification report with the original data:

# In[17]:


from sklearn.metrics import classification_report


report_m1_OG = classification_report(y_test, test_preds_OG, output_dict=True)
print(report_m1_OG)


# Classification report with JUST SMOTE. <b> (No Undersampling) <b/>

# In[18]:


report_m1_1 = classification_report(y_test, test_preds_1, output_dict=True)
print(report_m1_1)


# Now for the resampled date in which the majority class was undersampled first, then SMOTE was applied to the minority class.

# Undersampling with n_neighbors = 25 :

# In[19]:


report_m1_2 = classification_report(y_test, test_preds_2, output_dict=True)
print(report_m1_2)


# Undersampling with n_neighbors = 20 :

# In[20]:


report_m1_3 = classification_report(y_test, test_preds_3, output_dict=True)
print(report_m1_3)


# Undersampling with n_neighbors = 15 :

# In[21]:


report_m1_4 = classification_report(y_test, test_preds_4, output_dict=True)
print(report_m1_4)


# Undersampling with n_neighbors = 10 :

# In[22]:


report_m1_5 = classification_report(y_test, test_preds_5, output_dict=True)
print(report_m1_5)


# ## ROC Curve

# In[23]:


from sklearn import metrics


# In[24]:


from sklearn.metrics import auc, roc_curve

from sklearn.utils import check_matplotlib_support


class RocCurveDisplay:
    """ROC Curve visualization.
    It is recommend to use :func:`~sklearn.metrics.plot_roc_curve` to create a
    visualizer. All parameters are stored as attributes.
    Read more in the :ref:`User Guide <visualizations>`.
    Parameters
    ----------
    fpr : ndarray
        False positive rate.
    tpr : ndarray
        True positive rate.
    roc_auc : float
        Area under ROC curve.
    estimator_name : str
        Name of estimator.
    Attributes
    ----------
    line_ : matplotlib Artist
        ROC Curve.
    ax_ : matplotlib Axes
        Axes with ROC Curve.
    figure_ : matplotlib Figure
        Figure containing the curve.
    """

    def __init__(self, fpr, tpr, roc_auc, estimator_name):
        self.fpr = fpr
        self.tpr = tpr
        self.roc_auc = roc_auc
        self.estimator_name = estimator_name

    def plot(self, ax=None, name=None, **kwargs):
        """Plot visualization
        Extra keyword arguments will be passed to matplotlib's ``plot``.
        Parameters
        ----------
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.
        name : str, default=None
            Name of ROC Curve for labeling. If `None`, use the name of the
            estimator.
        Returns
        -------
        display : :class:`~sklearn.metrics.plot.RocCurveDisplay`
            Object that stores computed values.
        """
        check_matplotlib_support('RocCurveDisplay.plot')
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        name = self.estimator_name if name is None else name

        line_kwargs = {
            'label': "{} (AUC = {:0.2f})".format(name, self.roc_auc)
        }
        line_kwargs.update(**kwargs)

        self.line_ = ax.plot(self.fpr, self.tpr, **line_kwargs)[0]
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc='lower right')

        self.ax_ = ax
        self.figure_ = ax.figure
        return self


def plot_roc_curve(estimator, X, y, pos_label=None, sample_weight=None,
                   drop_intermediate=True, response_method="auto",
                   name=None, ax=None, **kwargs):
    """Plot Receiver operating characteristic (ROC) curve.
    Extra keyword arguments will be passed to matplotlib's `plot`.
    Read more in the :ref:`User Guide <visualizations>`.
    Parameters
    ----------
    estimator : estimator instance
        Trained classifier.
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Input values.
    y : array-like of shape (n_samples,)
        Target values.
    pos_label : int or str, default=None
        The label of the positive class.
        When `pos_label=None`, if y_true is in {-1, 1} or {0, 1},
        `pos_label` is set to 1, otherwise an error will be raised.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    drop_intermediate : boolean, default=True
        Whether to drop some suboptimal thresholds which would not appear
        on a plotted ROC curve. This is useful in order to create lighter
        ROC curves.
    response_method : {'predict_proba', 'decision_function', 'auto'} \
    default='auto'
        Specifies whether to use :term:`predict_proba` or
        :term:`decision_function` as the target response. If set to 'auto',
        :term:`predict_proba` is tried first and if it does not exist
        :term:`decision_function` is tried next.
    name : str, default=None
        Name of ROC Curve for labeling. If `None`, use the name of the
        estimator.
    ax : matplotlib axes, default=None
        Axes object to plot on. If `None`, a new figure and axes is created.
    Returns
    -------
    display : :class:`~sklearn.metrics.RocCurveDisplay`
        Object that stores computed values.
    """
    check_matplotlib_support('plot_roc_curve')

    if response_method not in ("predict_proba", "decision_function", "auto"):
        raise ValueError("response_method must be 'predict_proba', "
                         "'decision_function' or 'auto'")

    if response_method != "auto":
        prediction_method = getattr(estimator, response_method, None)
        if prediction_method is None:
            raise ValueError(
                "response method {} is not defined".format(response_method))
    else:
        predict_proba = getattr(estimator, 'predict_proba', None)
        decision_function = getattr(estimator, 'decision_function', None)
        prediction_method = predict_proba or decision_function

        if prediction_method is None:
            raise ValueError('response methods not defined')

    y_pred = prediction_method(X)

    if y_pred.ndim != 1:
        if y_pred.shape[1] > 2:
            raise ValueError("Estimator should solve a "
                             "binary classification problem")
        y_pred = y_pred[:, 1]
    fpr, tpr, _ = roc_curve(y, y_pred, pos_label=pos_label,
                            sample_weight=sample_weight,
                            drop_intermediate=drop_intermediate)
    roc_auc = auc(fpr, tpr)
    viz = RocCurveDisplay(fpr, tpr, roc_auc, estimator.__class__.__name__)
    return viz.plot(ax=ax, name=name, **kwargs)


# In[25]:


plt.style.use('ggplot')


# In[26]:


fig,ax = plt.subplots(figsize = (6,6))
out = plot_roc_curve(gs1, X_test, y_test, ax = ax, drop_intermediate = True) 
ax.plot([0,1],[0,1], 'k--', alpha = 0.5);
ax.set_xlim(-0.05,1)
ax.set_ylim(0,1.05)
ax.set_xlabel('False Positive Rate', fontsize = 14)
ax.set_ylabel('True Positive Rate', fontsize = 14)
ax.set_title("ROC of Model 1 trained on Orignal Data Balanced w/ SMOTE")
ax.legend(fontsize = 12);


# In[27]:


fig,ax = plt.subplots(figsize = (6,6))
out = plot_roc_curve(gs2, X_test, y_test, ax = ax, drop_intermediate = True) 
ax.plot([0,1],[0,1], 'k--', alpha = 0.5);
ax.set_xlim(-0.05,1)
ax.set_ylim(0,1.05)
ax.set_xlabel('False Positive Rate', fontsize = 14)
ax.set_ylabel('True Positive Rate', fontsize = 14)
ax.set_title("ROC of Model 1 trained w/ Undersampled (n_neighbors = 25)")
ax.legend(fontsize = 12);


# In[28]:


fig,ax = plt.subplots(figsize = (6,6))
out = plot_roc_curve(gs3, X_test, y_test, ax = ax, drop_intermediate = True) 
ax.plot([0,1],[0,1], 'k--', alpha = 0.5);
ax.set_xlim(-0.05,1)
ax.set_ylim(0,1.05)
ax.set_xlabel('False Positive Rate', fontsize = 14)
ax.set_ylabel('True Positive Rate', fontsize = 14)
ax.set_title("ROC of Model 1 trained w/ Undersampled (n_neighbors = 20)")
ax.legend(fontsize = 12);


# In[29]:


fig,ax = plt.subplots(figsize = (6,6))
out = plot_roc_curve(gs4, X_test, y_test, ax = ax, drop_intermediate = True) 
ax.plot([0,1],[0,1], 'k--', alpha = 0.5);
ax.set_xlim(-0.05,1)
ax.set_ylim(0,1.05)
ax.set_xlabel('False Positive Rate', fontsize = 14)
ax.set_ylabel('True Positive Rate', fontsize = 14)
ax.set_title("ROC of Model 1 trained w/ Undersampled (n_neighbors = 15)")
ax.legend(fontsize = 12);


# In[30]:


fig,ax = plt.subplots(figsize = (6,6))
out = plot_roc_curve(gs5, X_test, y_test, ax = ax, drop_intermediate = True) 
ax.plot([0,1],[0,1], 'k--', alpha = 0.5);
ax.set_xlim(-0.05,1)
ax.set_ylim(0,1.05)
ax.set_xlabel('False Positive Rate', fontsize = 14)
ax.set_ylabel('True Positive Rate', fontsize = 14)
ax.set_title("ROC of Model 1 trained w/ Undersampled (n_neighbors = 10)")
ax.legend(fontsize = 12);


# # A Less Complex Model
# 
# Now the Tox Model V2 is trained and tested to see if a less complex model will produce better results

# In[31]:


model = NeuralNetClassifier(ToxModelV2, train_split=None, iterator_train__shuffle=False,
                            optimizer=torch.optim.Adam, criterion = nn.CrossEntropyLoss)


# ## Grid Search Cross Validation

# First with the original data

# In[32]:


# Sklearn Grid Search

from sklearn.model_selection import GridSearchCV


params = { 'lr': [0.001,0.0001],
         'max_epochs':[5,10],
         'module__nonlin':[F.relu, torch.sigmoid],
         }

gs = GridSearchCV(model, params, cv=4, refit=True, scoring='accuracy', n_jobs=3)

gs.fit(X_train,y_train)

print(gs.best_score_, gs.best_params_)


# Now with the resampled data. First up is original data with SMOTE to oversample to minority class.

# In[33]:


params = { 'lr': [0.001,0.0001],
         'max_epochs':[5,10],
         'module__nonlin':[F.relu, torch.sigmoid],
         }

gs1 = GridSearchCV(model, params, cv=4, refit=True, scoring='accuracy', n_jobs=3)

gs1.fit(X_train_resamp1,y_train_resamp1)

print(gs1.best_score_, gs1.best_params_)


# Now for the data with undersampling (n_neighbors = 30) applied to the majority class first, then SMOTE to balance the minority class.

# In[34]:


params = { 'lr': [0.001,0.0001],
         'max_epochs':[5,10],
         'module__nonlin':[F.relu, torch.sigmoid],
         }

gs2 = GridSearchCV(model, params, cv=4, refit=True, scoring='accuracy', n_jobs=3)

gs2.fit(X_train_resamp2,y_train_resamp2)

print(gs2.best_score_, gs2.best_params_)


# Next for the data with undersampling of n_neighbors = 20

# In[35]:


params = { 'lr': [0.001,0.0001],
         'max_epochs':[5,10],
         'module__nonlin':[F.relu, torch.sigmoid],
         }

gs3 = GridSearchCV(model, params, cv=4, refit=True, scoring='accuracy', n_jobs=3)

gs3.fit(X_train_resamp3,y_train_resamp3)

print(gs3.best_score_, gs3.best_params_)


# Next for the data with undersampling of n_neighbors = 15

# In[36]:


params = { 'lr': [0.001,0.0001],
         'max_epochs':[5,10],
         'module__nonlin':[F.relu, torch.sigmoid],
         }

gs4 = GridSearchCV(model, params, cv=4, refit=True, scoring='accuracy', n_jobs=3)

gs4.fit(X_train_resamp4,y_train_resamp4)

print(gs4.best_score_, gs4.best_params_)


# Lastly, the data with undersampling of n_neighbors = 10

# In[37]:


params = { 'lr': [0.001,0.0001],
         'max_epochs':[5,10],
         'module__nonlin':[F.relu, torch.sigmoid],
         }

gs5 = GridSearchCV(model, params, cv=4, refit=True, scoring='accuracy', n_jobs=3)

gs5.fit(X_train_resamp5,y_train_resamp5)

print(gs5.best_score_, gs5.best_params_)


# # Testing

# In[38]:


test_preds_OG = gs.predict(X_test)

test_preds_1 = gs1.predict(X_test)

test_preds_2 = gs2.predict(X_test)

test_preds_3 = gs3.predict(X_test)

test_preds_4 = gs4.predict(X_test)

test_preds_5 = gs5.predict(X_test)


# In[39]:


test_pred_probs_OG = gs.predict_proba(X_test)

test_pred_probs_1 = gs1.predict_proba(X_test)

test_pred_probs_2 = gs2.predict_proba(X_test)

test_pred_probs_3 = gs3.predict_proba(X_test)

test_pred_probs_4 = gs4.predict_proba(X_test)

test_pred_probs_5 = gs5.predict_proba(X_test)


# # Evaluation

# Classification report with the original data:

# In[40]:


report_m2_OG = classification_report(y_test, test_preds_OG, output_dict=True)
print(report_m2_OG)


# Classification report with JUST SMOTE. <b> (No Undersampling) <b/>

# In[41]:


report_m2_1 = classification_report(y_test, test_preds_1, output_dict=True)
print(report_m2_1)


# Now for the resampled date in which the majority class was undersampled first, then SMOTE was applied to the minority class.

# Undersampling with n_neighbors = 25 :

# In[42]:


report_m2_2 = classification_report(y_test, test_preds_2, output_dict=True)
print(report_m2_2)


# Undersampling with n_neighbors = 20 :

# In[43]:


report_m2_3 = classification_report(y_test, test_preds_3, output_dict=True)
print(report_m2_3)


# Undersampling with n_neighbors = 15 :

# In[44]:


report_m2_4 = classification_report(y_test, test_preds_4, output_dict=True)
print(report_m2_4)


# Undersampling with n_neighbors = 10 :

# In[45]:


report_m2_5 = classification_report(y_test, test_preds_5, output_dict=True)
print(report_m2_5)


# ### ROC Curve

# fig,ax = plt.subplots(figsize = (6,6))
# out = plot_roc_curve(gs, X_test, y_test, ax = ax, drop_intermediate = True) 
# ax.plot([0,1],[0,1], 'k--', alpha = 0.5);
# ax.set_xlim(-0.05,1)
# ax.set_ylim(0,1.05)
# ax.set_xlabel('False Positive Rate', fontsize = 14)
# ax.set_ylabel('True Positive Rate', fontsize = 14)
# ax.set_title("ROC of Model 2 trained on Original Data")
# ax.legend(fontsize = 12);

# In[46]:


fig,ax = plt.subplots(figsize = (6,6))
out = plot_roc_curve(gs1, X_test, y_test, ax = ax, drop_intermediate = True) 
ax.plot([0,1],[0,1], 'k--', alpha = 0.5);
ax.set_xlim(-0.05,1)
ax.set_ylim(0,1.05)
ax.set_xlabel('False Positive Rate', fontsize = 14)
ax.set_ylabel('True Positive Rate', fontsize = 14)
ax.set_title("ROC of Model 2 trained on Orignal Data Balanced w/ SMOTE")
ax.legend(fontsize = 12);


# In[47]:


fig,ax = plt.subplots(figsize = (6,6))
out = plot_roc_curve(gs2, X_test, y_test, ax = ax, drop_intermediate = True) 
ax.plot([0,1],[0,1], 'k--', alpha = 0.5);
ax.set_xlim(-0.05,1)
ax.set_ylim(0,1.05)
ax.set_xlabel('False Positive Rate', fontsize = 14)
ax.set_ylabel('True Positive Rate', fontsize = 14)
ax.set_title("ROC of Model 2 trained w/ Undersampled (n_neighbors = 25)")
ax.legend(fontsize = 12);


# In[48]:


fig,ax = plt.subplots(figsize = (6,6))
out = plot_roc_curve(gs3, X_test, y_test, ax = ax, drop_intermediate = True) 
ax.plot([0,1],[0,1], 'k--', alpha = 0.5);
ax.set_xlim(-0.05,1)
ax.set_ylim(0,1.05)
ax.set_xlabel('False Positive Rate', fontsize = 14)
ax.set_ylabel('True Positive Rate', fontsize = 14)
ax.set_title("ROC of Model 2 trained w/ Undersampled (n_neighbors = 20)")
ax.legend(fontsize = 12);


# In[49]:


fig,ax = plt.subplots(figsize = (6,6))
out = plot_roc_curve(gs4, X_test, y_test, ax = ax, drop_intermediate = True) 
ax.plot([0,1],[0,1], 'k--', alpha = 0.5);
ax.set_xlim(-0.05,1)
ax.set_ylim(0,1.05)
ax.set_xlabel('False Positive Rate', fontsize = 14)
ax.set_ylabel('True Positive Rate', fontsize = 14)
ax.set_title("ROC of Model 2 trained w/ Undersampled (n_neighbors = 15)")
ax.legend(fontsize = 12);


# In[50]:


fig,ax = plt.subplots(figsize = (6,6))
out = plot_roc_curve(gs5, X_test, y_test, ax = ax, drop_intermediate = True) 
ax.plot([0,1],[0,1], 'k--', alpha = 0.5);
ax.set_xlim(-0.05,1)
ax.set_ylim(0,1.05)
ax.set_xlabel('False Positive Rate', fontsize = 14)
ax.set_ylabel('True Positive Rate', fontsize = 14)
ax.set_title("ROC of Model 2 trained w/ Undersampled (n_neighbors = 10)")
ax.legend(fontsize = 12);


# # Final Evaluation of Performance

# First we take a looking at the test set...

# In[51]:


pos = (y_test==1).sum()

neg = (y_test==0).sum()

plt.bar(['Non-Toxic','Toxic'], [neg,pos], align='center', alpha=1)
plt.ylabel('Number of Samples')
plt.title('Test Set')

plt.show()


# As you can see above, while the training set was balanced, the test set remained imbalanced.
# 
# Now we look at both models performances.

# # Accuracy

# In[52]:


methods = ['Original', 'Method 1', 'Method 2', 'Method 3', 'Method 4', 'Method 5']


# In[53]:


m1_acc = np.array([report_m1_OG['accuracy'],report_m1_1['accuracy'],report_m1_2['accuracy'],
          report_m1_3['accuracy'],report_m1_4['accuracy'],report_m1_5['accuracy']])

plt.bar(methods, m1_acc, align='center', alpha=1)
plt.ylim(0.35, 1.0)
plt.ylabel('Accuracy')
plt.title('Model 1 Accuracy')

plt.show()


# In[54]:


m2_acc = np.array([report_m2_OG['accuracy'],report_m2_1['accuracy'],report_m2_2['accuracy'],
          report_m2_3['accuracy'],report_m2_4['accuracy'],report_m2_5['accuracy']])

plt.bar(methods, m2_acc, align='center', alpha=1)
plt.ylim(0.35, 1.0)
plt.ylabel('Accuracy')
plt.title('Model 2 Accuracy')

plt.show()


# In terms of accuracy, the original data performed the best. However, next when we look at precision we will see why this is very misleading...

# # Precision

# In[55]:


m1_nontox_prec = np.array([report_m1_OG['0']['precision'],report_m1_1['0']['precision'],report_m1_2['0']['precision'],
          report_m1_3['0']['precision'],report_m1_4['0']['precision'],report_m1_5['0']['precision']])

m1_tox_prec = np.array([report_m1_OG['1']['precision'],report_m1_1['1']['precision'],report_m1_2['1']['precision'],
          report_m1_3['1']['precision'],report_m1_4['1']['precision'],report_m1_5['1']['precision']])

N = 6
ind = np.arange(N) 
width = 0.35       
plt.bar(ind, m1_nontox_prec, width, label='Non-Toxic')
plt.bar(ind + width, m1_tox_prec, width,
    label='Toxic')

plt.ylabel('Precision')
plt.xlabel('Data Preprocessing Type')
plt.title('Model 1 Precision')

plt.xticks(ind + width / 2, ('Original', 'Method 1', 'Method 2', 'Method 3', 'Method 4', 'Method 5'))
plt.legend(loc='best')
plt.show()


# In[56]:


m2_nontox_prec = np.array([report_m2_OG['0']['precision'],report_m2_1['0']['precision'],report_m2_2['0']['precision'],
          report_m2_3['0']['precision'],report_m2_4['0']['precision'],report_m2_5['0']['precision']])

m2_tox_prec = np.array([report_m2_OG['1']['precision'],report_m2_1['1']['precision'],report_m2_2['1']['precision'],
          report_m2_3['1']['precision'],report_m2_4['1']['precision'],report_m2_5['1']['precision']])

N = 6
ind = np.arange(N) 
width = 0.35       
plt.bar(ind, m2_nontox_prec, width, label='Non-Toxic')
plt.bar(ind + width, m2_tox_prec, width,
    label='Toxic')

plt.ylabel('Precision')
plt.xlabel('Data Preprocessing Type')
plt.title('Model 1 Precision')

plt.xticks(ind + width / 2, ('Original', 'Method 1', 'Method 2', 'Method 3', 'Method 4', 'Method 5'))
plt.legend(loc='best')
plt.show()


# From viewing the precisions we see that when trained with the original unbalanced data, both models classify every test sample as non-toxic. Thus it did not perform better than a random classifier. This is why the accuracy was so high for the models trained with the original data. The accuracy was aproximately <b>93%</b> because <b>93%</b> of the test molecules were non-toxic.
# 
# 
# Method 1 of model 2 faired the best with <b>0.96</b> precision on the non-toxic class and <b>0.41</b> on the toxic class. Interpretation of this is that out of all of the test molecules this model labeled as toxic, <b>41%</b> of them were actually toxic. More so out of all the samples the model labeled as non-toxic, <b>96%</b> of them were actually non-toxic.

# # Recall

# In[57]:


m1_nontox_recall = np.array([report_m1_OG['0']['recall'],report_m1_1['0']['recall'],report_m1_2['0']['recall'],
          report_m1_3['0']['recall'],report_m1_4['0']['recall'],report_m1_5['0']['recall']])

m1_tox_recall = np.array([report_m1_OG['1']['recall'],report_m1_1['1']['recall'],report_m1_2['1']['recall'],
          report_m1_3['1']['recall'],report_m1_4['1']['recall'],report_m1_5['1']['recall']])

N = 6
ind = np.arange(N) 
width = 0.35       
plt.bar(ind, m1_nontox_prec, width, label='Non-Toxic')
plt.bar(ind + width, m1_tox_prec, width,
    label='Toxic')

plt.ylabel('Recall')
plt.xlabel('Data Preprocessing Type')
plt.title('Model 1 Recall')

plt.xticks(ind + width / 2, ('Original', 'Method 1', 'Method 2', 'Method 3', 'Method 4', 'Method 5'))
plt.legend(loc='best')
plt.show()


# In[58]:


m2_nontox_recall = np.array([report_m2_OG['0']['recall'],report_m2_1['0']['recall'],report_m2_2['0']['recall'],
          report_m2_3['0']['recall'],report_m2_4['0']['recall'],report_m2_5['0']['recall']])

m2_tox_recall = np.array([report_m2_OG['1']['recall'],report_m2_1['1']['recall'],report_m2_2['1']['recall'],
          report_m2_3['1']['recall'],report_m2_4['1']['recall'],report_m2_5['1']['recall']])

N = 6
ind = np.arange(N) 
width = 0.35       
plt.bar(ind, m1_nontox_prec, width, label='Non-Toxic')
plt.bar(ind + width, m1_tox_prec, width,
    label='Toxic')

plt.ylabel('Recall')
plt.xlabel('Data Preprocessing Type')
plt.title('Model 2 Recall')

plt.xticks(ind + width / 2, ('Original', 'Method 1', 'Method 2', 'Method 3', 'Method 4', 'Method 5'))
plt.legend(loc='best')
plt.show()


# From viewing the recalls we see that both models performed the same in terms of highest recall. Both models trained with method 1 data had a recall of <b>0.4</b> for the toxic class and <b>0.96</b> for the non-toxic class. 
# 
# This can be interpreted as out of all the truly toxic molecules in the test set, both models correctly labeled <b>40%</b> of them. Moreover out of all of the truly non-toxic molecules in the test set, both models correctly labeled <b>96%</b> of them.
# 
# It is worth noting that the recall for the toxic class increased to <b>93%</b> when method 2 data was used to train the models, however the non-toxic class suffered with a recall of <b>38%</b> with model 1 and <b>37%</b> with model 2.

# # AUROC

# In[59]:


from IPython.display import Image
Image(filename='data/AUROC tables.png')


# Above are tables from the report on this research written in Latex showing the area under the ROC curve. The following is an excerpt from that report disscussing the AUROC.
# 
# "When examining the performance of a classification model, one of the most important metrics to refer to is the ROC curve and the area under it. The ROC is a probability curve and the area under it is a measure of separability, or how capable the model is of distinguishing between classes. When examining each instance of model 1 and model 2’s ROC curve, there was not much of a difference. The area under the ROC curves were the same for both models besides method 5, which had an AUC of <b>0.74</b> for model 1 and <b>0.76</b> for model 2. The method that had the largest AUC was method 3, with an AUC of <b>0.81</b>. Therefore data preprocessing method 3 created optimal training data for the models, thus the models trained with method 3 data were the best at distinguishing between toxic and non-toxic molecules."
