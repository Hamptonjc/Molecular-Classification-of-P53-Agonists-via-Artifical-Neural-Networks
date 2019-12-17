#!/usr/bin/env python
# coding: utf-8

# # Molecular Classification of P53 Agonists via Artifical Neural Network Part 1 (EDA & Data Preprocessing)
# 
# <br>
# <center><b>Author: Jonathan Hampton</b></center>
# 
# <br>
# <center><b>November 2019</b></center>

# In[38]:


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


# ## Data Load/Exploratory Data Analysis
# 
# First the data is loaded from the <b>tox21</b> CSV into a pandas dataframe.

# In[2]:


df = pd.read_csv('Data/tox21.csv') #CSV data into a pandas dataframe
df.head()


# Now some basic checks are done...

# In[4]:


df['SR-p53'].isna().sum() #check missing values in p53 column


# In[5]:


df.shape


# For this classification problem we are only predicting one class (SR-p53) so a new data frame is made with on the information that will be used.

# In[6]:


# New DF with columns we want to use
tox = pd.DataFrame([df['mol_id'],df['SR-p53'],df['smiles']]).transpose()
tox.head()


# Now some basic cleaning and checks are made...

# In[7]:


# drop missing values
tox = tox.dropna()


# In[8]:


# Check that there is no missing values
tox['SR-p53'].isna().sum()


# In[10]:


# Shape after missing values are removed
tox.shape


# In[9]:


# How many 1's
tox['SR-p53'].sum()


# In[11]:


# How many 0's
6774 - 423 


# ## Convert SMILES To Fingerprints
# 
# To make the feature data more suitable for a neural network we convert the molecule's SMILES to <b>Extended Connectivity Fingerprints</b> (ECFP). This encodes the SMILES into a vector of 1's and 0's that store the molecule's makeup and structure.

# In[12]:


smile_sz = tox['smiles'].apply(len)
smile_sz.describe() #check sizes/info on smiles


# In[13]:


# Convert smiles to Extended Connectivity Fingerprints
smiles = tox['smiles']
mols = [Chem.MolFromSmiles(smile) for smile in smiles]
feat = dc.feat.CircularFingerprint(size=5000)
chem_fp = feat.featurize(mols)


# In[14]:


fp_shape = chem_fp.shape # Checking Fingerprint shape
fp_shape


# In[15]:


fp_sz = fp_shape[1] # calling second number in tuple to get length of fingerprints
fp_sz


# In[16]:


tox['ECFP'] = np.split(chem_fp,fp_shape[0]) #Creating new column in tox DF to store ECFP
tox.head()


# ## Test/Train Split
# 
# Now the data is split into training and testing sets with a test size of <b>20%</b>.

# In[17]:


label = tox['SR-p53'] # grab the p53 column as our labels
features = tox.drop(columns=['SR-p53']) # Use other columns as our features


# In[18]:


# Pull out a final test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features['ECFP'], label, test_size=0.2, random_state=101)


# In[19]:


# Converting to numpy array

X_train = X_train.to_numpy()
y_train = y_train.to_numpy()

X_test = X_test.to_numpy()
y_test = y_test.to_numpy()


# In[20]:


# Concatenating the array of arrays into a single array
X_train = np.concatenate( X_train, axis=0 )
X_test = np.concatenate( X_test, axis=0 )


# In[21]:


# Changing the dtypes
y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)


# In[57]:


# Number of 'toxic' samples in the training set
y_train.sum()


# In[58]:


# Number of 'toxic' samples in the test set
y_test.sum()


# In[56]:


# Number of 'non-toxic' samples in the training set
y_train.shape[0] - y_train.sum()


# In[23]:


# Number of 'non-toxic' samples in the test set
y_test.shape[0] - y_test.sum()


# ## Balancing Data
# 
# Above it is evident that the data is extremely unbalanced. To combat this we try several methods to balance the data. First, Synthetic Minority Over-sampling Technique(SMOTE) is applied to the original data, without applying undersampling techniques to the majority class.

# In[24]:


from imblearn.over_sampling import SMOTE


# In[25]:


sm = SMOTE(sampling_strategy='minority', random_state=7)


# In[26]:


X_train_resamp1, y_train_resamp1 = sm.fit_sample(X_train, y_train)


# Next a nearest neighbors undersampling technique is applied to the majority.

# In[27]:


from imblearn.under_sampling import RepeatedEditedNearestNeighbours


# First with n_neighbors = 25

# In[28]:


enn = RepeatedEditedNearestNeighbours(sampling_strategy='majority',n_neighbors=25, n_jobs=3, random_state=101)
X_resamp2, y_resamp2 = enn.fit_resample(X_train, y_train)


# In[29]:


#number of resampled majority samples
y_resamp2.shape[0] - y_resamp2.sum()


# Now n_neighbors is decreased to 20

# In[30]:


enn = RepeatedEditedNearestNeighbours(sampling_strategy='majority',n_neighbors=20, n_jobs=3, random_state=101)
X_resamp3, y_resamp3 = enn.fit_resample(X_train, y_train)


# In[31]:


#number of resampled majority samples
y_resamp3.shape[0] - y_resamp3.sum()


# Now n_neighbors is decreased to 15

# In[32]:


enn = RepeatedEditedNearestNeighbours(sampling_strategy='majority',n_neighbors=15, n_jobs=3, random_state=101)
X_resamp4, y_resamp4 = enn.fit_resample(X_train, y_train)


# In[33]:


#number of resampled majority samples
y_resamp4.shape[0] - y_resamp4.sum()


# Lastly n_neighbors is decreased to 10

# In[34]:


enn = RepeatedEditedNearestNeighbours(sampling_strategy='majority',n_neighbors=10, n_jobs=3, random_state=101)
X_resamp5, y_resamp5 = enn.fit_resample(X_train, y_train)


# In[35]:


#number of resampled majority samples
y_resamp5.shape[0] - y_resamp5.sum()


# Now, these new undersampled datasets are passed through SMOTE to balance the minority class

# In[36]:


X_train_resamp2, y_train_resamp2 = sm.fit_sample(X_resamp2, y_resamp2)

X_train_resamp3, y_train_resamp3 = sm.fit_sample(X_resamp3, y_resamp3)

X_train_resamp4, y_train_resamp4 = sm.fit_sample(X_resamp4, y_resamp4)

X_train_resamp5, y_train_resamp5 = sm.fit_sample(X_resamp5, y_resamp5)


# In[37]:


# Saves
from numpy import save

save('Data/X_train_OG.npy', X_train)
save('Data/y_train_OG.npy', y_train)

save('Data/X_test.npy', X_test)
save('Data/y_test.npy', y_test)

save('Data/X_train_resamp1.npy', X_train_resamp1)
save('Data/y_train_resamp1.npy', y_train_resamp1)

save('Data/X_train_resamp2.npy', X_train_resamp2)
save('Data/y_train_resamp2.npy', y_train_resamp2)

save('Data/X_train_resamp3.npy', X_train_resamp3)
save('Data/y_train_resamp3.npy', y_train_resamp3)

save('Data/X_train_resamp4.npy', X_train_resamp4)
save('Data/y_train_resamp4.npy', y_train_resamp4)

save('Data/X_train_resamp5.npy', X_train_resamp5)
save('Data/y_train_resamp5.npy', y_train_resamp5)

