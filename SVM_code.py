# STEP 1 - Importing Libraries

import numpy as np
import sklearn
import pandas as pd
import matplotlib
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
import imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xlwings as xw 
from pandas import *

import warnings
warnings.filterwarnings("ignore")

# STEP 2 - Reading and cleaning file
# Read data
data = pd.read_excel("C:\Finance_model\OLogit Synthetic Credit Rating Model.xlsm",sheet_name='2023 Q4')

# Data cleaning techniques
data = data.rename(columns=data.iloc[1]).drop(data.index[0])
data = data.iloc[1:, :]

# STEP 3 - Data Preprocessing
# Dropping unnecessary columns
data = data.drop(['Company','Ticker','Industry','Numeric credit rating','Filtered Rating'], axis=1)

# Handling missing values
data=data.dropna(axis=1,how='all')
data = data.fillna(0)

# Financial measure are all the measures which have been derieved from the 5 values given
financial_measures = data.iloc[:,6:35]
financial_measures = financial_measures.fillna(0)

# STEP 4 - Specifying and extracting the financial measures from the sheet  
def remove_none_entries(data):
  """Removes entries with None values from a nested list structure.

  Args:
      data (list): The nested list containing data.

  Returns:
      list: A new list with entries containing None values removed.
  """
  filtered_data = []
  for entry in data:
    # Check if any value in the inner list is not None
    if any(value is not None for value in entry[0]):
      filtered_data.append(entry)
  return filtered_data

                                                                                
ws = xw.Book("OLogit Synthetic Credit Rating Model.xlsm").sheets['Ratios'] 
table = ws.range("I12:M40").value 
table = [[inner_list] for inner_list in table]

# Remove Null entries if any
table = remove_none_entries(table)

company = [cell_value for cell_value in list(ws.range('B12:B40').value) if cell_value is not None]

def remove_na_entries(table, company):
    """Removes entries with 'NA' values from table and its corresponding company list."""
    filtered_table = []
    filtered_company = []

    for i in range(len(table)):
        if 'NA' not in table[i][0]:
            filtered_table.append(table[i])
            filtered_company.append(company[i])

    return filtered_table, filtered_company

f_table, f_company = remove_na_entries(table, company)

# STEP 5 - Prediction of model using SVM
## Using SVM to predict credit ratings and their probabilities of taking place ##
np.random.seed(0)
X=financial_measures
y=data['S&P credit rating'].values

a=[]

# Resampling the target variables as bias exists using over sampling
sampling_strategy = "not minority"
over = RandomOverSampler(sampling_strategy=sampling_strategy,random_state=42)
steps = [('o',over)]
pipeline = Pipeline(steps=steps)
X, y = pipeline.fit_resample(X, y)

# Binarizing the variables for reference in future
lb = LabelBinarizer()
y_encoded = lb.fit_transform(y)

# Using Bagging Classifier after oversampling 
clf = BalancedBaggingClassifier(estimator=GradientBoostingClassifier(n_estimators=200), random_state=42)
clf.fit(X, y)

# Using SVM in an 'One-vs-Rest' manner with hypertuned parameters
clf = OneVsRestClassifier(SVC(C=200,gamma=4,coef0=1,tol=0.001,class_weight='balanced',degree=2,random_state=42,probability=True),n_jobs=1) 
clf.fit(X, y)

# STEP 6 - Prediction of Probabilities 
# Function to predict probabilities for all credit ratings
def predict_all_probabilities(X):
    predicted_probs = clf.predict_proba(X)
    probabilities = dict(zip(lb.classes_, predicted_probs[0]/ np.sum(predicted_probs[0])))
    return probabilities

probability_plot=[]
# For prediction
for i in f_table:
   new_data = np.array(i)
   rating_probs = predict_all_probabilities(new_data)
   rating_order = ['AAA','AA+','AA','AA-','A+','A','A-','BBB+','BBB','BBB-','BB+','BB','BB-','B+','B','B-','CCC+','CCC','CCC-','CC']
   sorted_ratings_probs = {rating: rating_probs[rating] for rating in rating_order if rating in rating_probs}
    
   # Normalizing the probabilities 
   transformed_probs = {rating: np.power(prob,0.3) for rating, prob in sorted_ratings_probs.items()}
   total_prob = sum(transformed_probs.values())
   normalized_probs = {rating: prob / total_prob for rating, prob in transformed_probs.items()}

   ratings = list(normalized_probs.keys())
   probabilities = [prob * 100 for prob in normalized_probs.values()] 
   
   probability_plot.append(probabilities)

# STEP 7 - Display the probabilities as a dataframe
df = DataFrame(probability_plot, index=f_company,columns=ratings)
df = df.applymap(lambda x: "{:.2f}%".format(x))

wb = xw.Book("OLogit Synthetic Credit Rating Model.xlsm").sheets['Results_SVM'] 
wb.range("B10:V37").clear_contents()
wb.range("B10").value = df
