  #!/usr/bin/python

import sys
import pickle
sys.path.append("tools/")
#sys.path.append("../feature_selection/")

###############################
###                         ###
###     MINI  RIAR          ###
###                         ###
###############################

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from email_preprocess import preprocess
from sklearn import tree
from sklearn import cross_validation

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
###Used different features_list's to do the initial tasks
#features_list = ['poi','salary','bonus','from_poi_to_this_person', 'total_payments', 'from_this_person_to_poi']
#features_list = ['poi','salary','bonus'] # You will need to use more features
features_list = ['poi', 'salary', 'bonus', 'deferral_payments', 'total_payments', 'loan_advances', 
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 
                 'exercised_stock_options', 'long_term_incentive', 'restricted_stock', 'director_fees',
                 'shared_receipt_with_poi', 'fraction_to_poi', 'fraction_from_poi','bonus_salary_ratio']


###Initial use of DecisionTreeClassifier, and trying out the database
###creating the different test and train segments and printing the top 150 features

clf = tree.DecisionTreeClassifier()
features_train, features_test, labels_train, labels_test = preprocess()
clf = clf.fit(features_train, labels_train)
importance = clf.feature_importances_
#The features are always randomly permuted at each split. Therefore, the best found split may vary, even with the same training data
#n_features = The number of features when fit is performed.
#The feature importances. The higher, the more important the feature. The importance of a feature is computed as the (normalized) total
#reduction of the criterion brought by that feature. It is also known as the Gini importance 
#print features_train[0]
#features_150=features_train[:150]
features_all=features_train[0]
num_features=0
c=0
for feature in importance:
   if feature > 0:
       #print (feature)
       #print ('number: ',c)
       num_features+=1
   c+=1
print "features all ;  ", num_features, "all features:  ", c


### Load the dictionary containing the dataset

data_dict = pickle.load( open("final_project_dataset.pkl", "r") )

num_POI=0

missing_values = {'poi':0, 'salary':0, 'bonus':0, 'deferral_payments':0, 'total_payments':0, 'loan_advances':0, 
                 'restricted_stock_deferred':0, 'deferred_income':0, 'total_stock_value':0, 'expenses':0, 
                 'exercised_stock_options':0, 'long_term_incentive':0, 'restricted_stock':0, 'director_fees':0,
                 'shared_receipt_with_poi':0}

c=0
for key in data_dict:
    if data_dict[key]['poi']==1:
        num_POI +=1
    for key2 in missing_values:
        if data_dict[key][key2]=='NaN':
            missing_values[key2] +=1

    '''
    if data_dict[key]['deferral_payments'] >=0:
        print "1: ",  data_dict[key]['deferral_payments']
    if data_dict[key]['restricted_stock_deferred'] >=0:
        print "2: ", data_dict[key]['restricted_stock_deferred']
    ''' 
    c+=1
print "num POIs ;  ", num_POI
for k in missing_values:
        print k, ":   ", missing_values[k]
         

### Task 2: Remove outliers


import matplotlib.pyplot
print len(data_dict)

### read in data dictionary, convert to numpy array/ remove Total line
#After reveiwing the dats- Removed the TOTAL Line as well as the user The Travel Agency in the Park
data_dict.pop('TOTAL',0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")

## Utilized the following features_list for plotting and other investigation
features_list = ['poi','salary','bonus','from_poi_to_this_person', 'total_payments', 'from_this_person_to_poi']
data = featureFormat(data_dict, features_list)

#Try plotting different features to see if there are any outliers/ anomalies
'''

I haveplotted numerous combinations, and have commented this out to see if it will work with the tester code

from matplotlib import cm

### your code below
for point in data:
    poi = point[0]
    salary = point[1]
    bonus = point[2]
    #from_poi = point[3]
    #total_payments = point[4]
    #to_poi = point[5]
       
    matplotlib.pyplot.scatter(salary, bonus, c=poi,cmap=cm.jet,vmin=0.,vmax=1.)
    #matplotlib.pyplot.scatter( salary, total_payments,c=poi,cmap=cm.jet,vmin=0.,vmax=1. )
    #matpotlib.pyplot.scatter( bonus, from_poi,c=poi,cmap=cm.jet,vmin=0.,vmax=1.)
    #matplotlib.pyplot.scatter( bonus, total_payments,c=poi,cmap=cm.jet,vmin=0.,vmax=1. )
    #matplotlib.pyplot.scatter( salary, from_poi,c=poi,cmap=cm.jet,vmin=0.,vmax=1. )
    #matplotlib.pyplot.scatter( bonus, to_poi,c=poi,cmap=cm.jet,vmin=0.,vmax=1. )
    

matplotlib.pyplot.xlabel("salary")
#matplotlib.pyplot.xlabel("bonus")

matplotlib.pyplot.ylabel("bonus")
#matplotlib.pyplot.ylabel("total_payments")
#matplotlib.pyplot.ylabel("from_poi_to_this_person")
#matplotlib.pyplot.ylabel("to_poi_from_this_person")

matplotlib.pyplot.show()
'''

### Task 3: Create new feature(s)
#from  featureSelectFinalProject import featureSelectFinal

### Store to my_dataset for easy export below.

### Create new featur-> ratio of bonus to salary:
for employee, features in data_dict.iteritems():
    if features['bonus'] == "NaN" or features['salary'] == "NaN" or (features['salary']==0) or features['bonus'] == 0 :
        features['bonus_salary_ratio'] = 0
    else:
        features['bonus_salary_ratio'] = float(features['bonus']) / float(features['salary'])

### Create new feature -> fraction of messages to a POI
for employee, features in data_dict.iteritems():
    if features['from_this_person_to_poi'] == "NaN" or features['from_messages'] == "NaN"  or features['from_this_person_to_poi'] == 0 or features['from_messages']==0 :
        features['fraction_to_poi'] = 0
    else:
        features['fraction_to_poi'] = float(features['from_this_person_to_poi'])/ float(features['from_messages'])

### Create  new feature -> fraction of messages from a POI                
for employee, features in data_dict.iteritems():
    if features['from_poi_to_this_person'] == "NaN" or features['to_messages'] == "NaN"  or features['from_poi_to_this_person'] == 0 or features['to_messages']==0 :
        features['fraction_from_poi'] = 0
    else:
        #features['bonus_salary_ratio'] = float(features['bonus']) / float(features['salary'])
        features['fraction_from_poi'] = float(features['from_poi_to_this_person'])/ float(features['to_messages'])

my_dataset = data_dict

'''
#print the new features:
for employee, features in my_dataset.iteritems():
    print "Employee:  ",employee, "  from POI ", features['fraction_from_poi'], "  to POI  ", features['fraction_to_poi']
'''

### Extract features and labels from dataset for local testing
'''
features_list = ['poi', 'salary', 'bonus', 'deferral_payments', 'total_payments', 'loan_advances', 
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 
                 'exercised_stock_options', 'long_term_incentive', 'restricted_stock', 'director_fees',
                 'shared_receipt_with_poi', 'fraction_to_poi', 'fraction_from_poi','bonus_salary_ratio']

features_list = ['poi', 'salary', 'bonus', 'deferral_payments', 'total_payments', 'loan_advances', 
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 
                 'exercised_stock_options', 'long_term_incentive', 'restricted_stock', 'director_fees',
                 'shared_receipt_with_poi', 'fraction_to_poi', 'fraction_from_poi']
'''
features_list = ['poi', 'salary', 'bonus', 'deferral_payments', 'total_payments', 'loan_advances', 
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 
                 'exercised_stock_options', 'long_term_incentive', 'restricted_stock', 'director_fees',
                 'shared_receipt_with_poi', 'fraction_to_poi', 'bonus_salary_ratio']


print "FEATURES LIST :  ", features_list
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### For initial testing with Classifier, I utilized the train_test_split algorithm:
#features_train,  features_test, labels_train,labels_test = cross_validation.train_test_split(features, labels, random_state=42, test_size=0.3)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression

### Create instances of each Classifier:
scaler = MinMaxScaler()
standscaler=StandardScaler()
select = SelectKBest()
dtc = DecisionTreeClassifier(random_state=42)
kmeans= KMeans(n_clusters=3, random_state = 42)
knc = KNeighborsClassifier(n_neighbors=3, algorithm='auto')
gnb=GaussianNB()
lin = LinearRegression()
svc = SVC(C=1000.0, kernel='rbf')
#svc = SVC(C=10.0, kernel='linear')
rfc = RandomForestClassifier(max_depth = 5,max_features = 'sqrt',n_estimators = 10, random_state = 42)

'''

### Use Naive Bayes Classifier

clf_nb = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy= clf.score(features_train, labels_train)
print "NB Gaussian accuracy: ",accuracy
print "precision score: ", precision_score(labels_test, pred),"  recall score: ", recall_score(labels_test, pred)

### Use Support Vector Machines Classifier

clf_svm = SVC(C=1000.0, kernel='rbf')
clf.fit(features_train, labels_train)
ped = clf.predict(features_test)
accuracy= clf.score(features_train, labels_train)
print "SVM accuracy: ",accuracy
print "precision score: ", precision_score(labels_test, pred),"  recall score: ", recall_score(labels_test, pred)

### Use DecisionTree Classifier

clf_dt= DecisionTreeClassifier(random_state=42)
clf_dt.fit(features_train, labels_train)
pred = clf_dt.predict(features_test)
accuracy= clf_dt.score(features_train, labels_train)
print "decision tree accuracy: ",accuracy
print "precision score: ", precision_score(labels_test, pred),"  recall score: ", recall_score(labels_test, pred))

### Use K Means Clusterign Classifier

clf_km= KMeans(n_clusters=3, random_state = 0)
clf_km.fit(data)
clf_km.labels_
#clf_km.fit(features_train, labels_train)
pred = clf_km.fit_predict(features_test)
clf.cluster_centers_
accuracy= clf.score(features_train, labels_train)
print "cluster accuracy: ",accuracy
print "precision score: ", precision_score(labels_test, pred),"  recall score: ", recall_score(labels_test, pred)

### Use K Neighbours Classifier

clf_knc= KNeighborsClassifier(n_neighbors=3, algorithm='auto')
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy= clf.score(features_train, labels_train)
print "k-nearest neighbors accuracy score: ", accuracy_score(labels_test,pred)
print "precision score: ", precision_score(labels_test, pred),"  recall score: ", recall_score(labels_test, pred)

# Use Liner Regression Classifier

clf_LR= linear_model.LinearRegression()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy= clf.score(features_train, labels_train)
print "linear regression accuracy: ", accuracy
print "precision score: ", precision_score(labels_test, pred),"  recall score: ", recall_score(labels_test, pred)

### Try using the test_classifier with the resulting model clf

#from tester import test_classifier 
#test_classifier(clf, my_dataset, features_list)
'''

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### Example starting point. Try investigating other evaluation techniques!

### Pipeline Steps

steps=[
       ('min_max_scaler',scaler),
       #('scaler',StandardScaler()),
       ('f_select',select),
       ('Dtree',dtc)
       #('svc',svc)
       #('knc',knc)
       #('rfc',rfc)
       #('gausian_nb',gnb)
       #('lin',lin)
       #('kmean',kmeans)
       ]
       
pipeline=Pipeline(steps)

#print sorted(pipeline.get_params().keys())

### parameters to utilize in pipeline in gridsearch
parameters = {'f_select__k':range(1,17),
#parameters = {'f_select__k': [16], #Number of top features to select
#parameters = {        
              'Dtree__criterion': ['gini','entropy'],
              #'Dtree__min_samples_split':[40, 10, 20],
              #'Dtree__min_samples_split':[5, 10, 20],
              #'Dtree__max_depth':[3,10,15,20,25,30],
              #'Dtree__max_leaf_nodes':[5,10,30]
              'Dtree__max_features': [0.4]
              #'knc__n_neighbors':[1,2,3,4,5],
              #'knc__leaf_size':[1, 10, 30, 60]
              #'knc__algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']
              }
'''
### Some available parameters:
['f_select', 'f_select__k', 'f_select__score_func', 'knc', 'knc__algorithm', 'knc__leaf_size', 'knc__metric', 'knc__metric_params', 'knc__n_jobs',
'knc__n_neighbors', 'knc__p', 'knc__weights', 'memory', 'min_max_scaler', 'min_max_scaler__copy', 'min_max_scaler__feature_range', 'steps']
'''

'''
Tried Logistic Regression classifier separatley from the pipeline:

clf = Pipeline(steps=[ ('scaler', StandardScaler()), ('classifier', LogisticRegression(tol = 0.001, C = 10**-10, penalty = 'l2', random_state = 42))])
#clf = Pipeline(steps=[ ('scaler', StandardScaler()), ('classifier', LogisticRegression(tol = 0.001, C = 10**-8, penalty = 'l2', random_state = 42))])
clf = RandomForestClassifier(max_depth = 5,max_features = 'sqrt',n_estimators = 10, random_state = 42)
#as an example. then you need to simply fit the data in it
'''

from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
#features_train, features_test, labels_train, labels_test = \
#train_test_split(features, labels, test_size=0.3, random_state=42)
#from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

#param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
#param_grid = parameters


sss = StratifiedShuffleSplit(labels, n_iter =100, test_size=0.3, random_state = 42)
grid_search = GridSearchCV(pipeline, param_grid=parameters, cv = sss)

### Tried different parameter for StratifiedShuffleSplit and GridSearcgCV
#sss= StratifiedShuffleSplit(n_iter = 20,test_size=0.5, random_state = 5)
#grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv = sss, verbose=10, scoring='f1')
#grid_search = GridSearchCV(pipeline, param_grid=parameters, cv = sss, error_score = 0, scoring='f1')

#print "Grid Search:   ", grid_search
#print(grid_search.best_estimator_.steps)
#print "\n", "Best parameters are: ", grid_search.best_params_, "\n"

grid_search.fit(features, labels)
clf = grid_search.best_estimator_

### Use test_classifier.py to test the best model found
from tester import test_classifier

# Use test_classifier to evaluate the model selected by GridSearchCV
print "\n", "Tester Classification report -  StratifiedShuffleSplit:" 
test_classifier(clf, data_dict, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#print features_list
dump_classifier_and_data(clf, my_dataset, features_list)
