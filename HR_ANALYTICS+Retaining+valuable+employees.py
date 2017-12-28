
# coding: utf-8

# In[143]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import accuracy_score, log_loss

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import normalize

from sklearn import tree
from sklearn import linear_model
from sklearn import svm

from sklearn.neural_network import MLPClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

import sys,os
import seaborn as sns

get_ipython().magic(u'matplotlib inline')


# In[4]:


training_data = pd.read_csv("C:\Users\ANUHYA\Desktop\CERT_PROJECT_EDUREKA\HR_Data.csv")
training_data.head()


# In[6]:


training_data.shape


# In[7]:


training_data.rename(columns={'sales':'department'}, inplace=True)


# In[8]:


training_data.head()


# In[9]:


value=[1]
training_data[training_data.satisfaction_level.isin(value)]


# In[10]:


print (training_data['left'] == 1).sum()


# In[12]:


training_data.describe()


# In[13]:


training_data.info()


# In[14]:


training_data = training_data.rename(columns={'satisfaction_level': 'satisfaction', 
                        'last_evaluation': 'evaluation',
                        'number_project': 'projectCount',
                        'average_montly_hours': 'avgMonthlyHrs',
                        'time_spend_company': 'yearsAtCompany',
                        'Work_accident': 'workAccident',
                        'promotion_last_5years': 'promotion',
                        })


# In[15]:


training_data.describe()


# In[16]:


my_tab = pd.crosstab(index = training_data["department"],  # Make a crosstab
                              columns="count")      # Name the count column

my_tab.plot.bar()


# In[17]:


my_tab = pd.crosstab(index = training_data["salary"],  # Make a crosstab
                              columns="count")      # Name the count column

my_tab.plot.bar()


# In[18]:


t = sns.heatmap(training_data[["left","satisfaction","evaluation","projectCount","avgMonthlyHrs","yearsAtCompany","workAccident","promotion"]].corr(),annot=True, fmt = ".3f", cmap = "coolwarm")


# In[20]:


sns.pairplot(training_data,hue='left')


# In[21]:


not_left_pop_satisfaction = training_data['satisfaction'][training_data['left'] == 0].mean()
left_satisfaction = training_data[training_data['left']==1]['satisfaction'].mean()

print( 'The mean satisfaction for employees that have not left is: ' + str(not_left_pop_satisfaction))
print( 'The mean satisfaction for employees that have left is: ' + str(left_satisfaction) )


# In[26]:


import scipy.stats as stats
stats.ttest_1samp(a=training_data[training_data['left']==1]['satisfaction'],popmean = not_left_pop_satisfaction) #Sample of Employee satisfaction who left # Employee Who Have not left satisfaction mean


# In[30]:


dof = len(training_data[training_data['left']==1])

RightQ = stats.t.ppf(0.975,dof)
LeftQ = stats.t.ppf(0.025,dof)

print('The right quartile range of this t-distribution is: ' + str(RightQ))
print('The left quartile range of this t-distribution is: ' + str(LeftQ))


# In[31]:


from sklearn import preprocessing
def encode_features(df_train):
    features = ['department', 'salary']
    data_combined = pd.concat([df_train[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(data_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
    return df_train
    
data_train = encode_features(training_data)
data_train.head()


# In[32]:


data_train.department.unique()


# In[33]:


data_train.salary.unique()


# In[38]:


t = sns.heatmap(data_train[["left","satisfaction","evaluation","projectCount","avgMonthlyHrs","yearsAtCompany","workAccident","promotion","department","salary"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")


# In[37]:


sns.pairplot(data_train,hue='left')


# In[47]:


f, axes = plt.subplots(ncols=3, figsize=(15, 6))

# Graph Employee Satisfaction
sns.distplot(training_data.satisfaction,color="m", ax=axes[0]).set_title('Employee Satisfaction Distribution')
axes[0].set_ylabel('Employee Count')

sns.distplot(training_data.evaluation, color="r", ax=axes[1]).set_title('Employee Evaluation Distribution')
axes[1].set_ylabel('Employee Count')

# Graph Employee Average Monthly Hours
sns.distplot(training_data.avgMonthlyHrs, color="b", ax=axes[2]).set_title('Employee Average Monthly Hours Distribution')
axes[2].set_ylabel('Employee Count')


# In[53]:


training_data.head()


# In[54]:


data_train.head()


# In[55]:


t_data = pd.read_csv("C:\Users\ANUHYA\Desktop\CERT_PROJECT_EDUREKA\HR_Data.csv")
t_data.head()


# In[58]:


t_data = t_data.rename(columns={'satisfaction_level': 'satisfaction', 
                        'last_evaluation': 'evaluation',
                        'number_project': 'projectCount',
                        'average_montly_hours': 'avgMonthlyHrs',
                        'time_spend_company': 'yearsAtCompany',
                        'Work_accident': 'workAccident',
                        'promotion_last_5years':'promotion',
                        'sales':'department'
                        })


# In[77]:


f, ax = plt.subplots(figsize=(15, 5))
sns.countplot(y="department", hue='left', data=training_data).set_title('Employee Department Turnover Distribution');


# In[78]:


f, ax = plt.subplots(figsize=(15, 4))
sns.countplot(y="salary", hue='left', data=training_data).set_title('Employee Salary Turnover Distribution');


# In[80]:


f, ax = plt.subplots(figsize=(15, 4))
sns.countplot(y="projectCount", hue='left', data=training_data).set_title('Employee Project Count Turnover Distribution');


# In[81]:


f, ax = plt.subplots(figsize=(15, 4))
sns.countplot(y="yearsAtCompany", hue='left', data=data_train).set_title('Employee yearsAtCompany Turnover Distribution');


# In[82]:


f, ax = plt.subplots(figsize=(15, 4))
sns.countplot(y="avgMonthlyHrs", hue='left', data=data_train).set_title('Employee avgMonthlyHrs Turnover Distribution')


# In[84]:


sns.set(style="darkgrid")

# Subset the dataset
left = training_data.query("left == 1")
non_left = training_data.query("left == 0")

# Set up the figure
f, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect("equal")

# Draw the two density plots
ax = sns.kdeplot(left.satisfaction, left.evaluation,
                 cmap="Reds", shade=True, shade_lowest=False)
ax = sns.kdeplot(non_left.satisfaction, non_left.evaluation,
                 cmap="Blues", shade=True, shade_lowest=False)

# Add labels to the plot
red = sns.color_palette("Reds")[-2]
blue = sns.color_palette("Blues")[-2]
ax.text(2.5, 8.2, "non_left", size=16, color=blue)
ax.text(3.8, 4.5, "left", size=16, color=red)


# In[88]:


sns.set(style="darkgrid")

# Subset the dataset
left = training_data.query("left == 1")
non_left = training_data.query("left == 0")

# Set up the figure
f, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect("equal")

# Draw the two density plots
ax = sns.kdeplot(left.satisfaction, left.projectCount,
                 cmap="Reds", shade=True, shade_lowest=False)
ax = sns.kdeplot(non_left.satisfaction, non_left.projectCount,
                 cmap="Blues", shade=True, shade_lowest=False)

# Add labels to the plot
red = sns.color_palette("Reds")[-2]
blue = sns.color_palette("Blues")[-2]
ax.text(2.5, 8.2, "non_left", size=10, color=blue)
ax.text(3.8, 4.5, "left", size=10, color=red)


# In[90]:


sns.set(style="darkgrid")

# Subset the dataset 
left = training_data.query("left == 1")
non_left = training_data.query("left == 0")

# Set up the figure
f, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect("equal")

# Draw the two density plots
ax = sns.kdeplot(left.satisfaction, left.yearsAtCompany,
                 cmap="Reds", shade=True, shade_lowest=False)
ax = sns.kdeplot(non_left.satisfaction, non_left.yearsAtCompany,
                 cmap="Blues", shade=True, shade_lowest=False)

# Add labels to the plot
red = sns.color_palette("Reds")[-2]
blue = sns.color_palette("Blues")[-2]
ax.text(2.5, 8.2, "non_left", size=10, color=blue)
ax.text(3.8, 4.5, "left", size=10, color=red)


# In[94]:


sns.set(style="darkgrid")

# Subset the dataset
left = training_data.query("left == 1")
non_left = training_data.query("left == 0")

# Set up the figure
f, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect("equal")

# Draw the two density plots
ax = sns.kdeplot(left.satisfaction, left.salary,
                 cmap="Reds", shade=True, shade_lowest=False)
ax = sns.kdeplot(non_left.satisfaction, non_left.salary,
                 cmap="Blues", shade=True, shade_lowest=False)

# Add labels to the plot
red = sns.color_palette("Reds")[-2]
blue = sns.color_palette("Blues")[-2]
ax.text(2.5, 8.2, "non_left", size=10, color=blue)
ax.text(3.8, 4.5, "left", size=10, color=red)
ax.text(3.6, 4.0, "salary 0.0-HIGH", size=10, color=red)
ax.text(3.2, 4.2, "salary 1.0-LOW", size=10, color=red)
ax.text(3.4, 4.8, "salary 2.0-MEDIUM", size=10, color=red)


# In[95]:


sns.set(style="darkgrid")

# Subset the iris dataset by species
left = training_data.query("left == 1")
non_left = training_data.query("left == 0")

# Set up the figure
f, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect("equal")

# Draw the two density plots
ax = sns.kdeplot(left.department, left.salary,
                 cmap="Reds", shade=True, shade_lowest=False)
ax = sns.kdeplot(non_left.department, non_left.salary,
                 cmap="Blues", shade=True, shade_lowest=False)

# Add labels to the plot
red = sns.color_palette("Reds")[-2]
blue = sns.color_palette("Blues")[-2]
ax.text(2.5, 8.2, "non_left", size=10, color=blue)
ax.text(3.8, 4.5, "left", size=10, color=red)
ax.text(3.6, 4.0, "salary 0.0-HIGH", size=10, color=red)
ax.text(3.2, 4.2, "salary 1.0-LOW", size=10, color=red)
ax.text(3.4, 4.8, "salary 2.0-MEDIUM", size=10, color=red)


# In[109]:


fig = plt.figure(figsize=(15,8))
ax=sns.kdeplot(training_data.loc[(training_data['left'] == 0),'avgMonthlyHrs'] , color='b',shade=True, label='non-left')
ax=sns.kdeplot(training_data.loc[(training_data['left'] == 1),'avgMonthlyHrs'] , color='r',shade=True, label='left')
ax.set(xlabel='Employee Average Monthly Hours', ylabel='Frequency')
plt.title('Employee AverageMonthly Hours Distribution - left V.S. Non-left')
plt.yticks(ax.get_yticks(), ax.get_yticks() * 2)


# In[113]:


fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(training_data.loc[(training_data['left'] == 0),'satisfaction'] , color='b',shade=True, label='non-left')
ax=sns.kdeplot(training_data.loc[(training_data['left'] == 1),'satisfaction'] , color='r',shade=True, label='left')
ax.set(xlabel='Employee Satisfaction', ylabel='Frequency')
plt.title('Employee Satisfaction - left V.S. Non-left')
plt.yticks(ax.get_yticks(), ax.get_yticks() * 2)


# In[114]:


sns.boxplot(x="projectCount", y="avgMonthlyHrs", hue="left", data=training_data)


# In[117]:


sns.boxplot(x="projectCount", y="satisfaction", hue="left", data=training_data)


# In[121]:


X = training_data.drop(['left'], axis=1)
Y = training_data.left
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[122]:


from sklearn.tree import DecisionTreeClassifier
classifiers = {}
dtree = DecisionTreeClassifier()
dtree.fit(X_train, Y_train)
Y_pred =  dtree.predict(X_test)

print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
classifiers["Decision Tree"]=dtree


# In[123]:


# Artifical Neural Network
clf = MLPClassifier()
clf.set_params(hidden_layer_sizes =(100,100), max_iter = 1000,alpha = 0.01, momentum = 0.7)
nn_clf = clf.fit(X_train,Y_train)
nn_predict = nn_clf.predict(X_test)

nn_acc = accuracy_score(Y_test,nn_predict)
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')

print("Artificial Nueral Network:")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["NeuralNetwork"]=clf


# In[124]:


#Deep Neural Network
clf = MLPClassifier()
clf.set_params(hidden_layer_sizes =(100,100,100,100), max_iter = 100,alpha = 0.3, momentum = 0.7,activation = "relu")
nn_clf = clf.fit(X_train,Y_train)
nn_predict = nn_clf.predict(X_test)
nn_acc = accuracy_score(Y_test,nn_predict)
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("Deep Neural Network:")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["DeepNeuralNetwork"]=clf


# In[126]:


#Multinomial Naive Bayes
clf = MultinomialNB()
clf.set_params(alpha = 0.1)
nb_clf = clf.fit(X_train,Y_train)
nb_predict = nb_clf.predict(X_test)
nb_acc = accuracy_score(Y_test,nb_predict)
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("Multinomial Naive Bayes:")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["MultinomialNaiveBayes"]=clf


# In[129]:


#Support Vector Machine
clf = svm.SVC()
clf.set_params(C = 100, kernel = "rbf")
svm_clf = clf.fit(X_train,Y_train)
svm_predict = svm_clf.predict(X_test)
svm_acc = accuracy_score(Y_test,svm_predict)
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("Support Vector Machines:")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["SupportVectorMachine"]=clf


# In[130]:


#Logistic Regression
clf = LogisticRegression()
clf.set_params(C = 10, max_iter = 10)
lr_clf = clf.fit(X_train,Y_train)
lr_predict = lr_clf.predict(X_test)
lr_acc = accuracy_score(Y_test,lr_predict)
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("Logistic Regression:")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["LogisticRegression"]=clf


# In[131]:


#k-NN Classifier
clf = KNeighborsClassifier()
clf.set_params(n_neighbors= 5,leaf_size = 30)
knn_clf = clf.fit(X_train,Y_train)
knn_predict = knn_clf.predict(X_test)
knn_acc = accuracy_score(Y_test,knn_predict)
param =  knn_clf.get_params()
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("k-NN :")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["KNN"]=clf


# In[132]:


#Random Forest Classifier
clf = RandomForestClassifier()
clf.set_params(n_estimators = 500, max_depth = 100)
rf_clf = clf.fit(X_train,Y_train)
rf_predict = rf_clf.predict(X_test)
rf_acc = accuracy_score(Y_test,rf_predict)
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("Random Forest Classifier:")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["RandomForest"]=clf


# In[133]:


#Gradient Boosting Classifier
clf = GradientBoostingClassifier()
clf.set_params(n_estimators = 100,learning_rate = 0.25)
gb_clf = clf.fit(X_train,Y_train)
gb_predict = gb_clf.predict(X_test)
gb_acc = accuracy_score(Y_test,gb_predict)
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("GradientBoostingClassifier:")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["GradientBoostingClassifier"]=clf


# In[134]:


#Perceptron
clf = linear_model.Perceptron()
pt_clf = clf.fit(X_train,Y_train)
pt_predict = pt_clf.predict(X_test)
pt_acc = accuracy_score(Y_test,pt_predict)
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("Perceptron:")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["Perceptron"]=clf


# In[145]:


#Here we print the performance of the various classifiers
print ("accuracy","              ","F-score")
for clf in classifiers.values():
    accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
    f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
    for i in classifiers:
        if classifiers[i]== clf:
            print (i),
            break
    print ( " : ",accuracy.mean(), "  ",f_score.mean())


# In[149]:


#our best classifiers are:
#1. RandomForest (' : ', 0.99158124583188378, '  ', 0.99166450977859477)
#2. GradientBoostingClassifier (' : ', 0.97816360057657459, '  ', 0.97841360057657456)
#3. Decision Tree (' : ', 0.9758288770802388, '  ', 0.97524568286737257)
#4. SupportVectorMachine (' : ', 0.95782859027448386, '  ', 0.95782859027448386)
#5. KNN (' : ', 0.93216218934874262, '  ', 0.93216218934874262)


model = GradientBoostingClassifier()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

# Determine the false positive and true positive rates
fpr, tpr, _ = roc_curve(Y_test, model.predict_proba(X_test)[:,1])
 
# Calculate the AUC
roc_auc = auc(fpr, tpr)
print ('ROC AUC: %0.2f' % roc_auc)
 
# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


# In[150]:


#Next we proceed to apply "Feature Scaling" to see if the performance of our various classifiers improves
#Feature scaling aims to bring the values of our numerical features between 0 and 1
#This is mainly done because large numerical values may skew our data and make the classifier weight it more

#this technique is known to improve the performance of classifiers using gradient descent such as neural nets,perceptron,etc

XX = training_data.drop(['left'], axis=1)
YY = training_data.left
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
XX[XX.columns] = scaler.fit_transform(XX[XX.columns])

XX.head()



# In[151]:


X_train, X_test, Y_train, Y_test = train_test_split(XX, YY, test_size = 0.20, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[152]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifiers = {}
dtree = DecisionTreeClassifier()
dtree.fit(X_train, Y_train)
Y_pred =  dtree.predict(X_test)
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
classifiers["Decision Tree"]=dtree


# In[153]:


# Artifical Neural Network
clf = MLPClassifier()
clf.set_params(hidden_layer_sizes =(100,100), max_iter = 1000,alpha = 0.01, momentum = 0.7)
nn_clf = clf.fit(X_train,Y_train)
nn_predict = nn_clf.predict(X_test)
nn_acc = accuracy_score(Y_test,nn_predict)
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("Artificial Nueral Network:")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["NeuralNetwork"]=clf


# In[154]:


#Deep Neural Network
clf = MLPClassifier()
clf.set_params(hidden_layer_sizes =(100,100,100,100), max_iter = 100,alpha = 0.3, momentum = 0.7,activation = "relu")
nn_clf = clf.fit(X_train,Y_train)
nn_predict = nn_clf.predict(X_test)
nn_acc = accuracy_score(Y_test,nn_predict)
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("Deep Neural Network:")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["DeepNeuralNetwork"]=clf


# In[155]:


#Support Vector Machine
clf = svm.SVC()
clf.set_params(C = 100, kernel = "rbf")
svm_clf = clf.fit(X_train,Y_train)
svm_predict = svm_clf.predict(X_test)
svm_acc = accuracy_score(Y_test,svm_predict)
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("Support Vector Machines:")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["SupportVectorMachine"]=clf


# In[156]:


#Multinomial Naive Bayes
clf = MultinomialNB()
clf.set_params(alpha = 0.1)
nb_clf = clf.fit(X_train,Y_train)
nb_predict = nb_clf.predict(X_test)
nb_acc = accuracy_score(Y_test,nb_predict)
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("Multinomial Naive Bayes:")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["MultinomialNaiveBayes"]=clf


# In[157]:


#Logistic Regression
clf = LogisticRegression()
clf.set_params(C = 10, max_iter = 10)
lr_clf = clf.fit(X_train,Y_train)
lr_predict = lr_clf.predict(X_test)
lr_acc = accuracy_score(Y_test,lr_predict)
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("Logistic Regression:")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["LogisticRegression"]=clf


# In[158]:


#k-NN Classifier
clf = KNeighborsClassifier()
clf.set_params(n_neighbors= 5,leaf_size = 30)
knn_clf = clf.fit(X_train,Y_train)
knn_predict = knn_clf.predict(X_test)
knn_acc = accuracy_score(Y_test,knn_predict)
param =  knn_clf.get_params()
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("k-NN :")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["KNN"]=clf


# In[159]:


#Random Forest Classifier
clf = RandomForestClassifier()
clf.set_params(n_estimators = 500, max_depth = 100)
rf_clf = clf.fit(X_train,Y_train)
rf_predict = rf_clf.predict(X_test)
rf_acc = accuracy_score(Y_test,rf_predict)
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("Random Forest Classifier:")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["RandomForest"]=clf


# In[160]:


#AdaBoost
clf = AdaBoostClassifier()
clf.set_params(n_estimators = 10, learning_rate = 0.5)
ada_clf = clf.fit(X_train,Y_train)
ada_predict = ada_clf.predict(X_test)
ada_acc = accuracy_score(Y_test,ada_predict)
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("AdaBoost:")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["AdaBoost"]=clf


# In[161]:


#Gradient Boosting Classifier
clf = GradientBoostingClassifier()
clf.set_params(n_estimators = 100,learning_rate = 0.25)
gb_clf = clf.fit(X_train,Y_train)
gb_predict = gb_clf.predict(X_test)
gb_acc = accuracy_score(Y_test,gb_predict)
accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
print("GradientBoostingClassifier:")
print (accuracy.mean(), " - ",f_score.mean())
classifiers["GradientBoostingClassifier"]=clf


# In[163]:


#Here we print the performance of the various classifiers
print ("accuracy","              ","F-score")
for clf in classifiers.values():
    accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')
    f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')
    for i in classifiers:
        if classifiers[i]== clf:
            print (i),
            break
    print ( " : ",accuracy.mean(), "  ",f_score.mean())


# In[164]:


#RandomForest (' : ', 0.99166450977859477, '  ', 0.99158124583188378)
#GradientBoostingClassifier (' : ', 0.97808019774087818, '  ', 0.97816360057657459)
#Decision Tree (' : ', 0.97649603061066959, '  ', 0.97616325260179582)
#NeuralNetwork (' : ', 0.96732977586558511, '  ', 0.96766227667519222)
#DeepNeuralNetwork (' : ', 0.96566338628244419, '  ', 0.96466345578480739)

model = GradientBoostingClassifier()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

# Determine the false positive and true positive rates
fpr, tpr, _ = roc_curve(Y_test, model.predict_proba(X_test)[:,1])
 
# Calculate the AUC
roc_auc = auc(fpr, tpr)
print ('ROC AUC: %0.2f' % roc_auc)
 
# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


# In[165]:


#ROC Curve for Random Forest Classifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
 
# shuffle and split training and test sets
#X_train, X_test, Y_train, Y_test = train_test_split(XX, YY, test_size = 0.20, random_state = 5)
X_train, X_test, Y_train, Y_test = train_test_split(XX, YY, test_size=.25)

forest = RandomForestClassifier()
#clf.set_params(n_estimators = 100, max_depth = 10, max_features = 3, criterion = 'gini')
#rf_clf = clf.fit(X_train,Y_train)
#rf_predict = rf_clf.predict(X_test)

forest.fit(X_train, Y_train)
 
# Determine the false positive and true positive rates
fpr, tpr, _ = roc_curve(Y_test, forest.predict_proba(X_test)[:,1])
 
# Calculate the AUC
roc_auc = auc(fpr, tpr)
print ('ROC AUC: %0.2f' % roc_auc)
 
# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


# In[166]:


model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

# Determine the false positive and true positive rates
fpr, tpr, _ = roc_curve(Y_test, model.predict_proba(X_test)[:,1])
 
# Calculate the AUC
roc_auc = auc(fpr, tpr)
print ('ROC AUC: %0.2f' % roc_auc)
 
# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

