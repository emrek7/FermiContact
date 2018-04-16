import pandas as pd #data frame
import numpy as np #linear algebra
import matplotlib.pyplot as plt #to plot
import operator
import pprint
import warnings
warnings.filterwarnings("ignore")

from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold


def prepare_data(nucs, num_n,dataset):
	df = pd.read_csv('{}all_{}.csv'.format(nucs,dataset))
	# Data contains r_beta_phi coordinates for 6 closest molecules to the polarizing agent 
	# and fermi contacts of electron spin in the nuclear spin. 
	# Data is obtained from 1ns simulations containing 5000 snapshots. 
	# If the nucs is h, acetone has 6 protons resulting 36 spherical coordinates set and fermi contacts.
	# nucs is ch, acetone has 2 methylene carbons therefore 12 coordinates and 2 fermi_contact sets.
	# nucs is co, acetone has 1 carbonyl carbon therefore 6 sets of coordinates and fermi contacts. 

	# Split the data and target. Remove the Fermi contact values 
	y = df['Fermi_contact{}'.format(num_n)]
	fcsToDrop = ['Fermi_contact{}'.format(i+1) for i in range(int(df.shape[1]/4))]
	X = df.drop(fcsToDrop,axis = 1)

	#Scale the data
	X = StandardScaler().fit(X).transform(X)
	return X,y

def main():
	X, y = prepare_data(nucs,1,'train')

	#Make polyfeatures?
	Xp =StandardScaler().fit(PolynomialFeatures(2).fit_transform(X)).transform(PolynomialFeatures(2).fit_transform(X))

	myScoring = 'r2' #possible arguments 'explained_variance','neg_mean_absolute_error','neg_mean_squared_error','neg_mean_squared_log_error','neg_median_absolute_error','r2'
	myCv = KFold(n_splits=5, random_state=7)
	
	estimators = {
		"Ridge Regression"         : linear_model.Ridge(),         #10ms
		"Bayesien Ridge"           : linear_model.BayesianRidge(), #20ms     
		"Elastic Net"              : linear_model.ElasticNet(),    #38ms 
		"Stochastic Grad Descent"  : linear_model.SGDRegressor(),  #12ms
		"GradientBoostingRegr"     : GradientBoostingRegressor(),  #459ms
		"MultiLayerPerceptron"     : MLPRegressor(),               #100ms
		"KNeighborsRegressor"      : KNeighborsRegressor(),        #1400ms
		"Decision Tree Regressor"  : DecisionTreeRegressor(),      #194ms
		"RandomForestRegressor"    : RandomForestRegressor(),      #768ms
		"ExtraTreesRegressor"      : ExtraTreesRegressor(),        #110ms
		"Support Vector Regressor" : svm.SVR()                     #4667ms
	}

	#Brute force for hyper-parameter search
	paramsSearch = {
	    "Ridge Regression"         : {'alpha' : np.logspace(-3,3,20)},
	    "Bayesien Ridge"           : {'alpha_1' : np.logspace(-3,0,4),
	                                  'alpha_2' : np.logspace(-3,0,4),
	                                  'lambda_1' : np.logspace(-9,-8,2),
	                                  'lambda_2' : np.logspace(-9,-8,2)},
	    "Elastic Net"              : {'l1_ratio' : np.linspace(0.1,0.5,5), 
	                                  'alpha' : np.logspace(-6,0,7)},
	    "Stochastic Grad Descent"  : {'loss' : ['squared_loss'],          # 'squared_loss', 'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'
	                                  'penalty' : ['l2','elasticnet'],    # 'l1', 'l2', 'elasticnet'
	                                  'alpha' : np.logspace(-6,0,7),
	                                  'l1_ratio' : np.linspace(0,1,11)},
	    "GradientBoostingRegr"     : {'loss': ['ls'],
	                                  'n_estimators':np.arange(40,100,10),
	                                  'max_depth' : np.arange(4,10)},
	    "MultiLayerPerceptron"     : {'hidden_layer_sizes' : [10,15],
	                                  'random_state' : [5,7,9],
	                                  'max_iter' : [5,10,15],
	                                  'warm_start' : [True,False]},
	    "KNeighborsRegressor"      : {'n_neighbors': np.arange(2, 20, 2),
	                                  'metric' : ["cityblock"],          # 'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'
	                                  'weights' : ['distance'],          # 'uniform', 'distance'
	                                  'algorithm' : ['ball_tree']},      # 'auto', 'ball_tree', 'kd_tree', 'brute'
	    "Decision Tree Regressor"  : {'max_features':['auto','sqrt'],
	                                  'max_depth':np.arange(10,100,10)},
	    "RandomForestRegressor"    : {'n_estimators' : np.arange(14,18,2),
	                                  'max_features' : ['auto'],
	                                  'max_depth': np.arange(7,10,1),
	                                  'min_samples_leaf' : [1],
	                                  'min_samples_split' : [5],
	                                  'bootstrap' : [True]},
	    "ExtraTreesRegressor"      : {'n_estimators' : np.arange(14,18,2),
	                                  'max_features' : ['auto'],
	                                  'max_depth': np.arange(7,10,1),
	                                  'min_samples_leaf' : [1],
	                                  'min_samples_split' : [5],
	                                  'bootstrap' : [True]},
	    "Support Vector Regressor" : {'kernel':['rbf'],                 # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
	                                  'C' : np.logspace(-2,0,3),
	                                  'epsilon':[0.1]}
	}

	best_params = {}
	best_scores = {}
	
	for name, estimator in estimators.items():
		# print name
		# 1 degree
		gr = GridSearchCV(estimator,paramsSearch[name], cv = myCv, scoring = myScoring, verbose = False).fit(X,y)
		best_scores[name]=gr.best_score_
		best_params[name]=gr.best_params_
		print "{} best score: {}, best parameters are {} ".format(name,gr.best_score_,gr.best_params_)
		# polynomial 2 degrees
		# gr2 = GridSearchCV(estimator,paramsSearch[name], cv = myCv, scoring = myScoring, verbose = False).fit(Xp,y)
		# print "{} best score: {}, best parameters are {} ".format(name,gr2.best_score,gr2.best_params_)

	pprint.pprint(sorted(best_scores.items(), key=operator.itemgetter(1),reverse = True)[0:3])
	best_estimator = sorted(best_scores.items(), key=operator.itemgetter(1),reverse = True)[0][0]
	best_estimator = estimators[best_estimator].set_params(**best_params[best_estimator])

	score   = cross_val_score(best_estimator,X,y,cv = myCv,scoring = myScoring).mean()
	# scorep  = cross_val_score(best_estimator,Xp,y,cv = myCv,scoring = myScoring).mean()
	# scoreb  = cross_val_score(BaggingRegressor(best_estimator),X,y,cv = myCv,scoring = myScoring).mean()
	# scorebp = cross_val_score(BaggingRegressor(best_estimator),Xp,y,cv = myCv,scoring = myScoring).mean()

	#plot?
	fetchInfo = 0
	if fetchInfo:
		predicted = best_estimator.fit(X,y).predict(X)
		pred = pd.DataFrame({'predicted' : predicted, 'measured': y})
		fig,ax = plt.subplots();ax.scatter(y, predicted);ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4);ax.set_xlabel('Measured');ax.set_ylabel('Predicted');plt.show()	
		print pred.describe()


	# # # Found the best estimator?
	best_estimator.fit(X,y)
	Y_test = pd.DataFrame()
	for i in range(int(X.shape[1]/3)):
		X_test,_ = prepare_data(nucs,i+1,'test')
		predicted = best_estimator.predict(X_test)
		cols = ['r{}'.format(i+1),'beta{}'.format(i+1),'phi{}'.format(i+1),'Fermi_contact{}'.format(i+1)]
		predicted_df = pd.DataFrame([X_test[:,i*3],X_test[:,i*3+1],X_test[:,i*3+2],predicted]).transpose()
		predicted_df.columns = cols
		Y_test = pd.concat([Y_test, predicted_df],axis = 1)

	Y_test.to_csv('{}all_test_predicted.csv'.format(nucs),index=False)

if __name__ == "__main__":
	nucs = 'ch'
	main()
	nucs = 'co'
	main()
	nucs = 'h'
	main()
