import pandas as pd #data frame
import numpy as np #linear algebra
import matplotlib.pyplot as plt #to plot
import operator
import pprint
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score



nucs_all = {'Carbonyl carbon': 'co', 'Methyl carbon':'ch','Hydrogen':'h'}

for name, nucs in nucs_all.items():
	df_test = pd.read_csv('{}all_test.csv'.format(nucs))
	df_test_predicted = pd.read_csv('{}all_test_predicted.csv'.format(nucs))

	pred = pd.DataFrame(columns = {'test','predicted'})

	for i in range(int(df_test.shape[1]/4)):
		test = df_test['Fermi_contact{}'.format(i+1)]
		predicted = df_test_predicted['Fermi_contact{}'.format(i+1)]
		tp = pd.DataFrame({'test':test,'predicted':predicted})
		pred = pd.concat([pred,tp],axis = 0)

	print name," Explained variance score:",explained_variance_score(pred['test'], pred['predicted'])
	print name," Mean absolute error     :",mean_absolute_error(pred['test'], pred['predicted'])
	print name," Mean squared error      :",mean_squared_error(pred['test'], pred['predicted'])
	print name," Median absolute error   :",median_absolute_error(pred['test'], pred['predicted'])
	print name," R2 error                :",r2_score(pred['test'], pred['predicted'])

	fig,ax = plt.subplots();ax.scatter(pred['test'], pred['predicted']);ax.plot([pred['test'].min(), pred['test'].max()], [pred['test'].min(), pred['test'].max()], 'k--', lw=4);ax.set_xlabel('Measured');ax.set_ylabel('Predicted');plt.show()	
	print pred.describe()
