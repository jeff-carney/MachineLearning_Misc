from sklearn.feature_selection import RFE
from try_subset import subset
from sklearn.svm import SVC
import numpy as np
import pandas as pd

# optimizes over permutations of parameter C and relevant features using recursive feature elimination, then fits optimal model based on F1 score


np.random.seed(5)		# consistent random seeds across files for subsetting

data_set = subset(12, 1)	# using optimal data_set found through optimization of random forest

train_set = data_set[0]		# grabbing training and test sets from output of subset function
test_set = data_set[1]

all_feats = []
for col in train_set.columns:
	if'avg_norm' in col:
		all_feats.append(col)		# adding the entire set of relevant variables to list all_feats


train_set['class'] = 0

ones = train_set[train_set['career_earnings'] >= 15000000]['id'].tolist()		# creating positive and negative classes

zeros = train_set[train_set['career_earnings'] < 15000000]['id'].tolist()		# creating positive and negative classes

train_set.loc[ones, 'class'] = 1		# creating positive and negative classes
train_set.loc[zeros, 'class'] = 0		# creating positive and negative classes

test_set['class'] = 0

ones = test_set[test_set['career_earnings'] >= 15000000]['id'].tolist()		# creating positive and negative classes

zeros = test_set[test_set['career_earnings'] < 15000000]['id'].tolist()		# creating positive and negative classes

test_set.loc[ones, 'class'] = 1		# creating positive and negative classes
test_set.loc[zeros, 'class'] = 0		# creating positive and negative classes

best_f1 = 0 # initializing best f1_score variable, used later to identify best model
best_c = 0 # initializing best value of c
best_features = 'a' # initializing best feature set
best_n = 0
best_pred = 0


def svm_feature_select(c, number_variables):		# performs recursive feature elimination for SVC using sklearn
	global best_f1, best_c, best_features, best_n, best_pred
	global train_set
	global test_set
	global all_feats


	mod_svc = SVC(C=c, kernel='linear')		# creating the model

	selector = RFE(mod_svc, number_variables)		# selecting top 10 variables

	selector = selector.fit(train_set.loc[:, all_feats], np.ravel(train_set.loc[:, ['class']]))	# fitting model

	rank = selector.ranking_	# grabbing rankings - 1 means variable was one of top 20
	rank = rank.tolist()		# changing from array to list for my convenience

	use_index = []		# initializing list to hold indices of most imp. variables

	for r in range(0, len(rank)):		# looping through ranking to grab index of only top 20 variables
		if rank[r] == 1:		# only want those variables with ranking of 1 (this means they were in top 20)
			use_index.append(r)		# appending relevant indices to use_index

	opt_features = []		# initializing list to hold optimal feature set
	
	# looping through use_index to grab indices of opt feats
	for i in use_index:
		feat = all_feats[i]
		opt_features.append(feat)		# grab optimal feat name from master list of features,add to opt_features
 
	
	mod_svc = SVC(C=c, kernel='linear')		# creating the model

	mod_svc = mod_svc.fit(train_set.loc[:, opt_features], np.ravel(train_set.loc[:, ['class']]))	# fitting model 

	predicts = mod_svc.predict(test_set.loc[:, opt_features])		# predicts the class of the test set
	pred_class = predicts.tolist()		# predictions to a list
    
   	name = test_set.loc[:, 'name'].tolist()		# list of names in test set

	ac = test_set['class'].tolist()		# list of actual class in test set
    
	acc = zip(name, pred_class, ac)		# zip lists defined above to then compute f1 score
	acc_df = pd.DataFrame(acc, columns=['name', 'pred_class', 'act_class'])	#dataframe of above lists
	acc_df = acc_df.sort_index(ascending=True)	

	# initializes columns of confusion matrix in the dataframe used for evaluating the model
	acc_df['true_pos'] = 0
	acc_df['false_pos'] = 0
	acc_df['true_neg'] = 0
	acc_df['false_neg'] = 0

	# classifies each prediction as true/false pos/meg
	tp = acc_df[acc_df.pred_class == 1][acc_df.act_class == 1].index.values.tolist()
	fp = acc_df[acc_df.pred_class == 1][acc_df.act_class == 0].index.values.tolist()
	tn = acc_df[acc_df.pred_class == 0][acc_df.act_class == 0].index.values.tolist()
	fn = acc_df[acc_df.pred_class == 0][acc_df.act_class == 1].index.values.tolist()

	acc_df.loc[tp, ['true_pos']] = 1
	ntp = fp + tn + fn
	ntp.sort()
	acc_df.loc[ntp, ['true_pos']] = 0

	acc_df.loc[fp, ['false_pos']] = 1
	nfp = tp + tn + fn
	nfp.sort()
	acc_df.loc[nfp, ['false_pos']] = 0

	acc_df.loc[tn, ['true_neg']] = 1
	ntn = fp + tp + fn
	ntn.sort()
	acc_df.loc[ntn, ['true_neg']] = 0

	acc_df.loc[fn, ['false_neg']] = 1
	nfn = fp + tn + tp
	nfn.sort()
	acc_df.loc[nfn, ['false_neg']] = 0

	# calculating F1 Score
	if sum(acc_df.true_pos) + sum(acc_df.false_pos) == 0:
	    f1_score = 0
	elif sum(acc_df.true_pos) + sum(acc_df.false_neg) == 0:
	    f1_score = 0
	elif sum(acc_df.true_pos) == 0:
	    f1_score = 0
	else:
	    precision = (float(sum(acc_df.true_pos))) / (sum(acc_df.true_pos) + sum(acc_df.false_pos))
	    recall = (float(sum(acc_df.true_pos))) / (sum(acc_df.true_pos) + sum(acc_df.false_neg))
	    f1_score = 2 * ((precision * recall) / (precision + recall))

	if f1_score > best_f1:		# if f1 score of this model better than previous best
		best_f1 = f1_score		# est. new best f1
		best_c = c 				# best c
		best_features = opt_features	# best feature set
		best_n = number_variables
		best_pred = predicts
	print f1_score, c, number_variables
	return best_f1, best_c, best_features, best_n, best_pred

c_list = []
for e in range(-2, 5):
    a = 10**e
    c_list.append(a)

for i in c_list:								# optimizing over the parameter c
	for n in range(1, 39):
		best_svm = svm_feature_select(i, n)

print best_svm
use_c = best_svm[1]
best_feat = best_svm[2]

def train_svc(c, features):
	mod_svc = SVC(C=c, kernel='linear')		# creating the model

	mod_svc = mod_svc.fit(train_set.loc[:, features], np.ravel(train_set.loc[:, ['class']]))	# fitting model 

	predicts = mod_svc.predict(test_set.loc[:, features])		# predicts the class of the test set
	pred_class = predicts.tolist()		# predictions to a list
    
   	name = test_set.loc[:, 'name'].tolist()		# list of names in test set

	ac = test_set['class'].tolist()		# list of actual class in test set
    
	acc = zip(name, pred_class, ac)		# zip lists defined above to then compute f1 score
	acc_df = pd.DataFrame(acc, columns=['name', 'pred_class', 'act_class'])	#dataframe of above lists
	acc_df = acc_df.sort_index(ascending=True)	

	# initializes columns of confusion matrix in the dataframe used for evaluating the model
	acc_df['true_pos'] = 0
	acc_df['false_pos'] = 0
	acc_df['true_neg'] = 0
	acc_df['false_neg'] = 0

	# classifies each prediction as true/false pos/meg
	tp = acc_df[acc_df.pred_class == 1][acc_df.act_class == 1].index.values.tolist()
	fp = acc_df[acc_df.pred_class == 1][acc_df.act_class == 0].index.values.tolist()
	tn = acc_df[acc_df.pred_class == 0][acc_df.act_class == 0].index.values.tolist()
	fn = acc_df[acc_df.pred_class == 0][acc_df.act_class == 1].index.values.tolist()

	acc_df.loc[tp, ['true_pos']] = 1
	ntp = fp + tn + fn
	ntp.sort()
	acc_df.loc[ntp, ['true_pos']] = 0

	acc_df.loc[fp, ['false_pos']] = 1
	nfp = tp + tn + fn
	nfp.sort()
	acc_df.loc[nfp, ['false_pos']] = 0

	acc_df.loc[tn, ['true_neg']] = 1
	ntn = fp + tp + fn
	ntn.sort()
	acc_df.loc[ntn, ['true_neg']] = 0

	acc_df.loc[fn, ['false_neg']] = 1
	nfn = fp + tn + tp
	nfn.sort()
	acc_df.loc[nfn, ['false_neg']] = 0

	# calculating F1 Score
	if sum(acc_df.true_pos) + sum(acc_df.false_pos) == 0:				# so we don't divide by zero
	    f1_score = 0
	elif sum(acc_df.true_pos) + sum(acc_df.false_neg) == 0:				# so we don't divide by zero
	    f1_score = 0
	elif sum(acc_df.true_pos) == 0:										# so we don't divide by zero
	    f1_score = 0
	else:
	    precision = (float(sum(acc_df.true_pos))) / (sum(acc_df.true_pos) + sum(acc_df.false_pos))
	    recall = (float(sum(acc_df.true_pos))) / (sum(acc_df.true_pos) + sum(acc_df.false_neg))
	    f1_score = 2 * ((precision * recall) / (precision + recall))

	return f1_score, c, features, acc_df


print train_svc(use_c, best_feat)			# outputs the results of the best model


