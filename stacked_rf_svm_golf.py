import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from try_subset import subset
from sklearn.svm import SVC

# runs a random forest model and a support vector machine, then stacks the model (averaging their predicted probabilites of each observation being in the positive class). F1 scores are computed for the individual models and the stacked models to see which one performs best

np.random.seed(5)

# random forest model
def random_forest_fit(earnings_cutoff, training_set, number_trees, max_features, prob_cutoff):
	np.random.seed(5)
	global optimal_features
	#Creates training and testing set in order to evaluate the algorithm, using F1 Score as an evaluation metric as opposed to OOB Score 
	train_set = training_set[0]
	test_set = training_set[1]

	# Classifies players based on whether or not they have made at least $15,000,000 in their career, 0 is negative class(<$15,000,000) and 1 is positive class(>$15,000,000)
	train_set['class'] = 0
	
	ones = train_set[train_set['career_earnings'] >= earnings_cutoff]['id'].tolist()
	
	zeros = train_set[train_set['career_earnings'] < earnings_cutoff]['id'].tolist()
	
	train_set.loc[ones, 'class'] = 1
	train_set.loc[zeros, 'class'] = 0

	test_set['class'] = 0
	
	ones = test_set[test_set['career_earnings'] >= earnings_cutoff]['id'].tolist()
	
	zeros = test_set[test_set['career_earnings'] < earnings_cutoff]['id'].tolist()
	
	test_set.loc[ones, 'class'] = 1
	test_set.loc[zeros, 'class'] = 0

	all_feats = []
	for col in train_set.columns:
		if'avg_norm' in col:
			all_feats.append(col)

	opt_features = ['avg_norm_Top5', 'avg_norm_wgr', 'avg_norm_Top25', 'avg_norm_Cuts_Made', 'avg_norm_sand_avg', 'avg_norm_Wins', 'avg_norm_dd_avg', 'avg_norm_Score_avg', 'avg_norm_yr_earnings', 'avg_norm_Top10', 'avg_norm_putt_avg', 'avg_norm_age', 'avg_norm_fair_avg', 'avg_norm_First_avg', 'avg_norm_Four_avg', 'avg_norm_Rounds', 'avg_norm_Starts', 'avg_norm_PostCut_avg', 'avg_norm_Sec_avg']


	mod = RandomForestClassifier(n_estimators=number_trees, max_features=max_features,oob_score=True, random_state=5)
	mod = mod.fit(train_set.loc[:,opt_features], np.ravel(train_set.loc[:,['class']]))
	probs = mod.predict_proba(test_set.loc[:,opt_features])[:,1]

	
	pred_class = []
	#sets probability cutoff for predicting positve class
	for i in range(0,len(probs)):
		if probs[i] >= prob_cutoff:
			pc = 1
		else:
			pc = 0
		pred_class.append(pc)
	name = test_set.loc[:,'name'].tolist()
	
	p = probs.tolist()
	ac = test_set['class'].tolist()
	#creates data frame to show predicted vs. actual class and to compute F1 Score
	acc = zip(name, p, pred_class, ac)
	acc_df = pd.DataFrame(acc, columns=['name', 'probs', 'pred_class', 'act_class'])
	acc_df = acc_df.sort_index(ascending=True)
	
	#initializes columns of confusion matrix in the dataframe used for evaluating the model
	acc_df['true_pos'] = 0
	acc_df['false_pos'] = 0
	acc_df['true_neg'] = 0
	acc_df['false_neg'] = 0

	#classifies each precdiction as true/false pos/meg
	tp = acc_df[acc_df.pred_class == 1][acc_df.act_class == 1].index.values.tolist()
	fp = acc_df[acc_df.pred_class == 1][acc_df.act_class == 0].index.values.tolist()
	tn = acc_df[acc_df.pred_class == 0][acc_df.act_class == 0].index.values.tolist()
	fn = acc_df[acc_df.pred_class == 0][acc_df.act_class == 1].index.values.tolist()

	acc_df.loc[tp,['true_pos']] = 1
	ntp = fp+tn+fn
	ntp.sort()
	acc_df.loc[ntp,['true_pos']] = 0

	acc_df.loc[fp,['false_pos']] = 1
	nfp = tp+tn+fn
	nfp.sort()
	acc_df.loc[nfp,['false_pos']] = 0

	acc_df.loc[tn,['true_neg']] = 1
	ntn = fp+tp+fn
	ntn.sort()
	acc_df.loc[ntn,['true_neg']] = 0

	acc_df.loc[fn,['false_neg']] = 1
	nfn = fp+tn+tp
	nfn.sort()
	acc_df.loc[nfn,['false_neg']] = 0


	#calculating F1 Score
	precision = (float(sum(acc_df.true_pos)))/(sum(acc_df.true_pos)+sum(acc_df.false_pos))
	recall = (float(sum(acc_df.true_pos)))/(sum(acc_df.true_pos)+sum(acc_df.false_neg))
	f1_score = 2*((precision*recall)/(precision+recall))

	return precision, recall, f1_score, acc_df['probs']



dataframe = subset(12, 1)
rf = random_forest_fit(15000000, dataframe, 47, 4, 0.8)

rf_probs = rf[3]
rf_f1 = rf[2]








# SVM model
train_set = subset(12, 1)[0]
test_set = subset(12, 1)[1]
train_set['class'] = 0

ones = train_set[train_set['career_earnings'] >= 15000000]['id'].tolist()

zeros = train_set[train_set['career_earnings'] < 15000000]['id'].tolist()

train_set.loc[ones, 'class'] = 1
train_set.loc[zeros, 'class'] = 0

test_set['class'] = 0

ones = test_set[test_set['career_earnings'] >= 15000000]['id'].tolist()

zeros = test_set[test_set['career_earnings'] < 15000000]['id'].tolist()

test_set.loc[ones, 'class'] = 1
test_set.loc[zeros, 'class'] = 0

best_f1 = 0
best_c = 0
best_gamma = 0
best_mod = 'svc'
best_acc ='s'
best_probs = 'p'


def svm_optimize(c, gamma):
    global best_f1
    global best_c
    global best_gamma
    global best_mod
    global best_acc
    global best_probs

    all_feats = []
    for col in train_set.columns:
        if 'avg_norm' in col:
            all_feats.append(col)


    opt_features = ['avg_norm_Top5', 'avg_norm_wgr', 'avg_norm_Top25', 'avg_norm_Cuts_Made', 'avg_norm_sand_avg', 'avg_norm_Wins', 'avg_norm_dd_avg', 'avg_norm_Score_avg', 'avg_norm_yr_earnings', 'avg_norm_Top10', 'avg_norm_putt_avg', 'avg_norm_age', 'avg_norm_fair_avg', 'avg_norm_First_avg', 'avg_norm_Four_avg', 'avg_norm_Rounds', 'avg_norm_Starts', 'avg_norm_PostCut_avg', 'avg_norm_Sec_avg']

    mod_svc = SVC(C=c, kernel='rbf', gamma=gamma, probability=True)
    mod_svc = mod_svc.fit(train_set.loc[:, opt_features], np.ravel(train_set.loc[:, ['class']]))
    probs = mod_svc.predict_proba(test_set.loc[:, opt_features])[:, 1]

    pred_class = []
    # sets probability cutoff for predicting positve class
    for i in range(0, len(probs)):
        if probs[i] >= 0.8:
            pc = 1
        else:
            pc = 0
        pred_class.append(pc)
    name = test_set.loc[:, 'name'].tolist()

    ac = test_set['class'].tolist()
    p = probs.tolist()
    # creates data frame to show predicted vs. actual class and to compute F1 Score
    acc = zip(name, pred_class, ac, p)
    acc_df = pd.DataFrame(acc, columns=['name', 'pred_class', 'act_class', 'probs'])
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

    if f1_score > best_f1:
        best_f1 = f1_score
        best_c = c
        best_gamma = gamma
        best_mod = mod_svc
        best_acc = acc_df
        best_probs = acc_df['probs']
    return best_mod, best_f1, best_c, best_gamma, best_acc, best_probs


c_list = []
for e in range(-2, 5):
    a = 10**e
    c_list.append(a)

gamma_list = []
for g in range(-9, 4):
    q = 10**g
    gamma_list.append(q)

for i in c_list:
    for g in gamma_list:
        best_svm = svm_optimize(i, g)
svm_probs = best_svm[5]
svm_f1 = best_svm[1]






# stacking
w = 0.9
stacked_probs = (w*rf_probs)+((1-w)*svm_probs)
diff_stacked = rf_probs-svm_probs


pred_class = []
# sets probability cutoff for predicting positve class
for i in range(0, len(stacked_probs)):
    if stacked_probs[i] >= 0.8:
        pc = 1
    else:
        pc = 0
    pred_class.append(pc)
name = test_set.loc[:, 'name'].tolist()

ac = test_set['class'].tolist()
p = stacked_probs.tolist()
# creates data frame to show predicted vs. actual class and to compute F1 Score
acc = zip(name, pred_class, ac, p)
acc_df = pd.DataFrame(acc, columns=['name', 'pred_class', 'act_class', 'probs'])
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
precision = (float(sum(acc_df.true_pos))) / (sum(acc_df.true_pos) + sum(acc_df.false_pos))
recall = (float(sum(acc_df.true_pos))) / (sum(acc_df.true_pos) + sum(acc_df.false_neg))
stacked_f1 = 2 * ((precision * recall) / (precision + recall))

stacked_probs = stacked_probs.tolist()
rf_probs = rf_probs.tolist()
svm_probs = svm_probs.tolist()
diff_rf_svm = diff_stacked.tolist()

final = zip(name, rf_probs, svm_probs, diff_rf_svm, stacked_probs, ac)
final_df = pd.DataFrame(final, columns=['name', 'rf_probs', 'svm_probs', 'diff_rf_svm', 'stacked_probs', 'actual'])
print final_df

model =['rf', 'svm', 'stacked']
f_score = [rf_f1, svm_f1, stacked_f1]
flist = zip(model, f_score)
fscore_df = pd.DataFrame(flist, columns=['model', 'f1_score'])
print fscore_df
print precision, recall
