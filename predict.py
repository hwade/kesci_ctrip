# -*- coding: utf-8 -*-
import gc
import time
import json
import warnings
from robot import *
from constMsg import *
from preprocess import *
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV

warnings.filterwarnings("ignore")

class make_scorer_xgb():
	def __init__(self, scorer_fun, kwargs):
		self.scorer_fun = scorer_fun
		self.kwargs = kwargs

	def __call__(self, pred_y, true_y):
		return self.scorer_fun(pred_y, true_y, self.kwargs)

def get_label_from_proba(pred_y, test_X):
	''' get label base on the probability from the same orderid '''
	
	def mark_label_pos(data):
		index = list(data['orderid'])
		index = sorted([(o,i) for i, o in enumerate(index)], key=lambda e:e[0], reverse=True)
		return index
	
	index = mark_label_pos(test_X)

	order = index[0][0]
	max_i = index[0][1]
	max_proba = pred_y[max_i]
	pred = [0] * len(pred_y)
	for item in index:
		if order != item[0]:
			pred[max_i] = 1
			order = item[0]
			max_i = item[1]
			max_proba = pred_y[item[1]]

		if max_proba < pred_y[item[1]]:
			max_proba = pred_y[item[1]]
			max_i = item[1]
	pred[max_i] = 1
	
	return pred

def recall_scorer(d1_y, d2_y, test_X, is_local = False):
	''' 
	scorer 
	d1_y: array like, probability of prediction in xgb, true value in sklearn
	d2_y: xgb.DMatrix object in xgb, pred value in sklearn
	data: get data label
	return: (str, float), value pair
	'''	
	if is_local:
		# sklearn
		pred = get_label_from_proba(d2_y, test_X)
		score = sum(np.array(pred) * np.array(d1_y))/float(sum(pred))
		return score
	else:
		# xgb
		pred = get_label_from_proba(d1_y, test_X)
		score = sum(np.array(pred) * np.array(d2_y.get_label()))/float(sum(pred))
		return ('recall', score)

def submit_result(model, test_data_iterator, cols, feat_iter=None, chunk=''):
	''' submit predict result'''
	start = time.time()
	loop = True
	num = 1
	# test_data_iterator = load_data(dataType = 'test', chunksize = 700000)
	pd.DataFrame([], columns=['orderid','predict_roomid']).to_csv('submission.csv', index=False)
	while loop:
		try:
			print_mode('processing chunk %d test data...'%num)
			test_set = test_data_iterator.next()

			# transfer test data feature	
			test_set = transfer_feature(test_set, feat_iter=feat_iter, chunk=chunk)
			test_set = test_set[cols]

			# make predict by model
			pred_y = model.predict_proba(test_set)[:,1]
			pred = get_label_from_proba(pred_y, test_set)

			# adjust result
			test_set = pd.concat([test_set.reset_index(), pd.DataFrame(pred, columns=['orderlabel'])], axis=1)
			result = test_set.loc[test_set['orderlabel']==1,['orderid','roomid']]
			result.loc[:,'orderid'] = ['ORDER_'+str(i) for i in result['orderid']]
			result.loc[:,'roomid'] = ['ROOM_'+str(i) for i in result['roomid']]
			result[['orderid','roomid']].to_csv('submission.csv', index=False, header=False, mode='a')

			num += 1
			del result, test_set, pred_y, pred
			gc.collect()
		except StopIteration:
			loop = False
			print_mode('Form submission file successfully! Iter num: %d'%num)

	# filter duplicate orderid
	submit_raw = pd.read_csv('submission.csv')
	ordid_size = submit_raw.groupby(['orderid']).size()
	index = list(ordid_size[ordid_size>1].index)
	for i in index:
		del_ind = submit_raw[submit_raw['orderid']==i].index[1]
		submit_raw = submit_raw.drop(del_ind, axis=0)
	submit_raw.to_csv('submission.csv', index=False)
	print_mode('Filter duplicate orderid.\nGenerate test data cost time: %.6fs'%(time.time()-start))

def model_fit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5):
	if useTrainCV:
		print_mode('')
	#Fit the algorithm on the data
	alg.fit(dtrain[predictors], dtrain['Disbursed'],eval_metric='auc')

	#Predict training set:
	dtrain_predictions = alg.predict(dtrain[predictors])
	dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

	#Print model report:
	print_mode("\nModel Report")
	print_mode("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
	print_mode("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))

	feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
	feat_imp.plot(kind='bar', title='Feature Importances')
	plt.ylabel('Feature Importance Score')

def model_xgb(train_set, train_labels):
	''' xgboost model '''
	import xgboost as xgb
	from sklearn.model_selection import GridSearchCV
	
	clf = xgb.XGBClassifier(
		# num_class: 2,
		max_depth=XGB_ARGS_VALUES['max_depth'], 
		learning_rate=XGB_ARGS_VALUES['learning_rate'], 
		n_estimators=XGB_ARGS_VALUES['n_estimators'], 
		silent=XGB_ARGS_VALUES['silent'], 
		objective=XGB_ARGS_VALUES['objective'], 
		booster=XGB_ARGS_VALUES['booster'], 
		gamma=XGB_ARGS_VALUES['gamma'], 
		min_child_weight=XGB_ARGS_VALUES['min_child_weight'], 
		max_delta_step=XGB_ARGS_VALUES['max_delta_step'], 
		subsample=XGB_ARGS_VALUES['subsample'], 
		colsample_bytree=XGB_ARGS_VALUES['colsample_bytree'], 
		colsample_bylevel=XGB_ARGS_VALUES['colsample_bylevel'], 
		reg_alpha=XGB_ARGS_VALUES['reg_alpha'], 
		reg_lambda=XGB_ARGS_VALUES['reg_lambda'], 
		scale_pos_weight=XGB_ARGS_VALUES['scale_pos_weight'], 
		n_jobs=6
	)
	xgb_param = clf.get_xgb_params()
	xgtrain = xgb.DMatrix(train_set.values, label=train_labels.values)
	cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=clf.get_params()['n_estimators'], nfold=2, 
					metrics='auc', early_stopping_rounds=100)
	XGB_ARGS_VALUES['n_estimators'] = cvresult.shape[0]
	print_mode('best n_estimators: %d'%cvresult.shape[0])
	
	best_model = None
	best_score = 0
	
	# find the best args
	while True:
		a_num = checkInput(XGB_INPUT_REMINDER, json.loads)
		if isinstance(a_num, int):
			a_num = [a_num]
		if 0 in a_num:
			break

		start_time = time.time()

		# input args
		params = {}
		for i in a_num:
			if i > 15 or i < 1:
				params = XGB_ARGS_VALUES
				continue
			nam = XGB_ARGS_PAIR[i-1]
			arg = checkInput(XGB_ARGS_INFO[nam] + '\n[!]Input list object!', json.loads)
			params[nam] = list(np.arange(arg[0],arg[1],arg[2]))
			#params[nam] = checkInput(XGB_ARGS_INFO[nam], XGB_CHECK_INPUT[nam])
		clf = xgb.XGBClassifier(
			max_depth=XGB_ARGS_VALUES['max_depth'], 
			learning_rate=XGB_ARGS_VALUES['learning_rate'], 
			n_estimators=XGB_ARGS_VALUES['n_estimators'], 
			silent=XGB_ARGS_VALUES['silent'], 
			objective=XGB_ARGS_VALUES['objective'], 
			booster=XGB_ARGS_VALUES['booster'], 
			gamma=XGB_ARGS_VALUES['gamma'], 
			min_child_weight=XGB_ARGS_VALUES['min_child_weight'], 
			max_delta_step=XGB_ARGS_VALUES['max_delta_step'], 
			subsample=XGB_ARGS_VALUES['subsample'], 
			colsample_bytree=XGB_ARGS_VALUES['colsample_bytree'], 
			colsample_bylevel=XGB_ARGS_VALUES['colsample_bylevel'], 
			reg_alpha=XGB_ARGS_VALUES['reg_alpha'], 
			reg_lambda=XGB_ARGS_VALUES['reg_lambda'], 
			scale_pos_weight=XGB_ARGS_VALUES['scale_pos_weight']
		)
		gs = GridSearchCV(estimator=clf, param_grid=params, scoring='roc_auc', n_jobs=4, iid=False, cv=2)
			
		# gridsearch fit
		gs.fit(train_set, train_labels)

		# print result
		print_mode(str(gs.best_params_)+'\n'+str(gs.best_score_)+'\nAdjust arguments took time: %ds'%(time.time()-start_time))
		
		for para in gs.best_params_.keys():
			XGB_ARGS_VALUES[para] = gs.best_params_[para]
	

	while True:
		a_num = checkInput(XGB_INPUT_REMINDER, json.loads)
		if isinstance(a_num, int):
			a_num = [a_num]
		if 0 in a_num:
			break

		start_time = time.time()

		# input args
		params = {}
		for i in a_num:
			if i > 15 or i < 1:
				params = XGB_ARGS_VALUES
				continue
			nam = XGB_ARGS_PAIR[i-1]
			params[nam] = checkInput(XGB_ARGS_INFO[nam], XGB_CHECK_INPUT[nam])
		
		# default args
		for para in XGB_ARGS_VALUES.keys():
			if not params.has_key(para):
				params[para] = XGB_ARGS_VALUES[para]

		clf = xgb.XGBClassifier(
			# num_class: 2,
			max_depth=params['max_depth'], 
			learning_rate=params['learning_rate'], 
			n_estimators=params['n_estimators'], 
			silent=params['silent'], 
			objective=params['objective'], 
			booster=params['booster'], 
			gamma=params['gamma'], 
			min_child_weight=params['min_child_weight'], 
			max_delta_step=params['max_delta_step'], 
			subsample=params['subsample'], 
			colsample_bytree=params['colsample_bytree'], 
			colsample_bylevel=params['colsample_bylevel'], 
			reg_alpha=params['reg_alpha'], 
			reg_lambda=params['reg_lambda'], 
			scale_pos_weight=params['scale_pos_weight'], 
			n_jobs=6
		)
		# split data
		# NOTICE! Here can't shuffle the data!
		train_size = int(0.7 * len(train_labels))
		train_X, train_y = train_set.loc[train_set.index[:train_size]], train_labels[train_labels.index[:train_size]]
		test_X, test_y = train_set.loc[train_set.index[train_size:]], train_labels[train_labels.index[train_size:]]

		# make scorer function
		orderid = test_X[['orderid']]
		scorer = make_scorer_xgb(recall_scorer, orderid)

		# xgb fit
		clf.fit(train_X, train_y, eval_set=[(test_X, test_y)], eval_metric=scorer)

		# local test set
		pred_y = clf.predict_proba(test_X)[:,1]

		# get score
		score = recall_scorer(test_y, pred_y, test_X = orderid, is_local = True)

		# save best args
		if score > best_score:
			best_score = score
			best_model = clf
		# find the most freq param
		for para in params.keys():
			XGB_ARGS_VALUES[para] = params[para]

		#print 'GridSearch cv results:\n%s'%str(gs.cv_results_)
		print_mode('\nThis iter score: %.6f\nBest score: %.6f'%(score, best_score))
		print_mode('Fit cost: %.6fs'%(time.time()-start_time))

	return best_model

def clean_data(df_data, fill_method = 'mode'):
	''' clean data and fill empty'''
	# transfer id feature to int
	for col in ID_COL:
		if col in USE_COL:
			df_data[col].fillna('_0', inplace=True)
			df_data.loc[:, col] = [int(i.split('_')[1]) for i in df_data[col]]
	for col in df_data.columns:
		if col.find('ratio') >= 0 or col.find('price') >= 0:
			df_data[col].fillna(df_data[col].mean(), inplace=True)
		elif col.find('roomservice') >= 0 or col.find('roomtag') >= 0:
			df_data[col].fillna(10, inplace=True)
		elif fill_method == 'mode':
			df_data[col].fillna(df_data[col].mode()[0], inplace=True)
		elif fill_method == 'mean':
			df_data[col].fillna(df_data[col].mean(), inplace=True)
		elif fill_method == 'quan':
			# quantile 0.5
			df_data[col].fillna(df_data[col].quantile(0.5), inplace=True)
		elif fill_method == 'pad':
			df_data[col].fillna(method='pad', inplace=True)
		elif fill_method == 'bfill':
			df_data[col].fillna(method='bfill', inplace=True)
		else:
			print_mode('fill_method {%s} dose not exist!'%fill_method)
			return
		# save memory
		df_data.loc[:,col] = df_data[col].astype(np.float32) if (df_data[col].dtype == np.dtype('float64')) \
						else df_data[col].astype(np.int32)

def load_feat(file_name, chunksize):
	''' load feature data as cache start '''	
	data = pd.read_csv(file_name, iterator=True)
	loop = True
	while loop:
		try:
			data_c = data.get_chunk(chunksize)
			yield data_c
			del data_c
			gc.collect()
		except StopIteration:
			loop = False
			print_mode('Stop loading feature cache')
			yield loop

def load_data(dataType = 'train', chunksize = 1000000):
	''' load train or test data '''

	print_mode('loading native data...')
	use_col = USE_COL
	if dataType == 'test':
		use_col.remove('orderlabel')
	data = pd.read_csv(CUR_PATH + '/data/competition_'+dataType+'.txt', sep='\t', iterator=True, dtype=DATA_DTYPE, usecols=use_col)

	loop = True
	while loop:
		try:
			data_c = data.get_chunk(chunksize)

			# fill nan
			clean_data(data_c, fill_method = 'mean')
			
			if dataType == 'train':
				yield data_c, data_c['orderlabel']
			else:
				yield data_c
			del data_c
			gc.collect()
		except StopIteration:
			loop = False
			print_mode('Stop loading %s data iteration'%dataType)

def stack_model_as_feature(df_data, label):
	from sklearn.linear_model import LogisticRegression
	lr = LogisticRegression(class_weight='balanced', n_jobs=3)
	for i in range(1,5):
		lr.fit(df_data, label)
		pred = lr.predict_proba(df_data)[:,1]
		df_data.loc[:,'lr_'+str(i)] = pred

def transfer_feature(df_data, feat_iter = None, chunk=''):
	'''
	transfer data features
	@df_data: pd.DataFrame, train or test data
	'''
	start = time.time()
	print_mode('transfering feature...')

	feat_in_price(df_data)			# in_price_all
	feat_is_last_id(df_data)		# is_last_roomid
	feat_roomservice_ratio(df_data)	# user_roomservice_3_ratio_col

	# contain pandas.merge operation, cannot cover original data set automatically
	df_data = feat_price_diff(df_data)	# price_diff_x
	# use cache file
	df_data = feat_max_ratio(df_data, feat_iter=feat_iter, chunk=chunk)	# basic_room_max_ratio_diff
	#df_data = feat_combine_label_feature(df_data, chunk)

	df_data.fillna(df_data.mean(), inplace=True)
	df_data.fillna(0, inplace=True)
	for col in df_data.columns:
		# save memory
		if df_data[col].dtype == np.dtype('float64'):
			df_data.loc[:,col] = df_data[col].astype(np.float32)
		elif df_data[col].dtype == np.dtype('int64'):
			df_data.loc[:,col] = df_data[col].astype(np.int32)

	print_mode('complete feature transfer, cost time: %.6fs'%(time.time() - start))

	return df_data

def predict(model_fun = model_xgb):
	'''
	preprocess data
	'''
	chunksize = checkInput('>input chunk size: ', int)
	skip_num = checkInput('>input chunk number you want to skip (0~%d): '%(7500000/chunksize), int)
	
	chunk = str(chunksize)+'_'+str(skip_num)
	file_name = CUR_PATH + '/data/feat_max_ratio_' + chunk + '.csv'	
	if not os.path.exists(file_name):
		feat_iter = None
	else:
		feat_iter = load_feat(file_name=file_name, chunksize=chunksize)

	train_data_iterator = load_data(dataType = 'train', chunksize = chunksize)
	test_data_iterator = load_data(dataType = 'test', chunksize = chunksize)
	while skip_num > 0:
		# skip training data skip_num instances
		train_data_iterator.next()
		skip_num -= 1

	# loading training data (chunk)
	train_set, train_labels = train_data_iterator.next()
	basic_col_len= len(train_set.columns)

	# transfer data
	train_set = transfer_feature(train_set, feat_iter=feat_iter, chunk=chunk)
	
	# usable native columns
	cols = FIT_COL
	cols.extend(list(train_set.columns[basic_col_len:]))
	train_set = train_set[cols]

	# combining model predict as feature
	#stack_model_as_feature(train_set, train_labels)

	# fit train data model
	best_model = model_fun(train_set, train_labels)
	del train_set, train_labels
	gc.collect()
		
	# predict testing data result
	submit_result(best_model, test_data_iterator, cols, feat_iter=feat_iter, chunk=chunk)

if __name__ == '__main__':
	start_time = time.time()
	
	try:
		global CUR_PATH
		# start wechat robot
		CUR_PATH = robot_start()

		predict(model_fun = model_xgb)

	except Exception, e:
		print_mode('Encounter error with {%s}'%(e), msgType = 'ERROR')

	print_mode( 'Total cost time: %4f s'%(time.time()-start_time), keep_alive = False)
