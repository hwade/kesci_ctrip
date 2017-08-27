# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

def feat_in_price(df_data):
	'''
	feature in_price_x: price_dedcut is within (user_minprice_x, user_maxprice_x), 1 for True, 0 for False, 2 default
	@df_data: not null DataFrame
	'''
	df_data.loc[:, 'in_price_all'] = 2
	sel_ins = (df_data['price_deduct']>=df_data['user_minprice'])\
			& (df_data['price_deduct']<=df_data['user_maxprice'])
	df_data.loc[sel_ins, 'in_price_all'] = 1
	df_data.loc[~sel_ins, 'in_price_all'] = 0
	
	for col in ['1week', '1month', '3month']:
		df_data.loc[:, 'in_price_'+col] = 2
		sel_ins = (df_data['price_deduct']>=df_data['user_minprice_'+col])\
				& (df_data['price_deduct']<=df_data['user_maxprice_'+col])
		df_data.loc[sel_ins, 'in_price_'+col] = 1
		df_data.loc[~sel_ins, 'in_price_'+col] = 0
	df_data.loc[:,'discount_ratio'] = df_data['returnvalue']/df_data['price_deduct']

def feat_is_last_id(df_data):
	'''
	feature is_last_x: 
	@df_data: not null DataFrame
	'''
	for col in ['roomid','basicroomid', 'rank', 'star']:
		df_data.loc[df_data[col]==df_data[col+'_lastord'], 'is_last_'+col] = 1
		df_data.loc[df_data[col]!=df_data[col+'_lastord'], 'is_last_'+col] = 0
	for i in range(2,7):
		col = 'roomservice_' + str(i)
		df_data.loc[df_data[col]==df_data[col+'_lastord'], 'is_last_'+col] = 1
		df_data.loc[df_data[col]!=df_data[col+'_lastord'], 'is_last_'+col] = 0
		col = 'roomtag_' + str(i)
		df_data.loc[df_data[col]==df_data[col+'_lastord'], 'is_last_'+col] = 1
		df_data.loc[df_data[col]!=df_data[col+'_lastord'], 'is_last_'+col] = 0
	df_data.loc[df_data['roomservice_8']==df_data['roomservice_8'+'_lastord'], 'is_last_'+col] = 1
	df_data.loc[df_data['roomservice_8']!=df_data['roomservice_8'+'_lastord'], 'is_last_'+col] = 0
		
	#return df_data['is_last_'+col]

def feat_roomservice_ratio(df_data):
	'''
	feature roomservice_x_ratio:
	@df_data: not null DataFrame
	'''
	# type_ == 2	
	sel_ind = df_data['roomservice_2']==1
	df_data.loc[:,'user_roomservice_2_ratio_all'] = 0.0
	df_data.loc[sel_ind,'user_roomservice_2_ratio_all'] = df_data.loc[sel_ind, 'user_roomservice_2_1ratio']

	# type_ == 3	
	sel_ind = df_data['roomservice_3'].isin([1,2,3])
	df_data.loc[:,'user_roomservice_3_ratio_all'] = 0.0
	df_data.loc[sel_ind,'user_roomservice_3_ratio_all'] = df_data.loc[sel_ind, 'user_roomservice_3_123ratio']
	
	# type_ == 4
	df_data.loc[:,'user_roomservice_4_ratio_all'] = 0.0
	for i in [0, 1, 2, 3, 4, 5]:
		df_data.loc[df_data['roomservice_4'] == i,'user_roomservice_4_ratio_all'] = \
					df_data.loc[df_data['roomservice_4'] == i, 'user_roomservice_4_'+str(i)+'ratio']
		
	# type_ == 5
	sel_ind = df_data['roomservice_5'].isin([3,4,5])
	df_data.loc[:,'user_roomservice_5_ratio_all'] = 0.0
	df_data.loc[sel_ind,'user_roomservice_5_ratio_all'] = df_data.loc[sel_ind, 'user_roomservice_5_345ratio']
	sel_ind = df_data['roomservice_5'] == 1
	df_data.loc[sel_ind,'user_roomservice_5_ratio_all'] = df_data.loc[sel_ind, 'user_roomservice_5_1ratio']

	# type_ == 6
	df_data.loc[:,'user_roomservice_6_ratio_all'] = 0.0
	df_data.loc[df_data['roomservice_6'] == 0,'user_roomservice_6_ratio_all'] = \
				df_data.loc[df_data['roomservice_6'] == 0, 'user_roomservice_6_0ratio']
	df_data.loc[df_data['roomservice_6'] == 1,'user_roomservice_6_ratio_all'] = \
				df_data.loc[df_data['roomservice_6'] == 1, 'user_roomservice_6_1ratio']
	df_data.loc[df_data['roomservice_6'] == 2,'user_roomservice_6_ratio_all'] = \
				df_data.loc[df_data['roomservice_6'] == 2, 'user_roomservice_6_2ratio']

	# type_ == 7
	df_data.loc[:,'user_roomservice_7_ratio_all'] = 0.0
	df_data.loc[df_data['roomservice_7'] == 0,'user_roomservice_7_ratio_all'] = \
				df_data.loc[df_data['roomservice_7'] == 0, 'user_roomservice_7_0ratio']

	# type_ == 8
	df_data.loc[:,'user_roomservice_8_ratio_all'] = 0.0
	df_data.loc[df_data['roomservice_8'] == 1,'user_roomservice_8_ratio_all'] = \
				df_data.loc[df_data['roomservice_8'] == 1, 'user_roomservice_8_1ratio']

	for col in ['1week','1month','3month']:
		# type_ == 3	
		sel_ind = df_data['roomservice_3'].isin([1,2,3])
		df_data.loc[:,'user_roomservice_3_ratio_'+col] = 0.0
		df_data.loc[sel_ind,'user_roomservice_3_ratio_'+col] = df_data.loc[sel_ind, 'user_roomservice_3_123ratio_'+col]
	
		# type_ == 4
		df_data.loc[:,'user_roomservice_4_ratio_'+col] = 0.0
		for i in [0, 2, 3, 4, 5]:
			df_data.loc[df_data['roomservice_4'] == i,'user_roomservice_4_ratio_'+col] = \
						df_data.loc[df_data['roomservice_4'] == i, 'user_roomservice_4_'+str(i)+'ratio_'+col]
		# type_ == 7
		df_data.loc[:,'user_roomservice_7_ratio_'+col] = 0.0
		df_data.loc[df_data['roomservice_7'] == 0,'user_roomservice_7_ratio_'+col] = \
					df_data.loc[df_data['roomservice_7'] == 0, 'user_roomservice_7_0ratio_'+col]
		df_data.loc[df_data['roomservice_7'] == 1,'user_roomservice_7_ratio_'+col] = \
					df_data.loc[df_data['roomservice_7'] == 1, 'user_roomservice_7_1ratio_'+col]
	#return df_data['user_roomservice_'+str(type_)+'_ratio_'+col]

def feat_set_label(df_data):
	''' 
	feature is_max_x:
	'''
	df_data.loc[:,'is_min_price_deduct'] = 0
	df_data.loc[:,'is_max_returnvalue'] = 0
	df_data.loc[:,'is_max_price_deduct'] = 0
	df_data.loc[:,'is_max_price_deduct'] = 0
	df_data.loc[:,'is_max_price_deduct'] = 0

def feat_price_diff(df_data):
	'''
	feature diff_price
	@return: df_data, return copy value 
	'''
	# order price diff
	order_min_price = df_data[['uid','price_deduct']].groupby(['uid']).apply(lambda g: min(g['price_deduct'])).reset_index()
	order_max_price = df_data[['uid','price_deduct']].groupby(['uid']).apply(lambda g: max(g['price_deduct'])).reset_index()
	order_avg_price = df_data[['uid','price_deduct']].groupby(['uid']).apply(lambda g: np.mean(g['price_deduct'])).reset_index()
	order_min_price.columns = ['uid', 'order_min_price']
	order_max_price.columns = ['uid', 'order_max_price']
	order_avg_price.columns = ['uid', 'order_avg_price']
	df_data = pd.merge(df_data, order_min_price, how='left', on='uid')
	df_data = pd.merge(df_data, order_max_price, how='left', on='uid')
	df_data = pd.merge(df_data, order_avg_price, how='left', on='uid')

	df_data.loc[:,'price_diff_ordermin'] = df_data['price_deduct'] - df_data['order_min_price']
	df_data.loc[:,'price_diff_ordermax'] = df_data['price_deduct'] - df_data['order_max_price']
	df_data.loc[:,'price_diff_orderavg'] = df_data['price_deduct'] - df_data['order_avg_price']
	del df_data['order_min_price'], df_data['order_max_price'], df_data['order_avg_price']

	# user price diff 1
	df_data.loc[:,'diff_deal_price'] = df_data['price_deduct'] - df_data['user_avgdealprice']
	df_data.loc[:,'diff_holiday_price'] = df_data['price_deduct'] - df_data['user_avgdealpriceholiday']
	df_data.loc[:,'diff_workday_price'] = df_data['price_deduct'] - df_data['user_avgdealpriceworkday']
	df_data.loc[:,'diff_star_price'] = df_data['price_deduct'] - df_data['user_avgprice_star']
	df_data.loc[:,'diff_return_promotion'] = df_data['returnvalue'] - df_data['user_avgpromotion']

	# user price diff 2
	df_data.loc[:,'price_diff_useravg'] = df_data['price_deduct'] - df_data['user_avgprice']
	df_data.loc[:,'price_diff_usermax'] = df_data['price_deduct'] - df_data['user_maxprice']
	df_data.loc[:,'price_diff_usermin'] = df_data['price_deduct'] - df_data['user_minprice']
	df_data.loc[:,'price_diff_usercv'] = df_data['price_deduct'] / df_data['user_cvprice']

	s = abs(df_data['price_deduct'] - df_data['user_avgprice']) - df_data['user_stdprice']
	df_data = pd.concat([df_data, pd.DataFrame(s, columns=['is_diff_price_std'])], axis=1)
	df_data.loc[df_data['is_diff_price_std']>0,'is_diff_price_std'] = 0
	df_data.loc[(df_data['is_diff_price_std']<0)&(df_data['is_diff_price_std']>-50),'is_diff_price_std'] = 1
	df_data.loc[df_data['is_diff_price_std']<0,'is_diff_price_std'] = 0

	# past price diff
	df_data.loc[:,'diff_avgprice_1week'] = df_data['price_deduct'] - df_data['user_avgprice_1week']
	df_data.loc[:,'diff_medprice_1week'] = df_data['price_deduct'] - df_data['user_medprice_1week']
	df_data.loc[:,'diff_minprice_1week'] = df_data['price_deduct'] - df_data['user_minprice_1week']
	df_data.loc[:,'diff_maxprice_1week'] = df_data['price_deduct'] - df_data['user_maxprice_1week']
	df_data.loc[:,'diff_avgprice_1month'] = df_data['price_deduct'] - df_data['user_avgprice_1month']
	df_data.loc[:,'diff_medprice_1month'] = df_data['price_deduct'] - df_data['user_medprice_1month']
	df_data.loc[:,'diff_minprice_1month'] = df_data['price_deduct'] - df_data['user_minprice_1month']
	df_data.loc[:,'diff_maxprice_1month'] = df_data['price_deduct'] - df_data['user_maxprice_1month']
	df_data.loc[:,'diff_avgprice_3month'] = df_data['price_deduct'] - df_data['user_avgprice_3month']
	df_data.loc[:,'diff_medprice_3month'] = df_data['price_deduct'] - df_data['user_medprice_3month']
	df_data.loc[:,'diff_minprice_3month'] = df_data['price_deduct'] - df_data['user_minprice_3month']
	df_data.loc[:,'diff_maxprice_3month'] = df_data['price_deduct'] - df_data['user_maxprice_3month']

	# last order price diff
	df_data.loc[:,'last_diff_return'] = df_data['returnvalue'] - df_data['return_lastord']
	df_data.loc[:,'last_diff_price'] = df_data['price_deduct'] - df_data['price_last_lastord']
	df_data.loc[:,'last_diff_hotel_price'] = df_data['price_deduct'] - df_data['hotel_minprice_lastord']
	df_data.loc[:,'last_diff_basicroom_price'] = df_data['price_deduct'] - df_data['basic_minprice_lastord']
	df_data.loc[:,'last_diff_basicroomprice'] = df_data['price_deduct'] - df_data['basic_minprice_lastord']
	
	return df_data

flag = True
def feat_max_ratio(df_data, feat_iter = None, chunk = ''):
	'''
	feature max_ratio
	@feat_iter: feature file iterator
	@chunk: feature data chunk to read
	@return: df_data, return copy value
	'''
	global flag
	file_name = '/home/hwade/work/dataMining/kesci/ctrip/data/feat_max_ratio_'+chunk+'.csv'
	set_header = False
	if not os.path.exists(file_name):
		set_header = True
	elif feat_iter != None and flag == True:
		feat = feat_iter.next()
		if type(feat) == bool:
			flag = False
			return feat_max_ratio(df_data, feat_iter=feat_iter, chunk=chunk)
		feat = feat.reset_index()
		if len(df_data[~df_data['orderid'].isin(feat['orderid'])]) == 0:
			print 'using cache feature data!'
			instance_num = len(df_data)
			cols = list(feat.columns)
			cols.remove('orderid')
			cols.remove('index')
			df_data = pd.concat([df_data, feat[cols]], axis=1)
			if instance_num - len(df_data) != 0:
				print df_data, 'concat data error!'
			return df_data
		del feat
		
	basic_col_len= len(df_data.columns)

	# apply group
	order_max_comment = df_data.groupby(['orderid']).apply(lambda g: max(g['basic_comment_ratio'])).reset_index()
	order_max_comment_week = df_data.groupby(['orderid']).apply(lambda g: max(g['basic_week_ordernum_ratio'])).reset_index()
	order_max_comment_3day = df_data.groupby(['orderid']).apply(lambda g: max(g['basic_recent3_ordernum_ratio'])).reset_index()
	order_max_basic_30day = df_data.groupby(['orderid']).apply(lambda g: max(g['basic_30days_ordnumratio'])).reset_index()
	order_max_basic_real = df_data.groupby(['orderid']).apply(lambda g: max(g['basic_30days_realratio'])).reset_index()
	order_max_room_30day = df_data.groupby(['orderid']).apply(lambda g: max(g['room_30days_ordnumratio'])).reset_index()
	order_max_room_real = df_data.groupby(['orderid']).apply(lambda g: max(g['room_30days_realratio'])).reset_index()
	
	basic_size = df_data.groupby(['orderid','basicroomid']).size().reset_index()
	basic_room_size = df_data.groupby(['basicroomid','roomid']).size().reset_index()
	basic_max_comment_30day = df_data.groupby(['basicroomid']).apply(lambda g: max(g['room_30days_ordnumratio'])).reset_index()
	basic_max_real = df_data.groupby(['basicroomid']).apply(lambda g: max(g['room_30days_realratio'])).reset_index()
	basic_max_return = df_data.groupby(['orderid','basicroomid']).apply(lambda g: max(g['returnvalue'])).reset_index()
	basic_min_price = df_data.groupby(['orderid','basicroomid']).apply(lambda g: min(g['price_deduct'])).reset_index()
	basic_max_return_ratio = df_data.groupby(['orderid','basicroomid']).apply(\
								lambda g: max(g['returnvalue']/g['price_deduct'])).reset_index()

	# fix columns
	order_max_comment.columns = ['orderid','order_max_comment']
	order_max_comment_week.columns = ['orderid','order_max_comment_week']
	order_max_comment_3day.columns = ['orderid','order_max_comment_3day']
	order_max_basic_30day.columns = ['orderid','order_max_basic_30day']
	order_max_basic_real.columns = ['orderid','order_max_basic_real']
	order_max_room_30day.columns = ['orderid','order_max_room_30day']
	order_max_room_real.columns = ['orderid','order_max_room_real']

	basic_size.columns = ['orderid','basicroomid','basic_size']
	basic_room_size.columns = ['basicroomid','roomid','basic_room_size']
	basic_max_comment_30day.columns = ['basicroomid','basic_max_comment_30day']
	basic_max_real.columns = ['basicroomid','basic_max_real']
	basic_max_return.columns = ['orderid','basicroomid','basic_max_return']
	basic_min_price.columns = ['orderid','basicroomid','basic_min_price']
	basic_max_return_ratio.columns = ['orderid','basicroomid','basic_max_return_ratio']

	# merge key
	df_data = pd.merge(df_data, order_max_comment, how='left', on=['orderid'])
	df_data = pd.merge(df_data, order_max_comment_week, how='left', on=['orderid'])
	df_data = pd.merge(df_data, order_max_comment_3day, how='left', on=['orderid'])
	df_data = pd.merge(df_data, order_max_basic_30day, how='left', on=['orderid'])
	df_data = pd.merge(df_data, order_max_basic_real, how='left', on=['orderid'])
	df_data = pd.merge(df_data, order_max_room_30day, how='left', on=['orderid'])
	df_data = pd.merge(df_data, order_max_room_real, how='left', on=['orderid'])

	df_data = pd.merge(df_data, basic_size, how='left', on=['orderid','basicroomid'])
	df_data = pd.merge(df_data, basic_room_size, how='left', on=['basicroomid','roomid'])
	df_data = pd.merge(df_data, basic_max_comment_30day, how='left', on=['basicroomid'])
	df_data = pd.merge(df_data, basic_max_real, how='left', on=['basicroomid'])
	df_data = pd.merge(df_data, basic_max_return, how='left', on=['orderid','basicroomid'])
	df_data = pd.merge(df_data, basic_min_price, how='left', on=['orderid','basicroomid'])
	df_data = pd.merge(df_data, basic_max_return_ratio, how='left', on=['orderid','basicroomid'])

	# diff feature
	df_data.loc[:, 'diff_order_max_comment'] = df_data['order_max_comment'] - df_data['basic_comment_ratio']
	df_data.loc[:, 'diff_order_max_comment_week'] = df_data['order_max_comment_week'] - df_data['basic_week_ordernum_ratio']
	df_data.loc[:, 'diff_order_max_comment_3day'] = df_data['order_max_comment_3day'] - df_data['basic_recent3_ordernum_ratio']
	df_data.loc[:, 'diff_order_max_basic_30day'] = df_data['order_max_basic_30day'] - df_data['basic_30days_ordnumratio']
	df_data.loc[:, 'diff_order_max_basic_real'] = df_data['order_max_basic_real'] - df_data['basic_30days_realratio']
	df_data.loc[:, 'diff_order_max_room_30day'] = df_data['order_max_room_30day'] - df_data['room_30days_ordnumratio']
	df_data.loc[:, 'diff_order_max_room_real'] = df_data['order_max_room_real'] - df_data['room_30days_realratio']

	df_data.loc[:, 'diff_basic_max_comment_30day'] = df_data['basic_max_comment_30day'] - df_data['room_30days_ordnumratio']
	df_data.loc[:, 'diff_basic_max_real'] = df_data['basic_max_real'] - df_data['room_30days_realratio']
	df_data.loc[:, 'diff_basic_max_return'] = df_data['basic_max_return'] - df_data['returnvalue']
	df_data.loc[:, 'diff_basic_min_price'] = df_data['price_deduct'] - df_data['basic_min_price']
	df_data.loc[:, 'diff_basic_max_return_ratio'] = df_data['basic_max_return_ratio'] - df_data['discount_ratio']
	
	# filter useless columns
	del df_data['order_max_comment'], df_data['order_max_comment_week'], df_data['order_max_comment_3day'], \
		df_data['order_max_basic_30day'], df_data['order_max_basic_real'], df_data['order_max_room_30day'], \
		df_data['order_max_room_real']

	# save feature
	cols = list(df_data.columns[basic_col_len:])
	cols.append('orderid')
	for col in cols:
		# save memory
		if df_data[col].dtype == np.dtype('float64'):
			df_data.loc[:,col] = df_data[col].astype(np.float32)
		elif df_data[col].dtype == np.dtype('int64'):
			df_data.loc[:,col] = df_data[col].astype(np.int32)

	df_data[cols].to_csv(file_name, index=False, header=set_header, mode='a')

	return df_data

def feat_combine_label_feature(df_data, chunk):
	file_name = '/home/hwade/work/dataMining/kesci/ctrip/data/feat_combine_label_feature_'+chunk+'.csv'
	if not os.path.exists(file_name):
		base_col_len = len(df_data.columns)
		tri_fun = lambda g: len(g[g['orderlabel']==1])/float(len(g))
		cols = []
		for i in [1,4]:
			cols.append('roomtag_'+str(i))
			for j in range(1,9):
				cols.append('roomservice_'+str(j))
				cols.append('roomtag_service_'+str(i)+'_'+str(j))
				cols.append('roomtag_service_'+str(i)+'_'+str(j)+'_tri_ratio')
				s = df_data.groupby(['roomtag_'+str(i),'roomservice_'+str(j)]).apply(tri_fun).reset_index()
				s.columns = ['roomtag_'+str(i),'roomservice_'+str(j),'roomtag_service_'+str(i)+'_'+str(j)+'_tri_ratio']
				df_data = pd.merge(df_data, s, how='left', on=['roomtag_'+str(i),'roomservice_'+str(j)])
				df_data.loc[:,'roomtag_service_'+str(i)+'_'+str(j)] = df_data['roomtag_'+str(i)]*10 \
																	+ df_data['roomservice_'+str(j)]
		cols = list(set(cols))
		df_data[cols].to_csv(file_name, index=False)
	else:
		feat = pd.read_csv(file_name)
		for i in [1,4]:
			for j in range(1,9):
				cols = ['roomtag_'+str(i),'roomservice_'+str(j),'roomtag_service_'+str(i)+'_'+str(j),\
						'roomtag_service_'+str(i)+'_'+str(j)+'_tri_ratio']
				df_data = pd.merge(df_data, feat[cols].drop_duplicates(), how='left', on=['roomtag_'+str(i),'roomservice_'+str(j)])
		del feat
		
	return df_data	



