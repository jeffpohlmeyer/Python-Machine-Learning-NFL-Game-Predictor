import os

os.system('clear')

import pandas as pd
import datetime
import numpy as np
from scipy import stats

import statsmodels.formula.api as smf
import statsmodels.stats.api as sms

from sklearn import preprocessing, model_selection, svm, linear_model
from sklearn.feature_selection import RFE, RFECV
from tpot import TPOTRegressor


import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

from plot_learning_curve import plot_learning_curve

from time import time

import plotting as pt


def input():
	raw_input('press enter to continue...')

""" This prints the full dataframe instead of just the top and bottom """
def print_full(x):
	pd.set_option('display.max_rows', len(x))
	print(x)
	pd.reset_option('display.max_rows')

"""This method converts the calculated probability to a spread"""
def spread_conversion(x):
	"""
	http://www.bettingtalk.com/win-probability-percentage-point-spread-nfl-nba/
	"""

	home_dog = False
	if x < 0.5:
		x = 1.0 - x
		home_dog = True

	if x < .513:
		return_val = -0.5
	elif x < .525:
		return_val = -1
	elif x < .535:
		return_val = -1.5
	elif x < .545:
		return_val = -2
	elif x < .594:
		return_val = -2.5
	elif x < .643:
		return_val = -3
	elif x < .658:
		return_val = -3.5
	elif x < .673:
		return_val = -4
	elif x < .681:
		return_val = -4.5
	elif x < .69:
		return_val = -5
	elif x < .707:
		return_val = -5.5
	elif x < .724:
		return_val = -6
	elif x < .752:
		return_val = -6.5
	elif x < .781:
		return_val = -7
	elif x < .791:
		return_val = -7.5
	elif x < .802:
		return_val = -8
	elif x < .807:
		return_val = -8.5
	elif x < .811:
		return_val = -9
	elif x < .836:
		return_val = -9.5
	elif x < .86:
		return_val = -10
	elif x < .871:
		return_val = -10.5
	elif x < .882:
		return_val = -11
	elif x < .885:
		return_val = -11.5
	elif x < .887:
		return_val = -12
	elif x < .893:
		return_val = -12.5
	elif x < .9:
		return_val = -13
	elif x < .924:
		return_val = -13.5
	elif x < .949:
		return_val = -14
	elif x < .956:
		return_val = -14.5
	elif x < .963:
		return_val = -15
	elif x < .981:
		return_val = -15.5
	elif x < .998:
		return_val = -16
	else:
		return_val = -16.5

	if home_dog == True:
		return float(-return_val)
	else:
		return float(return_val)

def normality_check(df):
	""" Check to see if the distribution of spreads is somewhat normal """
	plt.figure()
	sns.distplot(df)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	res = stats.probplot(df,plot=plt)
	ax.set_title(df.name)
	#skewness and kurtosis
	print df.name
	print "Skewness: %f" % df.skew()
	print "Kurtosis: %f" % df.kurt()
	print

def statistical_analysis(df):
	""" Check correlation of features to spread """
	#correlation matrix
	corrmat = df.corr()
	f, ax = plt.subplots(figsize=(12, 9))
	hm = sns.heatmap(corrmat, cbar=True, annot=True, square=True, fmt='.2f')
	plt.yticks(rotation=0)
	plt.xticks(rotation=90)

	corrvec = abs(df.corr()['result_spread'].copy())
	print corrvec.sort_values()

	#scatterplot
	sns.set()
	cols = ['result_spread','rush_attempt_diff','turn_diff','yards_diff','third_diff','sack_diff','sack_ydiff','p_attempt_diff']
	sns.pairplot(df[cols], size = 2.5)

	# normality_check(df['result_spread'])
	# normality_check(df['rush_attempt_diff'])
	# normality_check(df['turn_diff'])
	# normality_check(df['yards_diff'])
	# normality_check(df['third_diff'])
	# normality_check(df['sack_diff'])
	# normality_check(df['sack_ydiff'])
	# normality_check(df['poss_diff'])
	# normality_check(df['p_attempt_diff'])
	""" Rush attempt shows light tails but otherwise these main features appear normally distributed """

	# plt.show(block=False)

def clean_data(df):
	""" Clean Data """
	# Passing stats
	home_pass_stats = pd.DataFrame(df['hpass_tot'].str.split('-').tolist(),columns="completions attempts yards touchdowns interceptions".split())
	away_pass_stats = pd.DataFrame(df['apass_tot'].str.split('-').tolist(),columns="completions attempts yards touchdowns interceptions".split())

	away_pass_stats.replace('neg7','-7',inplace=True)
	home_pass_stats['yards'] = home_pass_stats['yards'].astype(float)
	home_pass_stats['attempts'] = home_pass_stats['attempts'].astype(float)
	away_pass_stats['yards'] = away_pass_stats.yards.astype(float)
	away_pass_stats['attempts'] = away_pass_stats.attempts.astype(float)
	df['home_pass_yards'] = home_pass_stats['yards']
	df['home_pass_attempts'] = home_pass_stats['attempts']
	df['away_pass_yards'] = away_pass_stats['yards']
	df['away_pass_attempts'] = away_pass_stats['attempts']
	del home_pass_stats, away_pass_stats

	# Rushing stats
	df['home_rush'].astype(float)
	df['hrush_att'].astype(float)
	df['away_rush'].astype(float)
	df['arush_att'].astype(float)

	"""
	# Calculate offensive efficiency (pass yards per attempt and rush yards per carry)
	df['home_pass'] = home_pass_stats['yards'] / home_pass_stats['attempts']
	df['away_pass'] = away_pass_stats['yards'] / away_pass_stats['attempts']
	df['home_rush'] = df['home_rush'] / df['hrush_att']
	df['away_rush'] = df['away_rush'] / df['arush_att']
	"""

	# Change the string date data in df to datetime format
	df['game_date'] = pd.to_datetime(df['game_date'],format='%Y-%m-%d')

	""" Create home time of possession differential """
	# Convert objects to datetime values
	df['home_poss'] = pd.to_datetime(df['home_poss'],format='%M:%S')
	df['away_poss'] = pd.to_datetime(df['away_poss'],format='%M:%S')
	# Convert datetime values to fractions of an hour
	df['home_poss'] = df['home_poss'].dt.minute / 60.0 + df['home_poss'].dt.second / 3600.0
	df['away_poss'] = df['away_poss'].dt.minute / 60.0 + df['away_poss'].dt.second / 3600.0
	# Find total possession time (only really matters because games can go to overtime)
	# And re-weight time of possession
	df['total_poss'] = df['home_poss'] + df['away_poss']
	df['home_poss'] = df['home_poss'] / df['total_poss']
	df['away_poss'] = df['away_poss'] / df['total_poss']
	df.drop('total_poss',axis = 1, inplace=True)	# Delete total possession column as it's no longer needed

	return df

def training_set(df,cutoff_date):
	""" Create training set to analyze """
	train_set = df[df.game_date < cutoff_date].copy()

	train_set['result_spread'] = train_set['home_score'] - train_set['away_score']
	train_set.drop(['overunder','apass_tot','hpass_tot','away_team','home_team','home_score','away_score','vegasline','home_pens','away_pens'],axis = 1,inplace = True)

	# Fill NaNs with outlier values
	train_set.fillna(-999999, inplace=True)

	# Calculate yards differential
	train_set['yards_diff'] = train_set['home_oyds'] - train_set['away_oyds']

	# Calculate time of possession differential
	train_set['poss_diff'] = train_set['home_poss'] - train_set['away_poss']

	# Calculate third down percentage differential
	train_set['third_diff'] = train_set['home_third'] - train_set['away_third']

	# Calculate third down percentage differential
	train_set['fourth_diff'] = train_set['home_four'] - train_set['away_four']

	# Calculate turnover differential
	train_set['turn_diff'] = train_set['home_turn'] - train_set['away_turn']

	# Calculate sack quantity differential
	train_set['sack_diff'] = train_set['home_sack'] - train_set['away_sack']

	# Calculate sack yards differential
	train_set['sack_ydiff'] = train_set['hsack_yds'] - train_set['asack_yds']

	# Calculate penalty yards differential
	train_set['pens_diff'] = train_set['hpens_yds'] - train_set['apens_yds']

	# # Calculate passing yardage differential
	# train_set['pass_diff'] = train_set['home_pass'] - train_set['away_pass']

	# Calculate rushing yardage differential
	train_set['rush_diff'] = train_set['home_rush'] - train_set['away_rush']

	# Calculate rush attempts differential
	train_set['rush_attempt_diff'] = train_set['hrush_att'] - train_set['arush_att']

	# Calculate passing yardage differential
	train_set['p_yards_diff'] = train_set['home_pass_yards'] - train_set['away_pass_yards']

	# Calculate passing attempts differential
	train_set['p_attempt_diff'] = train_set['home_pass_attempts'] - train_set['away_pass_attempts']

	train_set.sort_values('game_date',inplace=True)
	train_set.drop(['home_oyds','away_oyds','home_poss','away_poss','home_third','away_third','home_four','away_four','home_turn','away_turn','home_sack','away_sack','hsack_yds','asack_yds','hpens_yds','apens_yds','home_pass','away_pass','home_rush','away_rush','hrush_att','arush_att','home_pass_yards','away_pass_yards','home_pass_attempts','away_pass_attempts'],axis = 1,inplace = True)

	# print train_set.columns
	# statistical_analysis(train_set)
	# input()
	return train_set

def prediction_set(df,cutoff_date):
	""" Creating prediction dataset """
	# Create dataset to be used for prediction
	predicting_set = df[df.game_date >= cutoff_date].set_index('game_date')
	predicting_set = predicting_set.sort_index()	# Sort ascending by index
	predicting_set['week'] = np.nan # Create column indicating which game week the team is in

	# Populating the week column of predicting_set
	start_date = predicting_set.index.min().date()  # Find the first game date
	end_date = predicting_set.index.max().date()	# Find the last game date
	date_val = start_date						  # date_val is the value used to cycle through the datetime objects
	week_val = 1									# week_val is the iterating value that tracks the "current" week
	week_dict={}									# Create an empty dictionary
	# Cycle through the number of days
	for _ in range((end_date - start_date).days+1):
		if date_val.weekday() == 1:			  # Check if the weekday value is a Tuesday
			week_val += 1						  # Increment since I'm considering Tuesday to start a new week
		week_dict[date_val] = week_val			# Update the dictionary value for the "current" date
		date_val += datetime.timedelta(days=1)	# Increment date_val by one day
	week_dict[datetime.date(2017,2,5)] = 21	  # This is manually done since there are two weeks between conf championships and the final game

	# Update the week column in the dataframe with the dictionary values and then delete the dictionary and its components used to create it
	predicting_set['week'].update(pd.Series(week_dict))

	predicting_set['vegasline'].replace('Pick','0',inplace=True)

	del week_dict, start_date, end_date, date_val, week_val

	# Manually call the names of the columns from the scraped data
	home_columns = ['home_four','home_oyds','home_pass','home_pens','home_poss','home_rush','home_sack','home_score','home_team','home_third','home_turn','hpens_yds','hsack_yds','hpass_tot','hrush_att','home_pass_yards','home_pass_attempts']
	away_columns = ['apens_yds','asack_yds','away_four','away_oyds','away_pass','away_pens','away_poss','away_rush','away_sack','away_score','away_team','away_third','away_turn','apass_tot','arush_att','away_pass_yards','away_pass_attempts']
	# Create a mapping to combine home and away columns
	home_cols = {'game week': 'game week', 'home_four': 'fourth down', 'home_oyds': 'total yards', 'home_pass': 'pass yards', 'home_pens': 'penalties', 'home_poss': 'possession', 'home_rush': 'rush yards',
				 'home_sack': 'sacks', 'home_score': 'score', 'home_team': 'team', 'home_third': 'third down', 'home_turn': 'turnovers', 'hpens_yds': 'penalty yards', 'hsack_yds': 'sack yards',
				 'vegasline': 'spread', 'overunder': 'total score', 'hrush_att': 'rush_attempts', 'home_pass_yards': 'pass_yards', 'home_pass_attempts': 'pass_attempts'}
	away_cols = {'game week': 'game week', 'away_four': 'fourth down', 'away_oyds': 'total yards', 'away_pass': 'pass yards', 'away_pens': 'penalties', 'away_poss': 'possession', 'away_rush': 'rush yards',
				 'away_sack': 'sacks', 'away_score': 'score', 'away_team': 'team', 'away_third': 'third down', 'away_turn': 'turnovers', 'apens_yds': 'penalty yards', 'asack_yds': 'sack yards',
				 'vegasline': 'spread', 'overunder': 'total score', 'arush_att': 'rush_attempts', 'away_pass_yards': 'pass_yards', 'away_pass_attempts': 'pass_attempts'}
	# Create only home and away dataframes
	away = predicting_set.drop(home_columns,axis=1)
	home = predicting_set.drop(away_columns,axis=1)

	away.drop('away_pass',axis=1,inplace=True)
	away.drop('apass_tot',axis=1,inplace=True)
	home.drop('home_pass',axis=1,inplace=True)
	home.drop('hpass_tot',axis=1,inplace=True)

	# Create home and away scores which will be used to compare to the predicted value
	away_score = predicting_set[['away_score','week']]
	home_score = predicting_set[['home_score','week']]
	# Remove all games not included in the prediction
	away_score = away_score[away_score.week >= 4]
	home_score = home_score[home_score.week >= 4]
	# Drop the 'week' column as it is no longer needed
	away_score.drop('week',axis=1,inplace=True)
	home_score.drop('week',axis=1,inplace=True)

	# Rename the columns in these dataframes removing the home and away modifier according to the mapping
	away.rename(columns=away_cols,inplace=True)
	home.rename(columns=home_cols,inplace=True)

	# Sort the rows chronologically
	away.sort_index(axis=1,inplace=True)
	home.sort_index(axis=1,inplace=True)
	away_score.sort_index(inplace=True)
	home_score.sort_index(inplace=True)

	score = home_score['home_score'] - away_score['away_score']

	# Pull the actual spreads from the scraped data
	spreads = home[home['week'] >= 4]
	# If the home team is the team listed in the spread then the spread will not change, otherwise it will be multiplied by -1
	home_spread = np.where(spreads['team'].str.split() == spreads['spread'].str.split().str[:-1],1,-1)
	# Extract the spread and convert to numeric while simultaneously multiplying by the above multiplier
	spreads = pd.to_numeric(spreads['spread'].str.split().str[-1]) * home_spread

	home.drop(['spread','total score'],axis=1,inplace=True)
	away.drop(['spread','total score'],axis=1,inplace=True)

	# Group both home and away stats into one dataframe
	total_set = home.append(away)

	# Create a list of the 32 teams
	team_list = total_set['team'].unique()

	# This loop will pull in all data and calculate running averages
	for week in range(4,22):
		# Initialize a "temporary" dataframe for each loop
		weekly_stats = pd.DataFrame(columns=total_set.columns.values)
		# Iterate through each team in the list
		for team in team_list:
			# Pull data only for the team being addressed in each iteration
			mask_team = total_set[total_set['team'] == team]
			# Pull data only for the weeks of interest
			mask_week = mask_team[mask_team['week'] < week]
			# Append the "temporary" dataframe with this pulled data
			weekly_stats = weekly_stats.append(mask_week)

		# Calculate the mean of all data for each team
		weekly_stats = weekly_stats.groupby('team').mean()
		# Reset the week value to equal the week that will be used (for example, week 5 will have all averaged data from weeks 1-4)
		weekly_stats['week'] = week

		# The "total_stats" set is created from weekly_stats in order to have the same column values, and only needs to be done for the first week
		if week == 4:
			total_stats = weekly_stats
		# For all other weeks simply append the "temporary" values
		else:
			total_stats = total_stats.append(weekly_stats)

	# Create the needed columns for the eventual prediction
	matchup_columns = ['week', 'home_team', 'away_team', 'rush_attempt_diff', 'turn_diff', 'yards_diff', 'third_diff', 'sack_diff', 'sack_ydiff', 'poss_diff', 'p_attempt_diff']

	# Create the DataFrame and pull in the 'week' 'home_team' and 'away_team columns
	matchups = pd.DataFrame(columns = matchup_columns)
	matchups[['week','home_team','away_team']] = predicting_set[['week','home_team','away_team']]

	# Remove any results from the first four weeks
	matchups = matchups[matchups.week >= 4]

	# Create the actual differential values using averages from each week
	for row in range(len(matchups)):
		h_team = matchups.iloc[row]['home_team']
		a_team = matchups.iloc[row]['away_team']
		week = matchups.iloc[row]['week']

		# rush_attempt_diff
		h_rush_att = total_stats[((total_stats.index.values == h_team) & (total_stats.week == week))]['rush_attempts'].values[0]
		a_rush_att = total_stats[((total_stats.index.values == a_team) & (total_stats.week == week))]['rush_attempts'].values[0]
		matchups.ix[row,'rush_attempt_diff'] = h_rush_att - a_rush_att

		# turn_diff
		h_turn = total_stats[((total_stats.index.values == h_team) & (total_stats.week == week))]['turnovers'].values[0]
		a_turn = total_stats[((total_stats.index.values == a_team) & (total_stats.week == week))]['turnovers'].values[0]
		matchups.ix[row, 'turn_diff'] = h_turn - a_turn

		# yards_diff
		h_yards = total_stats[((total_stats.index.values == h_team) & (total_stats.week == week))]['total yards'].values[0]
		a_yards = total_stats[((total_stats.index.values == a_team) & (total_stats.week == week))]['total yards'].values[0]
		matchups.ix[row, 'yards_diff'] = h_yards - a_yards

		# third_diff
		h_third = total_stats[((total_stats.index.values == h_team) & (total_stats.week == week))]['third down'].values[0]
		a_third = total_stats[((total_stats.index.values == a_team) & (total_stats.week == week))]['third down'].values[0]
		matchups.ix[row, 'third_diff'] = h_third - a_third

		# sack_diff
		h_sack = total_stats[((total_stats.index.values == h_team) & (total_stats.week == week))]['sacks'].values[0]
		a_sack = total_stats[((total_stats.index.values == a_team) & (total_stats.week == week))]['sacks'].values[0]
		matchups.ix[row, 'sack_diff'] = h_sack - a_sack

		#sack_ydiff
		h_sack_yds = total_stats[((total_stats.index.values == h_team) & (total_stats.week == week))]['sack yards'].values[0]
		a_sack_yds = total_stats[((total_stats.index.values == a_team) & (total_stats.week == week))]['sack yards'].values[0]
		matchups.ix[row, 'sack_ydiff'] = h_sack_yds - a_sack_yds

		# poss_diff
		h_poss = total_stats[((total_stats.index.values == h_team) & (total_stats.week == week))]['possession'].values[0]
		a_poss = total_stats[((total_stats.index.values == a_team) & (total_stats.week == week))]['possession'].values[0]
		matchups.ix[row, 'poss_diff'] = h_poss - a_poss

		# p_attempt_diff
		h_pass_att = total_stats[((total_stats.index.values == h_team) & (total_stats.week == week))]['pass_attempts'].values[0]
		a_pass_att = total_stats[((total_stats.index.values == a_team) & (total_stats.week == week))]['pass_attempts'].values[0]
		matchups.ix[row, 'p_attempt_diff'] = h_pass_att - a_pass_att

	return matchups, spreads, score

def model_dev(train_set,matchups,spreads):

	""" Create the testing set for the algo creation """
	# Create a sample set to pass into the machine learning algorithm
	X = train_set[['rush_attempt_diff', 'turn_diff', 'yards_diff', 'third_diff', 'sack_diff', 'sack_ydiff', 'poss_diff', 'p_attempt_diff']].copy()
	# X = df[['poss_diff', 'third_diff', 'turn_diff', 'pass_diff', 'rush_diff']].copy()

	# Create results vector (a home win = 1, a home loss or tie = 0)
	train_set.rename(columns={'result_spread':'class'},inplace=True)
	y = train_set['class']#np.array(np.where(df['home_score'] > df['away_score'], 1, 0))

	""" Train, test, and predict the algorithm """
	# Scale the sample data
	scaler = preprocessing.StandardScaler().fit(X)
	X = scaler.transform(X)

	# Delete the dataframe to clear memory
	# del train_set

	# Split out training and testing data sets
	X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.25,random_state=0)

	# alphas = [0.1, 0.3, 0.9, 1.0, 1.3, 1.9, 2.0, 2.3, 2.9]
	# for alpha in alphas:
	# 	reg = linear_model.Ridge(alpha = alpha)
	# 	reg.fit(X_train,y_train)
	# 	print 'alpha = ',alpha,', score = ',reg.score(X_test,y_test)
	# input()
	pipeline_optimizer = TPOTRegressor(generations = 5, population_size = 10, random_state = 42, cv = 5, verbosity = 2, n_jobs = 3)#, scoring = 'f1')
	pipeline_optimizer.fit(X_train,y_train)
	print pipeline_optimizer.score(X_test,y_test)
	pipeline_optimizer.export('NFL_ML_TPOT_Regressor.py')

	# Remove the 'week' 'home_team' and 'away_team' columns from matchups as they are not used in the algorithm
	matchups.drop(['week', 'home_team', 'away_team'], axis=1, inplace=True)


	"""
	for feat in range(1,len(matchups.columns)):
		for c in C_vec:
			# Create the classifier and check the score
			# clf = LogisticRegression()
			clf = linear_model.LogisticRegression(C=c,random_state=42)
			selector = RFE(clf)
			selector = selector.fit(X_train,y_train)

			# Calculate probabilities using the predict_proba method for logistic regression
			probabilities = selector.predict_proba(scaler.transform(matchups))

			# Vectorize the spread_conversion function and apply the function to the probabilities result vector
			vfunc = np.vectorize(spread_conversion)
			predicted_spreads = np.apply_along_axis(vfunc,0,probabilities[:,0])

			# If the actual line for the home team is lower than the predicted line then you would take the away team, otherwise take the home team
			bet_vector = np.array(np.where(predicted_spreads > spreads,0,1))

			# Create the actual result vector where a tie counts as a loss for the home team
			game_result = np.array(np.where(home_score.ix[:,0] + predicted_spreads[:] > away_score.ix[:,0], 1, 0))

			# Check to see where the bet_vector equals the actual game result with the spread included
			result = np.array(np.where(bet_vector == game_result,1,0))

			prob_result = float(np.sum(result)) / len(result)

			# print 'Number of features =', feat, 'C =',c,'  Percent correct =',prob_result

			if prob_result > prob_val:
				prob_val = prob_result
				C_val = c
				feat_val = feat

	print 'Score =',selector.score(X_test,y_test)
	# print prob_val, C_val, feat

	clf = linear_model.LogisticRegression(C=C_val,random_state=42)
	clf = clf.fit(X_train,y_train)
	probabilities = clf.predict_proba(scaler.transform(matchups))
	vfunc = np.vectorize(spread_conversion)
	predicted_spreads = np.apply_along_axis(vfunc,0,probabilities[:,0])
	"""

	# predicted_spreads = pd.DataFrame(columns = ['game_date','results'])
	# predicted_spreads['game_date'] = train_set['game_date']
	predicted_spreads = pd.DataFrame(pipeline_optimizer.predict(scaler.transform(matchups)),columns=['results'])
	predicted_spreads = predicted_spreads.set_index(spreads.index)
	print predicted_spreads.head(20)
	print spreads.head(20)
	input()









	""" This is throwing a comparison error that you need to fix if you want to compare in the program """
	# bet_vector = np.array(np.where(predicted_spreads > spreads,0,1))
	# print spreads
	# print predicted_spreads
	# print bet_vector
	return predicted_spreads


def main():
	# Read in csv table
	df = pd.read_csv('nflscraper/command_results_v2.csv')
	# df = pd.read_csv('nflscraper/command_results_v2 (copy).csv')

	df = clean_data(df)

	""" Set date after which you don't want to use data to train classifier """
	cutoff_date = datetime.date(2016, 8, 1)

	train_set = training_set(df,cutoff_date)

	matchups, spreads, scores = prediction_set(df,cutoff_date)

	predicted_spreads = model_dev(train_set,matchups,spreads)
	scores.to_csv('scores.csv',sep=',')
	spreads.to_csv('spreads.csv',sep=',')
	predicted_spreads.to_csv('predicted_spreads.csv',sep=',')
	input()



if __name__ == '__main__':
	main()