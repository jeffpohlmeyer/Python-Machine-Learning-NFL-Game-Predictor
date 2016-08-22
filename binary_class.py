import pandas as pd
import datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm

# Read in csv table
df = pd.read_csv('nflscraper/command_results.csv')

# Change the string date data in df to datetime format
df['game_date'] = pd.to_datetime(df['game_date'],format='%Y-%m-%d')

# Set date after which you don't want to use data to train classifier
cutoff_date = datetime.date(2015, 8, 1)

# Create dataset to be used for prediction
predicting_set = df[df.game_date >= cutoff_date].set_index('game_date')
predicting_set = predicting_set.sort_index()    # Sort ascending by index
predicting_set['week'] = np.nan # Create column indicating which game week the team is in

# Populating the week column of predicting_set
start_date = predicting_set.index.min().date()  # Find the first game date
end_date = predicting_set.index.max().date()    # Find the last game date
date_val = start_date                           # date_val is the value used to cycle through the datetime objects
week_val = 1                                    # week_val is the iterating value that tracks the "current" week
week_dict={}                                    # Create an empty dictionary
# Cycle through the number of days
for _ in range((end_date - start_date).days):
    if date_val.weekday() == 1:                 # Check if the weekday value is a Tuesday
        week_val += 1                           # Increment since I'm considering Tuesday to start a new week
    week_dict[date_val] = week_val              # Update the dictionary value for the "current" date
    date_val += datetime.timedelta(days=1)      # Increment date_val by one day
week_dict[datetime.date(2016,2,7)] = 21         # This is manually done since there are two weeks between conf championships and the final game

# Update the week column in the dataframe with the dictionary values and then delete the dictionary and its components used to create it
predicting_set['week'].update(pd.Series(week_dict))
del week_dict, start_date, end_date, date_val, week_val
# print predicting_set.head()
# print predicting_set.tail()

home_columns = ['home_four','home_oyds','home_pass','home_pens','home_poss','home_rush','home_sack','home_score','home_team','home_third','home_turn','hpens_yds','hsack_yds','vegasline','overunder']
away_columns = ['apens_yds','asack_yds','away_four','away_oyds','away_pass','away_pens','away_poss','away_rush','away_sack','away_score','away_team','away_third','away_turn','vegasline','overunder']
home_cols = {'game week':'game week','home_four':'fourth down','home_oyds':'total yards','home_pass':'pass yards','home_pens':'penalties','home_poss':'possession','home_rush':'rush yards',
             'home_sack':'sacks','home_score':'score','home_team':'team','home_third':'third down','home_turn':'turnovers','hpens_yds':'penalties','hsack_yds':'sack yards'}
away_cols = {'game week': 'game week', 'away_four': 'fourth down', 'away_oyds': 'total yards', 'away_pass': 'pass yards', 'away_pens': 'penalties', 'away_poss': 'possession', 'away_rush': 'rush yards',
             'away_sack': 'sacks', 'away_score': 'score', 'away_team': 'team', 'away_third': 'third down', 'away_turn': 'turnovers', 'apens_yds': 'penalties', 'asack_yds': 'sack yards'}
away = predicting_set.drop(home_columns,axis=1)
home = predicting_set.drop(away_columns,axis=1)

away.rename(columns=away_cols,inplace=True)
home.rename(columns=home_cols,inplace=True)

# print home.info()
# print away.info()


# surveys_df = pd.read_csv("https://ndownloader.figshare.com/files/2292172",
#                          keep_default_na=False, na_values=[""])
# species_df = pd.read_csv('https://ndownloader.figshare.com/files/3299483',
#                          keep_default_na=False, na_values=[""])
# # read in first 10 lines of surveys table
# survey_sub = surveys_df.head(10)
# # grab the last 10 rows (minus the last one)
# survey_sub_last10 = surveys_df[-11:-1]
# #reset the index values to the second dataframe appends properly
# print survey_sub
# print survey_sub_last10
#
# survey_sub_last10=survey_sub_last10.reset_index(drop=True)
# # drop=True option avoids adding new index column with old index values
# print survey_sub_last10
#
# # stack the DataFrames on top of each other
# vertical_stack = pd.concat([survey_sub, survey_sub_last10], axis=0)
#
# # place the DataFrames side by side
# horizontal_stack = pd.concat([survey_sub, survey_sub_last10], axis=1)













# print list(away.columns.values)
# total_set = pd.concat([away,home],keys=['away','home'])#,axis=0,ignore_index=True)#,keys = list(away.columns.values))

# team_list = predicting_set['home_team'].unique()
#
# column_vals = ['game week','total yards','pass yards','rush yards','penalties','turnovers','third down','fourth down','sacks','sack yards','possession']
#
# for team in team_list:
#     team = pd.DataFrame(columns = column_vals)
#
# New_England = pd.DataFrame(columns = column_vals)
#
# print team_list[23].tail()


total_set=away.append(home)

















# df = df[df.game_date < cutoff_date]
#
# # Fill NaNs with outlier values
# df.fillna(-999999, inplace=True)
#
# # Calculate home yardage differential
# # df['yard_diff'] = df['home_oyds'] - df['away_oyds']
#
# # Create home time of possession differential
# # Convert objects to datetime values
# df['home_poss'] = pd.to_datetime(df['home_poss'],format='%M:%S')
# df['away_poss'] = pd.to_datetime(df['away_poss'],format='%M:%S')
#
# # Convert datetime values to fractions of an hour
# df['home_poss'] = df['home_poss'].dt.minute / 60.0 + df['home_poss'].dt.second / 3600.0
# df['away_poss'] = df['away_poss'].dt.minute / 60.0 + df['away_poss'].dt.second / 3600.0
#
# # Find total possession time (only really matters because games can go to overtime)
# # And re-weight time of possession
# df['total_poss'] = df['home_poss'] + df['away_poss']
# df['home_poss'] = df['home_poss'] / df['total_poss']
# df['away_poss'] = df['away_poss'] / df['total_poss']
# df.drop('total_poss',axis = 1, inplace=True)    # Delete total possession column as it's no longer needed
#
# # Calculate time of possession differential
# df['poss_diff'] = df['home_poss'] - df['away_poss']
#
# # Calculate third down percentage differential
# df['third_diff'] = df['home_third'] - df['away_third']
#
# # Calculate turnover differential
# df['turn_diff'] = df['home_turn'] - df['away_turn']
#
# # Calculate sack quantity differential
# df['sack_diff'] = df['home_sack'] - df['away_sack']
#
# # Calculate sack yards differential
# df['sack_ydiff'] = df['hsack_yds'] - df['asack_yds']
#
# # Calculate penalty yards differential
# df['pens_diff'] = df['hpens_yds'] - df['apens_yds']
#
# # Calculate passing yardage differential
# df['pass_diff'] = df['home_pass'] - df['away_pass']
#
# # Calculate rushing yardage differential
# df['rush_diff'] = df['home_rush'] - df['away_rush']
#
# # Create a sample set to pass into the machine learning algorithm
# X = df[['sack_diff', 'sack_ydiff', 'pens_diff', 'poss_diff', 'third_diff', 'turn_diff', 'pass_diff', 'rush_diff']].copy()
# # X = df[['poss_diff', 'third_diff', 'turn_diff', 'pass_diff', 'rush_diff']].copy()
#
# # Scale the sample data
# X = preprocessing.scale(X)
#
# # Create results vector (a home win = 1, a home loss or tie = 0)
# y = np.array(np.where(df['home_score'] > df['away_score'], 1, 0))
#
# # Delete the dataframe to clear memory
# del df
#
# # Split out training and testing data sets
# X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)
#
# # Create the classifier and check the score
# # clf = LogisticRegression()
# clf = svm.LinearSVC()
# clf.fit(X_train, y_train)
# accuracy = clf.score(X_test,y_test)
#
# print accuracy
#
# """
#
# Still need to see if you can calculate precision and recall from this data
#
# """