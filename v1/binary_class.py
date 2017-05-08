import pandas as pd
import datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm, linear_model
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve

"""Found at http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html"""
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# Read in csv table
df = pd.read_csv('nflscraper/command_results.csv')

# Calculate offensive efficiency (pass yards per attempt and rush yards per carry)
home_pass_stats = pd.DataFrame(df['hpass_tot'].str.split('-').tolist(),columns="completions attempts yards touchdowns interceptions".split())
away_pass_stats = pd.DataFrame(df['apass_tot'].str.split('-').tolist(),columns="completions attempts yards touchdowns interceptions".split())
away_pass_stats.replace('neg7','-7',inplace=True)
home_pass_stats['yards'] = home_pass_stats['yards'].astype(float)
home_pass_stats['attempts'] = home_pass_stats['attempts'].astype(float)
away_pass_stats['yards'] = away_pass_stats.yards.astype(float)
away_pass_stats['attempts'] = away_pass_stats.attempts.astype(float)

df['home_pass'] = home_pass_stats['yards'] / home_pass_stats['attempts']
df['away_pass'] = away_pass_stats['yards'] / away_pass_stats['attempts']
del home_pass_stats, away_pass_stats

df['home_rush'].astype(float)
df['hrush_att'].astype(float)
df['away_rush'].astype(float)
df['arush_att'].astype(float)
df['home_rush'] = df['home_rush'] / df['hrush_att']
df['away_rush'] = df['away_rush'] / df['arush_att']

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
df.drop('total_poss',axis = 1, inplace=True)    # Delete total possession column as it's no longer needed


""" Set date after which you don't want to use data to train classifier """
cutoff_date = datetime.date(2015, 8, 1)

""" Creating prediction dataset """
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

# Manually call the names of the columns from the scraped data
home_columns = ['home_four','home_oyds','home_pass','home_pens','home_poss','home_rush','home_sack','home_score','home_team','home_third','home_turn','hpens_yds','hsack_yds','vegasline','overunder','hpass_tot','hrush_att']
away_columns = ['apens_yds','asack_yds','away_four','away_oyds','away_pass','away_pens','away_poss','away_rush','away_sack','away_score','away_team','away_third','away_turn','vegasline','overunder','apass_tot','arush_att']
# Create a mapping to combine home and away columns
home_cols = {'game week': 'game week', 'home_four': 'fourth down', 'home_oyds': 'total yards', 'home_pass': 'pass yards', 'home_pens': 'penalties', 'home_poss': 'possession', 'home_rush': 'rush yards',
             'home_sack': 'sacks', 'home_score': 'score', 'home_team': 'team', 'home_third': 'third down', 'home_turn': 'turnovers', 'hpens_yds': 'penalty yards', 'hsack_yds': 'sack yards'}
away_cols = {'game week': 'game week', 'away_four': 'fourth down', 'away_oyds': 'total yards', 'away_pass': 'pass yards', 'away_pens': 'penalties', 'away_poss': 'possession', 'away_rush': 'rush yards',
             'away_sack': 'sacks', 'away_score': 'score', 'away_team': 'team', 'away_third': 'third down', 'away_turn': 'turnovers', 'apens_yds': 'penalty yards', 'asack_yds': 'sack yards'}
# Create only home and away dataframes
away = predicting_set.drop(home_columns,axis=1)
home = predicting_set.drop(away_columns,axis=1)
away.drop('apass_tot',axis=1,inplace=True)
away.drop('arush_att',axis=1,inplace=True)
away.drop('away_oyds',axis=1,inplace=True)
home.drop('hpass_tot',axis=1,inplace=True)
home.drop('hrush_att',axis=1,inplace=True)
home.drop('home_oyds',axis=1,inplace=True)

# Create home and away scores which will be used to compare to the predicted value
away_score = predicting_set[['away_score','week']]
home_score = predicting_set[['home_score','week']]
# Remove all games not included in the prediction
away_score = away_score[away_score.week >= 5]
home_score = home_score[home_score.week >= 5]
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

# Create the actual result vector where a tie counts as a loss for the home team
game_result = np.array(np.where(home_score.ix[:,0] > away_score.ix[:,0], 1, 0))

# Group both home and away stats into one dataframe
total_set = home.append(away)
del home, away
# Create a list of the 32 teams
team_list = total_set['team'].unique()

# This loop will pull in all data and calculate running averages
for week in range(5,22):
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
    if week == 5:
        total_stats = weekly_stats
    # For all other weeks simply append the "temporary" values
    else:
        total_stats = total_stats.append(weekly_stats)

# Create the needed columns for the eventual prediction
matchup_columns = ['week','home_team','away_team', 'sack_diff', 'sack_ydiff', 'pens_diff', 'poss_diff', 'third_diff', 'turn_diff', 'pass_diff', 'rush_diff']
# Create the DataFrame and pull in the 'week' 'home_team' and 'away_team columns
matchups = pd.DataFrame(columns = matchup_columns)
matchups[['week','home_team','away_team']] = predicting_set[['week','home_team','away_team']]

# Remove any results from the first four weeks
matchups = matchups[matchups.week >= 5]

# Create the actual differential values using averages from each week
for row in range(len(matchups)):
    h_team = matchups.iloc[row]['home_team']
    a_team = matchups.iloc[row]['away_team']
    week = matchups.iloc[row]['week']

    # pass_diff
    h_pass = total_stats[((total_stats.index.values == h_team) & (total_stats.week == week))]['pass yards'].values[0]
    a_pass = total_stats[((total_stats.index.values == a_team) & (total_stats.week == week))]['pass yards'].values[0]
    matchups.ix[row,'pass_diff'] = h_pass - a_pass

    # rush_diff
    h_rush = total_stats[((total_stats.index.values == h_team) & (total_stats.week == week))]['rush yards'].values[0]
    a_rush = total_stats[((total_stats.index.values == a_team) & (total_stats.week == week))]['rush yards'].values[0]
    matchups.ix[row, 'rush_diff'] = h_rush - a_rush

    # sack_diff
    h_sack = total_stats[((total_stats.index.values == h_team) & (total_stats.week == week))]['sacks'].values[0]
    a_sack = total_stats[((total_stats.index.values == a_team) & (total_stats.week == week))]['sacks'].values[0]
    matchups.ix[row, 'sack_diff'] = h_sack - a_sack

    #sack_ydiff
    h_sack_yds = total_stats[((total_stats.index.values == h_team) & (total_stats.week == week))]['sack yards'].values[0]
    a_sack_yds = total_stats[((total_stats.index.values == a_team) & (total_stats.week == week))]['sack yards'].values[0]
    matchups.ix[row, 'sack_ydiff'] = h_sack_yds - a_sack_yds

    # pens_diff
    h_pen_yds = total_stats[((total_stats.index.values == h_team) & (total_stats.week == week))]['penalty yards'].values[0]
    a_pen_yds = total_stats[((total_stats.index.values == a_team) & (total_stats.week == week))]['penalty yards'].values[0]
    matchups.ix[row, 'pens_diff'] = h_pen_yds - a_pen_yds

    # poss_diff
    h_poss = total_stats[((total_stats.index.values == h_team) & (total_stats.week == week))]['possession'].values[0]
    a_poss = total_stats[((total_stats.index.values == a_team) & (total_stats.week == week))]['possession'].values[0]
    matchups.ix[row, 'poss_diff'] = h_poss - a_poss

    # third_diff
    h_third = total_stats[((total_stats.index.values == h_team) & (total_stats.week == week))]['third down'].values[0]
    a_third = total_stats[((total_stats.index.values == a_team) & (total_stats.week == week))]['third down'].values[0]
    matchups.ix[row, 'third_diff'] = h_third - a_third

    # turn_diff
    h_turn = total_stats[((total_stats.index.values == h_team) & (total_stats.week == week))]['turnovers'].values[0]
    a_turn = total_stats[((total_stats.index.values == a_team) & (total_stats.week == week))]['turnovers'].values[0]
    matchups.ix[row, 'turn_diff'] = h_turn - a_turn

""" Create the testing set for the algo creation """
# Remove the predicting set from the dataframe
df = df[df.game_date < cutoff_date]

# Fill NaNs with outlier values
df.fillna(-999999, inplace=True)

# Calculate time of possession differential
df['poss_diff'] = df['home_poss'] - df['away_poss']

# Calculate third down percentage differential
df['third_diff'] = df['home_third'] - df['away_third']

# Calculate turnover differential
df['turn_diff'] = df['home_turn'] - df['away_turn']

# Calculate sack quantity differential
df['sack_diff'] = df['home_sack'] - df['away_sack']

# Calculate sack yards differential
df['sack_ydiff'] = df['hsack_yds'] - df['asack_yds']

# Calculate penalty yards differential
df['pens_diff'] = df['hpens_yds'] - df['apens_yds']

# Calculate passing yardage differential
df['pass_diff'] = df['home_pass'] - df['away_pass']

# Calculate rushing yardage differential
df['rush_diff'] = df['home_rush'] - df['away_rush']

# Create a sample set to pass into the machine learning algorithm
X = df[['sack_diff', 'sack_ydiff', 'pens_diff', 'poss_diff', 'third_diff', 'turn_diff', 'pass_diff', 'rush_diff']].copy()
# X = df[['poss_diff', 'third_diff', 'turn_diff', 'pass_diff', 'rush_diff']].copy()

# Create results vector (a home win = 1, a home loss or tie = 0)
y = np.array(np.where(df['home_score'] > df['away_score'], 1, 0))

title = "Learning Curves (SVM)"#, RBF kernel, $\gamma=0.001$)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=10,
                                   test_size=0.2, random_state=0)
estimator = svm.LinearSVC()
plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)

plt.show()


""" Train, test, and predict the algorithm """
# Scale the sample data
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)

# Delete the dataframe to clear memory
del df

# Split out training and testing data sets
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

# Create the classifier and check the score
# clf = LogisticRegression()
clf = svm.LinearSVC()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test,y_test)
print 'Accuracy:',accuracy

# Remove the 'week' 'home_team' and 'away_team' columns from matchups as they are not used in the algorithm
matchups.drop(['week','home_team','away_team'],axis=1,inplace=True)
prediction_result = clf.predict(scaler.transform(matchups))

prediction_accuracy = np.array(np.where(prediction_result == game_result,1,0))

print 'Prediction:',float(np.sum(prediction_accuracy)) / len(prediction_accuracy)