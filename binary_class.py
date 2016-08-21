import pandas as pd
import datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LogisticRegression

# Read in csv table
df = pd.read_csv('nflscraper/command_results.csv')

# Change the string date data in df to datetime format
df['game_date'] = pd.to_datetime(df['game_date'],format='%Y-%m-%d')

# Set date after which you don't want to use data to train classifier
cutoff_date = datetime.date(2015, 8, 1)
"""Before this line actually takes place you'll want to assign the predicting set elsewhere"""
df = df[df.game_date < cutoff_date]

# Fill NaNs with outlier values
df.fillna(-999999, inplace=True)

# Calculate home yardage differential
# df['yard_diff'] = df['home_oyds'] - df['away_oyds']

# Create home time of possession differential
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

# Scale the sample data
X = preprocessing.scale(X)

# Create results vector (a home win = 1, a home loss or tie = 0)
y = np.array(np.where(df['home_score'] > df['away_score'], 1, 0))

# Split out training and testing data sets
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.25)

# Create the classifier and check the score
clf = LogisticRegression()
# clf = svm.SVC()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test,y_test)

print accuracy

"""

Still need to see if you can calculate precision and recall from this data

"""