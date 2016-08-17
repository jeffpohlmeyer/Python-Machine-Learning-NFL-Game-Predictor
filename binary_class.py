import pandas as pd
import datetime

# Read in csv table
df = pd.read_csv('nflscraper/command_results.csv')

# Change the string date data in df to datetime format
df['game_date'] = pd.to_datetime(df['game_date'],format='%Y-%m-%d')

# Set date after which you don't want to use data to train classifier
cutoff_date = datetime.date(2015, 8, 1)
df = df[df.game_date < cutoff_date]

# Calculate home yardage differential
df['yard_diff'] = df['home_oyds'] - df['away_oyds']

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

print df.tail()
