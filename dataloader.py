def loader():
	import pandas as pd
	import numpy as np


	df1 = pd.read_csv('players11.csv')
	df2 = pd.read_csv('players12.csv')
	df3 = pd.read_csv('players14.csv')

	frames = [df1]
	df = pd.concat(frames,ignore_index=True)



	#Group parameters
	df = df[(df['club_pos']!='SUB') & (df['club_pos']!='RES')]

	bar_df = df
	mapp = {'GK': 0,'CB': 1,'LCB': 1,'RCB': 1,'LB':1,'RB': 1,'RWB': 1,'LWB': 1,'CM' : 2,'RM' : 2,'LDM': 2,'LAM': 2,'RDM': 2,'RAM': 2,'RCM' : 2,'LCM' : 2,'CDM': 2,'CAM': 2,'LM' : 2,'RM' : 2,'LW': 3,'RW': 3,'LF': 3, 'CF': 3, 'RF': 3, 'RS': 3,'ST': 3,'LS' : 3}
	Ydata =  df['club_pos'].map(mapp)
	Features = df[['SlidingTackle','StandingTackle', 'LongPassing', 'ShortPassing','Acceleration','SprintSpeed','Agility', 'Balance', 'BallControl', 'Aggression','Composure', 'Crossing', 'Curve', 'Dribbling', 'FKAccuracy', 'Finishing', 'GKDiving','GKHandling','GKKicking','GKPositioning','GKReflexes', 'HeadingAccuracy', 'Interceptions', 'Jumping', 'LongShots', 'Marking', 'Penalties', 'Positioning', 'ShotPower', 'Stamina', 'Strength', 'Vision', 'Volleys','weight','height']]
	Featuresnp=Features.values
	featurenames=['SlidingTackle','StandingTackle', 'LongPassing', 'ShortPassing','Acceleration','SprintSpeed','Agility', 'Balance', 'BallControl', 'Aggression','Composure', 'Crossing', 'Curve', 'Dribbling', 'FKAccuracy', 'Finishing', 'GKDiving','GKHandling','GKKicking','GKPositioning','GKReflexes', 'HeadingAccuracy', 'Interceptions', 'Jumping', 'LongShots', 'Marking', 'Penalties', 'Positioning', 'ShotPower', 'Stamina', 'Strength', 'Vision', 'Volleys','weight','height']
	Ydata=Ydata.values
	return Featuresnp, Ydata,featurenames

def loader_mlmodel():
	import pandas as pd
	#read file
	df1 = pd.read_csv('players11.csv')
	df2 = pd.read_csv('players12.csv')
	df3 = pd.read_csv('players14.csv')

	frames = [df1, df2]
	df = pd.concat(frames,ignore_index=True)

	df = df[(df['club_pos']!='SUB') & (df['club_pos']!='RES')]
	return df


