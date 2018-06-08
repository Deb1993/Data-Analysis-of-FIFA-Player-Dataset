def loader():
	'''
	Returns Features, labels and list of feature names

	Loads data from various players.csv and outputs the data in the above numpy array format

	'''
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
	'''
	Returns pandas dataframe

	Loads data from various players.csv and outputs numpy dataframe

	'''
	import pandas as pd
	#read file
	df1 = pd.read_csv('players11.csv')
	df2 = pd.read_csv('players12.csv')
	df3 = pd.read_csv('players14.csv')

	frames = [df1, df2]
	df = pd.concat(frames,ignore_index=True)
	
	df = df[(df['club_pos']!='SUB') & (df['club_pos']!='RES')]
	df= df.reset_index()
	return df

def position(name,df):
	'''
	name : name should be a string, name of the player whose index should be returned
	df   : df should be a panda data frame, dataframe from which the index of the player should be returned 
	'''
	assert isinstance(name,str)
	for i in range(len(df['full_name'])):
		if (df['full_name'][i]):
			if df['full_name'][i]==name:
				return i

def playerpos(player,df,clf,Featuresnp):
	'''
	player: player should be a string, name of the player whose position should be returned
	df:     df should be a panda data frame, dataframe from which the index of the player should be returned 
	clf:    clf is the SVM classifier model from sklearn
	Featuresnp : numpy array of features of all players
	'''
	import numpy as np
	assert isinstance(player,str)
	assert isinstance(Featuresnp,np.ndarray)
	pos=position(player,df)
	result= clf.predict(Featuresnp[pos:pos+1])
	if result[0]==3:
		print("He should play as a Striker")
	elif result[0]==2:
		print("He should play as a Midfielder")
	elif result[0]==1:
		print("He should play as a Defender")
	elif result[0]==0:
		print("He should play as a Goalkeeper")
