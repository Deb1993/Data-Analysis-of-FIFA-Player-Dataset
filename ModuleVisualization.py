# Libraries
import matplotlib.pyplot as plt
import pandas as pd
import math
from math import pi
import numpy as np
import datetime as DT
import bokeh
from bokeh.plotting import figure, show, output_notebook, output_file, gmap
from bokeh.models import NumeralTickFormatter, ColumnDataSource, GMapOptions, Circle
import re

att = ['Acceleration','Aggression','Agility','Balance','BallControl','Composure','Crossing','Curve','Dribbling','FKAccuracy','Finishing','GKDiving','GKHandling','GKKicking','GKPositioning','GKReflexes','HeadingAccuracy','Interceptions','Jumping','LongPassing','LongShots','Marking','Penalties','Positioning','Reactions','ShortPassing','ShotPower','SlidingTackle','SprintSpeed','Stamina','StandingTackle','Strength','Vision','Volleys','age','club','club_pos','country','dob','full_name','height','image','potential','pref_pos','preff_foot','rating','short_name','value','wage','weight']







def radar_plot(df,Attributes,mapping,outfile,textsize=20):
    
    """
    plot position profiles for each position 
    average of all values
    User-defined characteristics
    :param: df, dataset
    :type: pandas dataframe
    :param: Attributes, list of attributes to be viewed
    :type: list, str
    :param: mapping, dictionary mapping playing position keys to positions
    :type: dict
    :param: outfile, name of output file
    :type: str
    :param: textsize, size of text
    :type: int

    """
    assert isinstance(df, pd.DataFrame),'Input must be a dataframe!'
    assert isinstance(Attributes, list),'Attributes must be in a list!'
    assert all(isinstance(item, str) for item in Attributes), 'Attributes must be strings!'
    assert isinstance(mapping,dict), 'Mapping must be a dictionary!'
    assert isinstance(outfile,str), 'Output filename must be a string!'
    assert isinstance(textsize,int), 'Text size must be an integer!'

    #define dataframe used for radar plots
    df_user_radar = df

    df_user_radar = df_user_radar[(df_user_radar['club_pos']!='SUB')] #don't care about substitution players
    df_user_radar = df_user_radar[(df_user_radar['club_pos']!='RES')] #don't care about reserve players

    #Take wanted parameters
    #set filter: what attributes would you like to see?    
    df_user_radar = df_user_radar[Attributes]

    #group positions by above mapping
    grouped =  df_user_radar.set_index('club_pos').groupby(mapping)
    df_user_radar = grouped.agg([np.nanmean])
    #remove unnecessary titles
    df_user_radar.columns = df_user_radar.columns.droplevel(1)

    #cleanup
    df_user_radar.reset_index(level=0, inplace=True)
    df_user_radar.rename(index=int, columns={"index": "Position"},inplace=True)

    #set data
    size = len(df_user_radar)   
    colors = ['b','r','g','m'] 
    fig = plt.figure(figsize=(20,10),dpi=220)
    for i in xrange(0,size):
        # number of variable
        categories=list(df_user_radar)[1:]
        N = len(categories)

        # We are going to plot the first line of the data frame.
        # But we need to repeat the first value to close the circular graph:
        values=df_user_radar.loc[i].drop('Position').values.flatten().tolist()
        values += values[:1]

        # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]
        #angles = [math.degrees(float(angles[c])) for c in range(len(angles))]
        #angles = xrange(0,360,360/N)


        plots = 111 + 10*len(df_user_radar)+i
        # Initialise the spider plot
        ax = fig.add_subplot(plots, polar=True)
        #ax.set_theta_offset(pi / 2)
        #ax.set_theta_direction(1)

        ax.set_title(df_user_radar['Position'][i]+'\n'+'\n',size=textsize*1.3)
        # Draw one axe per variable + add labels labels yet
        plt.xticks(angles[:-1], categories, color='black', size=textsize)
        ax.tick_params(axis='x', which='major', pad=30)
        # set ticklabels location at 1.3 times the axes' radius
        # Draw ylabels
        plt.yticks(color="grey", size=textsize*0.7)
        plt.ylim(0,100)

        # Plot data
        ax.plot(angles, values, linewidth=1, color=colors[i],linestyle='solid')

        # Fill area
        ax.fill(angles, values, colors[i], alpha=0.1)
    plt.tight_layout() 
    fig.savefig('%s'%outfile, bbox_inches='tight')
    return fig

def radar_plot_single(df,Attributes,mapping,outfile,textsize=20,position=None):
    
    """
    plot position profiles for each position 
    average of all values
    User-defined characteristics
    :param: df, dataset
    :type: pandas dataframe
    :param: Attributes, list of attributes to be viewed
    :type: list, str
    :param: mapping, dictionary mapping playing position keys to positions
    :type: dict
    :param: outfile, name of output file
    :type: str
    :param: textsize, size of text
    :type: int
    :param: position, what position to show
    :type: str

    """
    assert isinstance(df, pd.DataFrame),'Input must be a dataframe!'
    assert isinstance(Attributes, list),'Attributes must be in a list!'
    assert all(isinstance(item, str) for item in Attributes), 'Attributes must be strings!'
    assert isinstance(mapping,dict), 'Mapping must be a dictionary!'
    assert isinstance(outfile,str), 'Output filename must be a string!'
    assert isinstance(textsize,int), 'Text size must be an integer!'
    if position != None:
        assert position in ['Attack','Defense','Midfield','Goalkeeper']
        assert isinstance(position,str), 'Position must be a string!'
    

    #define dataframe used for radar plots
    df_user_radar = df

    df_user_radar = df_user_radar[(df_user_radar['club_pos']!='SUB')] #don't care about substitution players
    df_user_radar = df_user_radar[(df_user_radar['club_pos']!='RES')] #don't care about reserve players

    #Take wanted parameters
    #set filter: what attributes would you like to see?    
    df_user_radar = df_user_radar[Attributes]

    #group positions by position mapping
    grouped =  df_user_radar.set_index('club_pos').groupby(mapping)
    df_user_radar = grouped.agg([np.nanmean])
    #remove unnecessary titles
    df_user_radar.columns = df_user_radar.columns.droplevel(1)

    #cleanup
    df_user_radar.reset_index(level=0, inplace=True)
    df_user_radar.rename(index=int, columns={"index": "Position"},inplace=True)

    #select position to be viewed
    if position != None:
        df_user_radar = df_user_radar[df_user_radar.Position == position]
        df_user_radar.index = range(len(df_user_radar.index))
             
    #set data
    size = len(df_user_radar)    
    fig = plt.figure(figsize=(5,5),dpi=220)
    plots = 111
    colors = ['b','r','g','m']
    if position == 'Attack':
        colors = ['b']
    elif position == 'Defense':
        colors = ['r']
    elif position == 'Midfield':
        colors = ['g']
    elif position == 'Goalkeeper':
        colors = ['m']
        
    
    # Initialise the spider plot
    ax = fig.add_subplot(plots, polar=True)
    for i in xrange(0,size):
        # number of variable
        categories=list(df_user_radar)[1:]
        N = len(categories)

        # We are going to plot the first line of the data frame.
        # But we need to repeat the first value to close the circular graph:
        values=df_user_radar.loc[i].drop('Position').values.flatten().tolist()
        values += values[:1]

        # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]
        #angles = [math.degrees(float(angles[c])) for c in range(len(angles))]
        #angles = xrange(0,360,360/N)



        #ax.set_theta_offset(pi / 2)
        #ax.set_theta_direction(1)

        # Draw one axe per variable + add labels labels yet
        plt.xticks(angles[:-1], categories, color='black', size=textsize)
        ax.tick_params(axis='x', which='major', pad=30)
        # set ticklabels location at 1.3 times the axes' radius
        # Draw ylabels
        plt.yticks(color="grey", size=textsize*0.7)
        plt.ylim(0,100)

        # Plot data
        ax.plot(angles, values, color = colors[i], linewidth=1, linestyle='solid',label=df_user_radar['Position'][i])

        # Fill area
        ax.fill(angles, values, colors[i], alpha=0.1)
    plt.tight_layout() 
    # Add legend
    if position == None:
        ax.legend(bbox_to_anchor=(0,0),fontsize='x-large')
    fig.savefig('%s'%outfile, bbox_inches='tight')
    return plt

def bar_plot(df,x='age',y='wage',color='firebrick'):
    '''
    make bar plot 

    :param: df, dataset
    :type: pandas dataframe
    :param: x, attribute to be considered on x axis
    :type: str
    :param: y, attribute to be considered on y axis
    :type: str
    :param: color, color of bars
    :type: str
    :param: droptop, choose to drop top players
    :type: boolean
    
    '''

    assert isinstance(df, pd.DataFrame),'Input must be a dataframe!'
    assert isinstance(x, str), 'X-axis must be a string!'
    assert isinstance(y, str), 'Y-axis must be a string!'
    assert isinstance(color,str), 'Color must be a string!'
    assert x in att, 'Not an attribute!'
    assert y in att, 'Not an attribute!'
    
    output_notebook()

    df_2b = df

    #group by age
    df_2b = df_2b[[y,x]]
    grouped = df_2b.set_index(x).groupby(x)
    df_2b = grouped.agg([np.nanmean]).reset_index()
    age = df_2b[x]
    wage = df_2b[y]['nanmean']


    #plot
    p = figure(plot_width=700, plot_height=400, title='%s VS %s'%(y,x))
    p.vbar(x=age, width=0.5, bottom=0, top=wage, color=color )
    p.yaxis[0].formatter = NumeralTickFormatter(format="00,000")
    p.xaxis.axis_label = x
    p.yaxis.axis_label = y
    return p

def line_plot(df,x,y1,y2):

    """
    line chart showing 
    Y1 and Y2 vs X

    :param: df, dataset
    :type: pandas dataframe
    :param: x, attribute to be considered on x axis
    :type: str
    :param: y1, attribute to be considered on y axis
    :type: str 
    :param: y2, attribute to be considered on y axis
    :type: str

    """
   
    assert isinstance(df, pd.DataFrame),'Input must be a dataframe!'
    assert isinstance(x, str), 'X-axis must be a string!'
    assert isinstance(y1, str), 'Y1 series must be a string!'
    assert isinstance(y2,str), 'Y2 series must be a string!'
    assert x in att, 'Not an attribute!'
    assert y1 in att, 'Not an attribute!'
    assert y2 in att, 'Not an attribute!'

    
    output_notebook()   
        
    df_3e = df

    #filter
    df_3e = df_3e[[y1,y2,x]]
    #group by age
    grouped = df_3e.set_index(x).groupby(x)
    df_3e = grouped.agg([np.nanmean])

    #plot
    p = figure(plot_width=800, plot_height=400,title='%s & %s VS %s'%(y1,y2,x))
    p.line(df_3e.index, df_3e[y1]['nanmean'],line_width=4,color='firebrick',legend=y1)
    p.line(df_3e.index, df_3e[y2]['nanmean'],line_width=4,color='navy',legend=y2)
    p.xaxis.axis_label = '%s'%x
    p.yaxis.axis_label = 'Magnitude'

    return p

def line_plot_pos(df,x,y,mapping):
    """
    line chart for positions

    Showing y vs x per position

    :param: df, dataset
    :type: pandas dataframe
    :param: x, attribute to be considered on x axis
    :type: str
    :param: y, attribute to be considered on y axis
    :type: str
    :param: mapping, dictionary mapping playing position keys to positions
    :type: dict

    """

    assert isinstance(df, pd.DataFrame),'Input must be a dataframe!'
    assert isinstance(x, str), 'X-axis must be a string!'
    assert isinstance(y, str), 'Y series must be a string!'
    assert isinstance(mapping,dict), 'Mapping must be a dictionary!'
    assert x in att, 'Not an attribute!'
    assert y in att, 'Not an attribute!'
    
    output_notebook()

    df_3d = df

    #filter 
    df_3d = df_3d[[y, 'club_pos','full_name',x]]


    #group positions by position mapping
    grouped_pos = df_3d.set_index('club_pos').groupby(mapping)



    #Attack
    y1 = grouped_pos.get_group('Attack')
    grouped =  y1.set_index(x).groupby(x)
    y1 = grouped.agg([np.mean])
    #Defense
    y2 = grouped_pos.get_group('Defense')
    grouped =  y2.set_index(x).groupby(x)
    y2 = grouped.agg([np.mean])
    #Midfield
    y3 = grouped_pos.get_group('Midfield')
    grouped =  y3.set_index(x).groupby(x)
    y3 = grouped.agg([np.mean])

    #plot
    p = figure(plot_width=800, plot_height=400,title='%s VS %s'%(y,x))
    p.line(y1.index, y1[y]['mean'],line_width=4,color='firebrick',legend="Attack")
    p.line(y2.index, y2[y]['mean'],line_width=4,color='navy',legend="Defense")
    p.line(y3.index, y3[y]['mean'],line_width=4,color='olive',legend="Midfield")
    p.yaxis[0].formatter = NumeralTickFormatter(format="000,000")
    p.xaxis.axis_label = x
    p.yaxis.axis_label = y

    return p

def scatter_plot(df,x,y):
    """
    scatter plot showing y vs x
    showing all players and mean

    :param: df, dataset
    :type: pandas dataframe
    :param: x, attribute to be considered on x axis
    :type: str
    :param: y, attribute to be considered on y axis
    :type: str
    """

    assert isinstance(df, pd.DataFrame),'Input must be a dataframe!'
    assert isinstance(x, str), 'X-axis must be a string!'
    assert isinstance(y, str), 'Y1 series must be a string!'
    assert x in att, 'Not an attribute!'
    assert y in att, 'Not an attribute!'
    
    
    output_notebook()

    df_4d = df
    df_gk = df
    #group by rating
    df_4d = df_4d[[x,y]]
    grouped = df_4d.set_index(x).groupby(x)
    df_4d = grouped.agg([np.nanmean])

    #plot
    p = figure(title="%s VS %s"%(y,x),plot_width=800, plot_height=400)
    p.background_fill_color = "white"
    p.scatter(df_gk[x],df_gk[y], marker='o', size=15,
                color="orange", alpha=0.8)
    p.scatter(df_4d.index,df_4d[y]['nanmean'], marker='o', size=15,
                color="red", alpha=0.9)
    p.xaxis.axis_label = '%s'%x
    p.yaxis.axis_label = '%s'%y
    return p

def bubble_scatter_plot(df,x,y,z,norm=1.6):

    """
    bubble plot showing y vs x
    with z as the size of bubbles

    :param: df, dataset
    :type: pandas dataframe
    :param: x, attribute to be on x axis
    :type: str
    :param: y, attribute to be on y axis
    :type: str
    :param: z, attribute to be size of bubbles
    :type: str

    """
    assert isinstance(df, pd.DataFrame),'Input must be a dataframe!'
    assert isinstance(x, str), 'X-axis must be a string!'
    assert isinstance(y, str), 'Y-axis must be a string!'
    assert isinstance(z, str), 'Z-axis must be a string!'
    assert x in att, 'Not an attribute!'
    assert y in att, 'Not an attribute!'
    assert z in att, 'Not an attribute!'
    assert isinstance(norm,float)
    
    
    output_notebook()

    df_4e = df
    #filter
    df_4e = df_4e[[x,y,z]]

    #group by rating
    grouped = df_4e.set_index(x).groupby(x)
    df_4e = grouped.agg([np.nanmean])


    #plot
    p = figure(title="%s vs %s vs %s"%(z,y,x),plot_width=800, plot_height=400)
    p.grid.grid_line_color = None
    p.background_fill_color = "white"

    #circle = Circle(x='x', y='y', size='size', fill_color='orange', fill_alpha=0.4, line_color=None)
    #p.add_glyph(source, circle)
    p.scatter(df[x],df[y],marker='o', size=(df[z]-18)*norm,
                color="orange", alpha=0.8)
    p.scatter(df_4e.index,df_4e[y]['nanmean'], marker='o', size=15,
                color="red", alpha=0.9)
    p.xaxis.axis_label = '%s'%x
    p.yaxis.axis_label = '%s'%y

    #legend
    p
    return p


def world_map_plot(df,norm,outfile,club=True,color='blue',alpha=0.8):

    """
    plot world map given parameters
    :param: df, dataset 
    :type: pandas Dataframe
    :param: club, choose to calculate via club or Country of origin
    :type: boolean
    :param: norm, size of bubbles
    :type: float
    :param: outfile, name of output image
    :type: str
    
    """
    assert isinstance(df, pd.DataFrame),'Input must be a dataframe!'
    assert isinstance(outfile, str), 'Output file must be a string!'
    assert isinstance(club, bool), 'Club Money or Country Of Origin? True or False'
    assert isinstance(color, str), 'Color must be a string!'
    assert isinstance(alpha,float), 'Alpha must be float'
    assert isinstance(norm,float), 'Size must be float'
    
    
    
    
    
    map_options = GMapOptions(lat=18.85605, lng=11.34108, map_type="roadmap", zoom=1) 

    # For GMaps to function, Google requires you obtain and enable an API key:
    #
    #     https://developers.google.com/maps/documentation/javascript/get-api-key
    #
    # Replace the value below with your personal API key:

    p = gmap('AIzaSyBQrkR6fnk-LXapwd5dRtuVpGNLNl3gzXQ', map_options)


    #pick dataset
    df_6a = df[['country','wage']]
    #clean up country column
    df_6aa = pd.DataFrame(df_6a['country'].str.split(', ').values.tolist())
    df_6a = pd.concat([df_6a['wage'], df_6aa], axis=1)

    #Make country club in first column, Country of origin second
    df_6a[1], df_6a[0] = np.where(df_6a[1].isnull(), [df_6a[0], df_6a[1]], [df_6a[1], df_6a[0] ])

    #If doing country club, = 0. Origin = 1
    if club == True:
        df_6a = df_6a[['wage',0]]
        df_6a.drop(df_6a[df_6a[0].isnull()].index, inplace=True)
        df_6a[0] = df_6a[0].apply(lambda x: re.sub("[^a-zA-Z ]+", "", x))
        df_6a.rename(index=int, columns={0: "country"},inplace=True)
        #average country wage
        grouped =  df_6a.groupby('country')
        df_6a = grouped.agg([np.mean])
        df_6a.drop(df_6a[df_6a.index == ''].index, inplace = True)
        sizes = df_6a['wage']['mean']
    else:
        df_6a = df_6a[[1]]
        df_6a.drop(df_6a[df_6a[1].isnull()].index, inplace=True)
        df_6a[1] = df_6a[1].apply(lambda x: re.sub("[^a-zA-Z ]+", "", x))
        df_6a.rename(index=int, columns={1: "country"},inplace=True)
        grouped = df_6a.groupby('country')
        df_6a = grouped.size()
        df_6a.drop(df_6a[df_6a.index == ''].index, inplace = True)
        sizes = df_6a.values
    #convert country to coordinates
    ccoor = pd.read_csv('CountryLatLong.csv')
    ccoor[['Country','Latitude (average)','Longitude (average)']]
    ccdict = ccoor.set_index('Country')[['Longitude (average)','Latitude (average)']].apply(tuple,axis=1).to_dict()

    lonn = []
    latt = []
    print df_6a

    # put appropriate coordinates in list
    for i in xrange(0,len(df_6a)):
        coords = ccdict[str(df_6a.index[i])]
        lonn.append(coords[0])
        latt.append(coords[1])
        
    #play with norm factor to get right sizes
    source = ColumnDataSource(
        data=dict(lat=latt,
                lon=lonn,
                size=sizes*norm))

    p.circle(x="lon", y="lat", size="size", fill_color=color,line_color=color, fill_alpha=alpha, source=source)

    output_file(outfile)

    return p

def get_Position(df,mapping):
    """
    cleans dataset to show positions
    :param: df, dataset
    :type: pandas dataframe
    """

    assert isinstance(df, pd.DataFrame),'Input must be a dataframe!'
    assert isinstance(mapping,dict), 'Mapping must be a dictionary!'
    
    df['club_pos'].map(mapping)
    return df

def drop_player(df,x,y):
    """
    drop players according to criteria
    :param: df, dataset
    :type: pandas dataframe
    :param: x, column name
    :type: str
    :param: y, criterion
    :type: str
    """
    assert isinstance(df, pd.DataFrame),'Input must be a dataframe!'
    assert isinstance(x, str), 'Column name must be a string!'
    assert isinstance(y, str), 'Criteria must be a string!'
    assert x in att, 'Not an attribute!'
    
    df.drop(df[df[x] == y].index, inplace=True)
    return df

def get_player(df,x,y):
    """
    get players according to criteria
    :param: df, dataset
    :type: pandas dataframe
    :param: x, column name
    :type: str
    :param: y, criterion
    :type: str
    """
    assert isinstance(df, pd.DataFrame),'Input must be a dataframe!'
    assert isinstance(x, str), 'Column name must be a string!'
    assert isinstance(y, str), 'Criteria must be a string!'
    assert x in att, 'Not an attribute!'
    
    df = df[df[x] == y]
    return df