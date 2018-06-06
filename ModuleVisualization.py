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


def radar_plot(df,Attributes,mapping,outfile,user=True,GK=True):
    
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
    :param: user, whether to use user parameters or machine learned
    :type: boolean
    :param: GK, whether to keep or drop goalkeepers
    :type: boolean

    """

    #define dataframe used for radar plots
    df_user_radar = df

    if user == True:
        #Group parameters
        df_user_radar['Tackling'] = df_user_radar[['SlidingTackle','StandingTackle']].mean(axis=1)
        df_user_radar['Passing'] = df_user_radar[['LongPassing','ShortPassing']].mean(axis=1)
        df_user_radar['Movement'] = df_user_radar[['Acceleration','SprintSpeed','Agility']].mean(axis=1)

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

    if GK == True:
        #IF not including goalkeepers
        df_user_radar.drop('Goalkeeper',inplace=True)

    #cleanup
    df_user_radar.reset_index(level=0, inplace=True)
    df_user_radar.rename(index=int, columns={"index": "Position"},inplace=True)
    
    #set data
    size = len(df_user_radar)    
    fig = plt.figure(figsize=(20,20))
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

        ax.set_title(df_user_radar['Position'][i]+'\n'+'\n')
        # Draw one axe per variable + add labels labels yet
        plt.xticks(angles[:-1], categories, color='black', size=10)
        ax.tick_params(axis='x', which='major', pad=19)
        # set ticklabels location at 1.3 times the axes' radius
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks(color="grey", size=7)
        plt.ylim(0,100)

        # Plot data
        ax.plot(angles, values, linewidth=1, linestyle='solid')

        # Fill area
        ax.fill(angles, values, 'b', alpha=0.1)
    plt.tight_layout() 
    plt.show()

    fig.savefig('%s.png'%outfile)

def bar_plot(df,x='age',y='wage',color='firebrick',droptop=True):
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

    
    
    output_notebook()

    df_2b = df
    if droptop == True:
        #Drop Messi and Cristiano
        df_2b.drop(df_2b[df_2b.full_name == 'L. Messi'].index, inplace=True)
        df_2b.drop(df_2b[df_2b.full_name == 'Cristiano Ronaldo'].index, inplace=True)

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
    show(p)
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

def line_plot_pos(df,x,y,mapping,droptop=True):
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
    :param: droptop, choose whether to drop top players
    :type: boolean

    """

    output_notebook()

    df_3d = df

    #filter 
    df_3d = df_3d[[y, 'club_pos','full_name',x]]


    #group positions by above mapping
    grouped_pos = df_3d.set_index('club_pos').groupby(mapping)



    #Attack
    y1 = grouped_pos.get_group('Attack')
    #FOR B or C
    if droptop == True:
        y1.drop(y1[y1.full_name == 'L. Messi'].index, inplace=True)
        y1.drop(y1[y1.full_name == 'Cristiano Ronaldo'].index, inplace=True)
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
                color="orange", alpha=0.4)
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

    output_notebook()

    df_4e = df
    #filter
    df_4e = df_4e[[x,y,z]]

    #group by rating
    grouped = df_4e.set_index(x).groupby(x)
    df_4e = grouped.agg([np.nanmean])

    source = ColumnDataSource(
        data=dict(x=df[x],
                y=df[y],
                size=(df[z]-18)*norm))


    #plot
    p = figure(title="%s vs %s vs %s"%(z,y,x),plot_width=800, plot_height=400)
    p.grid.grid_line_color = None
    p.background_fill_color = "white"

    circle = Circle(x='x', y='y', size='size', fill_color='orange', fill_alpha=0.4, line_color=None)
    p.add_glyph(source, circle)

    p.scatter(df_4e.index,df_4e[y]['nanmean'], marker='o', size=15,
                color="red", alpha=0.9)
    p.xaxis.axis_label = '%s'%x
    p.yaxis.axis_label = '%s'%y

    #legend
    return p


def world_map_plot(df,norm,outfile,club=True,color='blue'):

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
        sizes = df_6a['wage']['mean']
    else:
        df_6a = df_6a[[1]]
        df_6a.drop(df_6a[df_6a[1].isnull()].index, inplace=True)
        df_6a[1] = df_6a[1].apply(lambda x: re.sub("[^a-zA-Z ]+", "", x))
        df_6a.rename(index=int, columns={1: "country"},inplace=True)
        grouped =  df_6a.groupby('country')
        sizes = grouped.size()
        df_6a = grouped.size()
    #convert country to coordinates
    ccoor = pd.read_csv('CountryLatLong.csv')
    ccoor[['Country','Latitude (average)','Longitude (average)']]
    ccdict = ccoor.set_index('Country')[['Longitude (average)','Latitude (average)']].apply(tuple,axis=1).to_dict()

    lonn = []
    latt = []

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

    p.circle(x="lon", y="lat", size="size", fill_color=color,line_color=color, fill_alpha=0.8, source=source)

    output_file(outfile)

    return p

def get_Position(df,mapping):
    """
    cleans dataset to show positions
    :param: df, dataset
    :type: pandas dataframe
    """
    df['club_pos'].map(mapping)
    return df
