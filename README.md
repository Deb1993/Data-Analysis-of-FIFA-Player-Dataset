# Data Analysis of FIFA Player Dataset
## install bokeh to view images with the below command
conda install bokeh


# Scraping
## Datascrapper.ipynb downloads the data from sofifa.com using Beautiful Soup. This file is written in Python3. All the remaining files are written in Python2.
## dataloader.py has various functions used by Feauture_Selection.ipynb and MachineLearning_models.ipynb
## Feature_Selection.ipynb selects the features used for classification using RFE or recursive feature elimination
## MachineLearning_models.ipynb fits an SVM model to the data set and gives the feature importance for each position
## Similarity Matrix.ipynb gives the similarities/correlation between each of the 4 positions and plots a heatmap based on it
# Dataset
## players11.csv - Top 480 players data, 48 attributes
## players12.csv - ~8000 players data, 48 attributes
## players14.csv - ~10000 players data, 48 attributes
# Visualization
## Data Visualization Football Dataset (ECE 143).ipynb contains all visualizations used in presentation
Uses Bokeh! Will not be viewable from Github, must be downloaded and run to view graphs.
## ModuleVisualization.py contains all functions used to plot data
## CountryLatLong.csv contains latitude and longitude dictionary used for world map plots
obtained from https://opendata.socrata.com/dataset/Country-List-ISO-3166-Codes-Latitude-Longitude/mnkm-8ram/data
