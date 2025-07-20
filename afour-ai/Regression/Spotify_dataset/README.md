This is the dataset of spotify songs picked from the kaggle.
Link: https://www.kaggle.com/pepepython/spotify-huge-database-daily-charts-over-3-years
This contains 2 csv files.

●	"Database to Calculate Popularity" includes all the daily entries (8mln+) for the songs which made it to the top 200 . Among these data, quite intuitively, you will find the same song being in the charts for more than one day. We then created a popularity score, unique for a given song in a given country, which took into account the position in the charts and the days it stayed there
●	"Final Database" includes many data for each song. It aggregates the popularity for songs into a single score for each. For each song several variables were retrieved by using Spotify's API (such as artist, country, genre, …)
Working on the preprocessing, data cleaning, reduction. Transformation : https://colab.research.google.com/drive/17RUWyF-b_RkRU-IDwEHOVFqOZ1QtK0YK#scrollTo=Z-VWLMTHL0wm
