# investing-echo-chambers
Research on investing-related submissions on Reddit. The aim is to detect user communities and analyze their sentiment, to understand if there is any echo chamber.

# How to run

## Python script
Create a PyCharm project with the root directory set to "data-gathering".

Follow the instructions on the praw.ini.sample file to create a new praw.ini file. 
This is necessary to access the Reddit API.

After that, the script app.py can be run.

## RStudio project
Open the project by clicking on "wsb-network-visualization.Rproj".
The code to produce the visualizations is inside "src/visualization.Rmd",
there is no additional setup needed.