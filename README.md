# Data-Challenges-Ceramic
Repository for the Ceramic Challenge

## Dependency:

>python 3.10 - 3.12 (python 3.13 is too new for the MongoDB package)

>MongoDB(Download: https://www.mongodb.com/try/download/community)

Install MongoDB with default settings and with compass.

Running the main will automatically setup the database and collections, but not populate them

## Manage Files

> .SVG uploads

Upload multiple svgs at once. Press the Upload Button to confirm

Upload the svg with their full names as they are (Eg. recons_10001.svg), the program reads the sample_id from the file name

> .CSV uploads

Open the excel file and save it as a .csv

There are still issues with the quality of the data: Duplicate entries are more or less ignored and multiple entries in one cell are stored as just one value

The exact format of the csv should not matter, as the program detects it itself. If there are suspected issues, please report them

Confirm the upload by pressing the Upload button

## Analyse files

> Select .SVG to analyse

Select the sample_id of an SVG from the dropdown. You can enter text to narrow down the options. An .SVG needs to be uploaded to the database before it can be selected here

> Show .SVG

You can display the SVG by pressing the corresponding button