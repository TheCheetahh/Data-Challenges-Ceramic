# Data-Challenges-Ceramic
Repository for the Ceramic Challenge

We work on the dev branch. Whenever you want to work on the project, create a new feature branch. If the feature branch has no issues, you can create a pull request and merge it with dev

## Dependency:

>python 3.12 (python 3.13 is too new for the MongoDB package and everything else is too old)

>MongoDB(Download: https://www.mongodb.com/try/download/community)

Install MongoDB with default settings and with compass.

Running the main will automatically setup the database and collections, but not populate them

## Manage Files

> .SVG uploads

Upload multiple svgs at once. Press the Upload Button to confirm

Upload the svg with their full names as they are (Eg. recons_10001.svg), the program reads the sample_id from the file name

> .CSV uploads

Open the excel file and save it as a .csv. Use ";" as separators (it is a german excel table)

There are still issues with the quality of the data: Duplicate entries are more or less ignored and multiple entries in one cell are stored as just one value

The exact format of the csv should not matter, as the program detects it itself. If there are suspected issues, please report them

Confirm the upload by pressing the Upload button

> DO NOT FORGET TO PRESS CLEAN SVG

Sorry for capslock. Not pressing should not cause any errors, but you cannot analyse until you press this for now.

This gets rid of everything in the .svg except the most complex blob, which is used for analysis.

## Analyse files

> Set settings for the analysis

Select smoothing method and other properties

Do not fiddle with it while comparing samples, as different settings will make the program recalculate all sample data with the new settings

> Analyze .SVG

Select the sample_id of an SVG from the dropdown. You can enter text to narrow down the options. An .SVG needs to be uploaded to the database before it can be selected here

Press the Analyze Button to display the cleaned svg, the 1D curvature graph and the curvature color map and the closest other sample with its calculated distance.

Note: Depending on the "Abtastpunkte" the size of all distances will change, no matter how close or far the samples are from each other, as this settings essentially scales the distance between samples. 

The first calculation for a given configuration can worst case take up to a few minutes, depending on your device.
