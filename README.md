# Data-Challenges-Ceramic
Repository for the Ceramic Challenge

We work on the dev branch. Whenever you want to work on the project, create a new feature branch. If the feature branch has no issues, you can create a pull request and merge it with dev

## Install and dependencies:

>python 3.12 (python 3.13 is too new for the MongoDB package and everything else is too old)

>MongoDB (Download: https://www.mongodb.com/try/download/community)

Install MongoDB with default settings and with compass.

>On Windows (and some linux):  GTK3: https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases

Add it to PATH

>Load all python package dependencies from requirements_pip.txt

Write in terminal: pip install -r requirements_pip.txt

>Load all python package dependencies with conda from environment.yml

Write in terminal: conda env create -f environment.yml

## How to launch:

Start the program by running main.py and wait until it has booted up. 
It is done when the terminal displays: * Running on local URL:  http://127.0.0.1:7860

Then open http://127.0.0.1:7860 in your browser and enjoy!

Note: The local webpage is just the user interface. Reloading the page does not change anything the program does, but it does reset the ui.

## Manage Files

Any and all data about samples and types that cannot be viewed in the user interface can be viewed in the MongoDB. 

### File Upload

> sample .SVG uploads

Upload multiple svgs of your shard samples at once. Press the Upload Button to confirm the download.

For the templates you can upload the ones you find in this project. They were cleaned of their licenced elements (they used to have a screenshot of their source attached, which disallowed us from sharing them previously).

Upload the svg with their full names as they are, the program reads the sample_id from the file name.
Currently, the correct naming scheme is: recons_#####.svg (with ##### being a number of any length). The program was created assuming this is the name of the sample files, although other names could possibly work too without throwing an error. The _ and everything before it will be deleted from the file name to create the sample id.

> .CSV uploads for additional sample information

Use ";" as separators for the csv file (because its a german program)

The sample_id is being used as the key to match the csv data to the uploaded samples. All csv fields are being uploaded to the corresponding sample.

If there are issues with the upload, the samples either have a different naming scheme or the csv is not using ";"

Confirm the upload by pressing the Upload button

> theory .SVG uploads

Upload multiple svgs of your theory types at once. Press the Upload Button to confirm

Name them exactly after their type. The file name will be used as the type. If you want to upload multiple templates of the same name, make sure they have unique names (add a number or something)

### File Download

> CSV Download of sample data

Download sample data as a .csv

The relevant field would be the type that was selected in the program, the other data is from a potential .csv upload and is not created by this program.

## Edit Sample svg

>Change cropping of sample svgs

Choose the start and endpoint of the desired cropping. You can also adjust where the cropping should start. 

Press the save button to confirm your changes before analyzing or changing to a different sample, otherwise changes will be discarded.

## Analyse files

> Set settings for the analysis

In the settings you can change the calculation method. Choose between ICP, LAA, Orb and DISK

The program will terminate, but some calculations might take a bit longer, up to a few minutes.
 
The other settings are best not touched, except for the show synonym checkbox. It hides all matches from the results that are worse than a different template in the same synonyme group.

Changing calculation method and un/ticking the checkbox needs a press of the analyze button to show the changes. 

> Analyze .SVG

Select the sample_id of an SVG from the dropdown. You can enter text to narrow down the options. An .SVG needs to be uploaded to the database before it can be selected here

Press the Analyze Button to calculate and show the results. 

Navigate the results with the arrow buttons or use the dropdown to navigate them faster. 

The size of the distances/score for the templates changes depending on calculation method. The distance is not comparable between methods.

You can pin the currently selected template to the middle to keep it for comparison. 

The type field on the left displays the type of the sample. It can be edited and the changes will be saved by pressing the save button. Do not forget to press the button before moving on to the next sample!

There are more graphs in an accordion box under the plots, for some calculation methods they can display additional information.

The batch analyse button analyses all samples in one go. The analysis tab cannot be used in the meantime. Depending on the method chosen and the CPU used, this will take hours. It is recommended to first analyse a single sample and see how long it takes. Multiply that time by the amount of samples to get a time estimate for the batch operation. The actual calculation will however take shorter than that time, because it is better parallelized than single analysis. 

## Synonym Rules

> Create synonym groups 

Add the exact names of 2 types to make or add them to a synonym group. Adding a type to another that is already in a group just adds it to said group. 

The current effect of this is just that synonymes are displayed in the analysis tab when a type of a synonym group is selected and the checkbox in the setting can hide worse synonyme results. 

You can also remove a type from its group.