Use lane-detection.ipynb

Implementation of YOLOP architecture
https://github.com/hustvl/YOLOP

Added additional lane numbering of cars.

<h1>Examples</h1>
<img src="https://github.com/nathan-lai-engineering/lane-detection/blob/main/readme-images/example%201.PNG?raw=true">
Above, frame of car detections with 3 closest lane numbers (adjacent and current lanes)
<br/>
<br/>
<br/>

<img src="https://github.com/nathan-lai-engineering/lane-detection/blob/main/readme-images/example%202.PNG?raw=true">
Above, frame of misdetection of current lane (numbered 0). Lane number assignment is limited by accuracy of original car detection (car in front not detected)

<h1>Usage</h1>
<h3>Setup</h3>
Enters the YOLOP directory

<h3>Parameters</h3>
The parameters such as directory paths and detection settings
preset_center can be found using a later cell.
enforce_enternal_boxes forces the usage of external bounding boxes. The only format accepted is JSON files from MASK RCNN.

<h3>Functions</h3>
Utility functions to be used later on

<h3>Detection</h3>
The call to the YOLOP architecture to identify cars and assign lane numbers

<h3>Post-Processing</h3>

Removes lane 0 outliers and adjusts the lane 0 detection to ensure only one lane 0 detection

<br/>

Final detections will be located in the processed folder in the following format:

[lane number, x1, y1, x2, y2, confidence, lane center]

<br/>
If there are external bounding boxes in the JSON format from MASK-RCNN the format will be:

[lane number, x1, y1, x2, y2, confidence, lane center, class id]

<h3>Closest Boxes</h3>
Creates copies of post-processed detections with only the closest cars in the adjacent and current lane

<h3>Generate Videos With Bounding Boxes</h3>
Generates a full video with labeled detections

<h3>Test Center</h3>
Draws circles on a frame of a video, use this to find preset center by adjusting the cell's test_center variable to find the center







