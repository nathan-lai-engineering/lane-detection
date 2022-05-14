Use lane-detection.ipynb

Implementation of YOLOP architecture
https://github.com/hustvl/YOLOP

Added additional lane numbering of cars.

<h1>Cells Explained</h1>
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
&NewLine;
Final detections will be located in the processed folder in the following format:
[lane number, x1, y1, x2, y2, confidence, lane center]
&NewLine;
If there are external bounding boxes in the JSON format from MASK-RCNN the format will be:
[lane number, x1, y1, x2, y2, confidence, lane center, class id]

<h3>Closest Boxes</h3>
Creates copies of post-processed detections with only the closest cars in the adjacent and current lane








