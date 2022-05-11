Use lane-detection.ipynb

Implementation of YOLOP architecture
https://github.com/hustvl/YOLOP

Modified to include include lane of the car detections

Final detections will be located in the processed folder in the following format:

[lane number, x1, y1, x2, y2, confidence, lane center]

If there are external bounding boxes in the JSON format from MASK-RCNN the format will be:

[lane number, x1, y1, x2, y2, confidence, lane center, class id]
