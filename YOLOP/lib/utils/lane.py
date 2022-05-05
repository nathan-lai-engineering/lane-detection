import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
import statistics as stats
import scipy.stats as scistats

# Internal function
# Given a row, returns a list of lane pairs [left_edge, right_edge]
def find_edge_pairs(whole_row):
  edges = []
  for i in range(len(whole_row) - 1):
    if (i == 0 and whole_row[i] == 1) or (whole_row[i] == 0 and whole_row[i + 1] == 1):
      if len(edges) == 0 or i - edges[-1][1] > 50:
        #print(edges)
        edges.append([i, i])
    elif whole_row[i] == 1:
      edges[-1][1] = max(i, edges[-1][1])
  #print(edges)
  return edges
    


# Internal function
# Finds lanes using find_edge_pairs
def find_lanes_pairs(y, ll_seg_mask, inp_center):
  if inp_center is None:
    x_center = get_center(ll_seg_mask)[0]
  else:
    x_center = inp_center
  edge_pairs = find_edge_pairs(ll_seg_mask[y])

  #print("edges", edge_pairs)
  left = []
  right = []

  if len(edge_pairs) > 1:
    edges_to_right = 0
    edges_to_left = 0


    for pair in edge_pairs:
      # Recenters if center is inside a lane
      if x_center >= pair[0] and x_center <= pair[1]:
        if abs(x_center - pair[0]) < abs(x_center - pair[1]):
          x_center = pair[0] - 1
        else:
          x_center = pair[1] + 1

      # Counts edges from center
      if pair[0] >= x_center:
        right.append(pair[0])
      elif pair[1] <= x_center:
        left.append(pair[1])

    # Ensures x_center is within bounds
    x_center = min(x_center, len(ll_seg_mask[y]))
    x_center = max(x_center, 0)

  left.reverse()
  return left, right




# Library function
def get_lane(xy, ll_seg_mask, inp_center=None):
  x_scale = ll_seg_mask.shape[1] / 1280
  y_scale = ll_seg_mask.shape[0] / 720

  inp_center = int(inp_center * x_scale)

  x_center = int(stats.mean([int(xy[0]), int(xy[2])]) * x_scale)
  y_center = int(stats.mean([int(xy[1]), int(xy[3])]) * y_scale)
  y_bottom = int(int(xy[3]) * y_scale)
  y_bottom = min(y_bottom, ll_seg_mask.shape[0] - 1)
  c_center = (x_center, y_center)
  c_bot = (x_center, y_bottom)
 
  left_lanes, right_lanes = find_lanes_pairs(y_bottom, ll_seg_mask, inp_center)

  #print(left_lanes, right_lanes)

  y_img_bottom_quarter = int(4 * ll_seg_mask.shape[0] / 5)
  img_x_center = img_x_center = int(len(ll_seg_mask[y_bottom]) / 2)
  car_tip = (img_x_center, y_img_bottom_quarter)
  dy = len(ll_seg_mask) - 1 - y_bottom
  dx = x_center - img_x_center
  if dx != 0 and dy != 0:
    slope = dx / dy
  else:
    slope = 0
  line_to_car = []
  #print(img_x_center, y_img_bottom_quarter, x_center, y_bottom)
  for i in range(int(dy)):
    x_change = int(slope * i) + img_x_center
    #print(x_change, y_img_bottom_quarter - i)
    line_to_car.append(ll_seg_mask[len(ll_seg_mask) - 1 - i][x_change])
  
  line_to_car_intersections = []
  points_to_graph = []
  for i in range(len(line_to_car) - 1):
    if line_to_car[i] == 0 and line_to_car[i + 1] == 1:
      if len(line_to_car_intersections) > 0:
        if abs(line_to_car_intersections[-1] - i) >= 5:
          line_to_car_intersections.append(i)
          points_to_graph.append((int(line_to_car[i]), int(i)))
      else:
        line_to_car_intersections.append(i)
        points_to_graph.append((int(line_to_car[i]), int(i)))
  
  #print(xy)
  #print(line_to_car)
  #print(line_to_car_intersections)
  #print("\n\n")


  #print(x_center)
  
  iterations = 1
  done = False
  # goes down further if it cannot find lanes to go in between
  while not done and iterations <= 5:
    y_bottom_further =  int(y_bottom + (ll_seg_mask.shape[0] * 0.02 * iterations))
    y_bottom_further = min(y_bottom_further, ll_seg_mask.shape[0] - 1)
    left_lanes, right_lanes = find_lanes_pairs(y_bottom_further, ll_seg_mask, inp_center)

    #points_to_graph = []
    #for i in find_edge_pairs(ll_seg_mask[y_bottom_further]):
    #  points_to_graph.append((int(i[0] / x_scale), int(y_bottom_further / y_scale)))
    #  points_to_graph.append((int(i[1] / x_scale), int(y_bottom_further / y_scale)))
    #print(points_to_graph)

    if len(left_lanes) > 0 and len(right_lanes) > 0:
      if x_center > left_lanes[0] and x_center < right_lanes[0]:
        if len(line_to_car_intersections) == 0:
          return 0, points_to_graph
      elif (x_center > left_lanes[0] or x_center < right_lanes[0]) and len(line_to_car_intersections) > 0:
        done = True
      
    iterations += 1

  if len(left_lanes) == 0 or len(right_lanes) == 0:
    if len(line_to_car_intersections) == 0:
      return 0, points_to_graph
    elif len(line_to_car_intersections) > 0:
      if x_center > img_x_center:
        return len(line_to_car_intersections), points_to_graph
      else:
        return -len(line_to_car_intersections), points_to_graph
    return None, points_to_graph

  elif x_center <= left_lanes[0]:
    lane_number = 1
    for i in range(len(left_lanes)):
      if x_center <= left_lanes[i]:
        lane_number = max(lane_number, i + 1)

    return -lane_number, points_to_graph

  elif x_center >= right_lanes[0]:
    lane_number = 1
    for i in range(len(right_lanes)):
      if x_center >= right_lanes[i]:
        lane_number = max(lane_number, i + 1)
    return lane_number, points_to_graph

  return None, points_to_graph

# Library function
def get_center(ll_seg_mask):
  x_scale = ll_seg_mask.shape[1] / 1280
  y_scale = ll_seg_mask.shape[0] / 720
  center = int(ll_seg_mask.shape[1] / 2)
  y_img_bottom_quarter = int(4 * ll_seg_mask.shape[0] / 5)
  y_img_center = int(5 * ll_seg_mask.shape[0] / 8)

  # list of lanes [lane1, lane2] = [[p1,p2,p3], [p1,p2,p3]] = [[[x,y], [x,y]], [[x,y]]]
  # lanes are lists of points [point1, point2, point3] = [[x,y], [x,y], [x,y]]
  # points are len=2 lists of [x, y]
  lanes_by_points = []

  points_to_graph = []
  points_max = 0
  i = 0
  while i < 15 and points_max < 2:
    y_point = y_img_bottom_quarter - int(ll_seg_mask.shape[0] * 0.01 * i)
    y_point = min(y_point, ll_seg_mask.shape[0])
    y_point = max(y_point, 0)
    edges_pairs = find_edge_pairs(ll_seg_mask[y_point])
    edge_midpoints = [int(stats.mean(k)) for k in edges_pairs ]

    if len(lanes_by_points) == 0:
      lanes_by_points = [[(k, y_point)] for k in edge_midpoints]
      #print(lanes_by_points)
    else:
      for k in edge_midpoints:
        xy_point = (k, y_point)
        points_to_graph.append((int(k / x_scale), int(y_point / y_scale)))
        min_diff = None
        for c in range(len(lanes_by_points)):
          if len(lanes_by_points[c]) > 0 and abs(lanes_by_points[c][-1][0] - k) <= 50:
            if min_diff is None or abs(lanes_by_points[c][-1][0] - k) < abs(lanes_by_points[min_diff][-1][0] - k):
              min_diff = c
        if min_diff is None:
          lanes_by_points.append([xy_point])
        else:
          lanes_by_points[min_diff].append(xy_point)

    points_max = 0
    for lane in lanes_by_points:
      if len(lane) >= 3:
        points_max += 1
    i += 1

  
  vanish_x_list = []
  #print(lanes_by_points)
  for lane in lanes_by_points:
    #print("lane", lane)
    slope, intercept, _, _, _ = scistats.linregress(np.asarray(lane))
    if not np.isnan(slope) and not np.isnan(intercept) and slope != 0:
      try:
        vanishing_x = int((y_img_center - intercept) / slope)
        if not vanishing_x in vanish_x_list:
          vanish_x_list.append(vanishing_x)
      except:
        print("error", y_img_center, intercept, slope)
    #print(slope,intercept)
    #print(vanishing_x, lane)

  #print("center", int(stats.mean(vanish_x_list)))
  #print(y_img_center)
  
  #print("vanish_x", vanish_x_list)
  if len(vanish_x_list) > 0:
    center = int(stats.mean(vanish_x_list))
  #print("center", center)
  center = min(center, ll_seg_mask.shape[1])
  center = max(center, 0)

  #print(center, y_img_center)

  center_scaled = int(center / (ll_seg_mask.shape[0] / 720))

  return center_scaled, points_to_graph


