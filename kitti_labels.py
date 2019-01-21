import numpy as np
import csv
import matplotlib.pyplot as plt

class LabelFile:
    ''' Read a label file and return list of labels each represetned as a dict '''
    def __init__(self, labelfile):
        self.labels = []
        try:
            reader = csv.reader(open(labelfile), delimiter="\t")
            for line in reader:
                label = {}
                vals = line[0].split(' ')
                label['type'] = vals[0]
                label['truncation'] = float(vals[1])
                label['occlusion'] = float(vals[2])
                label['alpha'] = float(vals[3])
                label['x1'] = float(vals[4])
                label['y1'] = float(vals[5])
                label['x2'] = float(vals[6])
                label['y2'] = float(vals[7])
                label['h'] = float(vals[8])
                label['w'] = float(vals[9])
                label['l'] = float(vals[10])
                label['t'] = (float(vals[11]), float(vals[12]), float(vals[13]))
                label['ry'] = float(vals[14])
                self.labels.append(label)
        except FileNotFoundError:
            raise

class CalibFile:
    ''' Read Calibration File and return the calibration Matrix '''
    def __init__(self, filepath):

        try:
            reader = csv.reader(open(filepath), delimiter="\t")
            line = next(reader)
            self.P0 = np.array(line[0].split(' ')[1:], dtype=np.float64).reshape((3,4))
            line = next(reader)
            self.P1 = np.array(line[0].split(' ')[1:], dtype=np.float64).reshape((3,4))
            line = next(reader)   
            self.P2 = np.array(line[0].split(' ')[1:], dtype=np.float64).reshape((3,4))
            line = next(reader)
            self.P3 = np.array(line[0].split(' ')[1:], dtype=np.float64).reshape((3,4))
            line = next(reader)
            self.R0_rect = np.eye(4)
            self.R0_rect[0:3, 0:3] = np.array(line[0].split(' ')[1:], dtype=np.float64).reshape((3,3))
        except:
            raise





def computeBox3D(label, P):
    ''' this function take CalibFile Object and a Label dict
        and computes the 3D Box 
    '''
    face_idx = np.array([
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 4, 8, 7],
        [4, 1, 5, 8]])
    ry = label['ry']
    R = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]])
    l = label['l']
    w = label['w']
    h = label['h']

    # 3D bounding box corners
    x_corners = np.array([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2])
    y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])
    z_corners = np.array([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2])

    corners_3D = R @ np.vstack((x_corners, y_corners, z_corners))
    corners_3D[0,:] = corners_3D[0,:] + label['t'][0]
    corners_3D[1,:] = corners_3D[1,:] + label['t'][1]
    corners_3D[2,:] = corners_3D[2,:] + label['t'][2]
    
    if any(corners_3D[2,:] <0.1):
        return []
    
    return projectToImage(corners_3D, P)

def projectToImage(pts_3D, P):
    pts_2D = P @ np.vstack((pts_3D,np.ones((1,pts_3D.shape[1]))))
    pts_2D[0,:] = pts_2D[0,:] / pts_2D[2,:]
    pts_2D[1,:] = pts_2D[1,:] / pts_2D[2,:]
    return pts_2D[:2]


def computeOrientation3D(label, P):
    ry = label['ry']
    R = np.array([[np.cos(ry),  0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]])
    #orientation in label coordinate system
    orientation_3D = np.array([[0.0, label['l']],
                  [0.0, 0.0],
                  [0.0, 0.0]])
    #rotate and translate in camera coordinate system, project in image
    orientation_3D      = R @ orientation_3D
    orientation_3D[0,:] = orientation_3D[0,:] + label['t'][0]
    orientation_3D[1,:] = orientation_3D[1,:] + label['t'][1]
    orientation_3D[2,:] = orientation_3D[2,:] + label['t'][2]
    if any(orientation_3D[2,:]<0.1):
        orientation_2D = []
        return orientation_2D
    orientation_2D = projectToImage(orientation_3D, P)
    return orientation_2D

