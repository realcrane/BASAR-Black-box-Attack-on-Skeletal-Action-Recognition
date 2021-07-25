import sys
import socket
import threading
import numpy as np
import scipy.io as io
import scipy.interpolate as interpolate

sys.path.append('../../motion')
import BVH as BVH
import Animation as Animation
import AnimationPositions as AnimationPositions
import TimeWarp as TimeWarp
from Quaternions import Quaternions
from Pivots import Pivots
from InverseKinematics import JacobianInverseKinematics

kinect_captures = [
    'SecondCapture1', 'SecondCapture2', 'SecondCapture3',
    'SecondCapture4', 'SecondCapture5', 'SecondCapture6',
    'SecondCapture7', 'SecondCapture8', 'SecondCapture9',
    'SecondCapture10', 'SecondCapture12']
    
kinect_starts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
kinect_stops  = [None, None, 5588, None, None, None, None, None, None, None, None]
kinect_skels  = [2, 0, 1, 4, 4, 4, 4, 4, 4, 4, 3]

xsens_captures = [
    'KinectCapture-001_Dan', 'KinectCapture-002_Dan', 'KinectCapture-003_Dan',
    'KinectCapture-004_Dan', 'KinectCapture-005_Dan', 'KinectCapture-006_Dan',
    'KinectCapture-008_Dan', 'KinectCapture-009_Dan', 'KinectCapture-010_Dan',
    'KinectCapture-011_Dan', 'KinectCapture-013_Dan']
    
xsens_starts = [615, 374, 237, 231, 206, 304, 155, 183, 296, 213, 345]
xsens_stops  = [615+1133, 374+6373, 237+5625, 231+9393, 206+11167, 
     304+10543, 155+7809, 183+11454, 296+8375, 213+6728, 345+5286]

rest, names, _ = BVH.load('./cmu/rest.bvh')
BVH.save('./edin_kinect/rest.bvh', rest, names)
BVH.save('./edin_xsens/rest.bvh', rest, names)

for i, kin, kin_start, kin_stop, kin_skel, xsen, xsen_start, xsens_stop in zip(
    range(len(kinect_captures)),
    kinect_captures, kinect_starts, kinect_stops, 
    kinect_skels, xsens_captures, xsens_starts, xsens_stops):
    
    print("Processing Capture %i of %i" % (i, len(kinect_captures)))
    
    ###
    
    xanimation, _, _ = BVH.load('../external/edin_kinect/'+xsen+'.bvh')
    xanimation = xanimation[xsen_start:xsens_stop]
    
    xsenstargets = Animation.positions_global(xanimation) * 0.16
    xsenstargets = interpolate.griddata(
        np.linspace(0, 1, len(xsenstargets)*1), xsenstargets,
        np.linspace(0, 1, len(xsenstargets)*4), method='linear')
    
    xanim = rest.copy()
    xanim.positions = xanim.positions.repeat(len(xsenstargets), axis=0)
    xanim.rotations.qs = xanim.rotations.qs.repeat(len(xsenstargets), axis=0)
    xanim.positions[:,0] = xsenstargets[:,1]
    
    across1 = xsenstargets[:,12] - xsenstargets[:, 8]
    across0 = xsenstargets[:,19] - xsenstargets[:,15]
    across = across0 + across1
    across = across / np.sqrt((across**2).sum(axis=-1))[...,np.newaxis]

    forward = np.cross(across, np.array([[0,1,0]]))
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]
    target = np.array([[0,0,1]]).repeat(len(forward), axis=0)

    xanim.rotations[:,0:1] = -Quaternions.between(forward, target)[:,np.newaxis]  
    
    mapping = {
          0:  2,
          2: 19,  3: 20,  4: 21,  5: 22,
          7: 15,  8: 16,  9: 17, 10: 18,
         12:  3, 13:  4, 15:  5, 16:  6,
         18: 12, 19: 13, 20: 14,
         25:  8, 26:  9, 27: 10,
    }
    
    targetmap = {}
    
    for k in mapping:
        targetmap[k] = xsenstargets[:,mapping[k]]
    
    ik = JacobianInverseKinematics(xanim, targetmap, iterations=10, damping=2.0, silent=False)
    ik()
    
    BVH.save('./edin_xsens/capture_%03i.bvh' % i, xanim, names, 1.0/120)
        
    #########
    
    kinecttargets = io.loadmat('../external/edin_kinect/'+kin+'.mat')['skel'].astype(np.float)
    kinecttargets = kinecttargets[kin_start:kin_stop,kin_skel,:,:3]
    
    kinecttargets = interpolate.griddata(
        np.linspace(0, 1, len(kinecttargets)*1), kinecttargets,
        np.linspace(0, 1, len(kinecttargets)*4), method='linear')
    
    kinectangle, kinectheight = 6.0, 0.75
    
    rotation1 = Quaternions.from_angle_axis(
        np.ones([1,1]) * np.radians(kinectangle), 
        np.array([[[1,0,0]]]))
    
    kinecttargets = rotation1 * kinecttargets
    
    kinecttargets = kinecttargets * 16.0
    kinecttargets[:,:,1] = kinecttargets[:,:,1] - kinecttargets[:,:,1].min()
    
    kanim = rest.copy()
    kanim.positions = kanim.positions.repeat(len(kinecttargets), axis=0)
    kanim.rotations.qs = kanim.rotations.qs.repeat(len(kinecttargets), axis=0)
    kanim.positions[:,0] = (kinecttargets[:,1] + kinecttargets[:,0]) / 2
    
    across1 = kinecttargets[:, 4] - kinecttargets[:, 8]
    across0 = kinecttargets[:,12] - kinecttargets[:,16]
    across = across0 + across1
    across = across / np.sqrt((across**2).sum(axis=-1))[...,np.newaxis]

    forward = np.cross(across, np.array([[0,1,0]]))
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]
    target = np.array([[0,0,1]]).repeat(len(forward), axis=0)

    kanim.rotations[:,0:1] = Quaternions.between(forward, target)[:,np.newaxis]  
    
    mapping = {
         0:  0,
         2: 12,  3: 13,  4: 14,  5: 15,
         7: 16,  8: 17,  9: 18, 10: 19,
        12:  1, 14:  2, 15:  2, 16:  3,
        18:  4, 19:  5, 20:  6, 22:  7,
        25:  8, 26:  9, 27: 10, 29: 11,
    }
    
    targetmap = {}
    
    for k in mapping:
        targetmap[k] = kinecttargets[:,mapping[k]]
    
    ik = JacobianInverseKinematics(kanim, targetmap, iterations=10, damping=2.0, silent=False)
    ik()
    
    BVH.save('./edin_kinect/capture_%03i.bvh' % i, kanim, names, 1.0/120)
    
    #########
    
    kanimwarp = kanim.copy()
    kanimwarp.positions[:,0] = 0
    kanimwarp.rotations[:,0] = Quaternions.id(1)
    kanimwarp = Animation.positions_global(kanimwarp).reshape(len(kanim), -1)
    
    xanimwarp = xanim.copy()
    xanimwarp.positions[:,0] = 0
    xanimwarp.rotations[:,0] = Quaternions.id(1)
    xanimwarp = Animation.positions_global(xanimwarp).reshape(len(xanim), -1)
    
    astartw = TimeWarp.AStarTW(kanimwarp, xanimwarp, silent=False)
    
    kanim.positions = interpolate.griddata(
        np.arange(len(kanim)), kanim.positions,
        astartw(np.arange(len(xanim))),
        method='linear')
    
    kanim.rotations.qs = Quaternions.exp(interpolate.griddata(
        np.arange(len(kanim)), kanim.rotations.log(),
        astartw(np.arange(len(xanim))),
        method='linear')).qs
    
    BVH.save('./edin_kinect/capture_%03i.bvh' % i, kanim, names, 1.0/120)
    