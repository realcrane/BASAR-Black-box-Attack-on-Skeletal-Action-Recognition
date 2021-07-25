import os
import sys
import numpy as np
import scipy.interpolate as interpolate

sys.path.append('../../motion')
import BVH as BVH
import Animation as Animation
from Quaternions import Quaternions
from InverseKinematics import BasicJacobianIK
from InverseKinematics import JacobianInverseKinematics

path='../external/edin_punching/bvh/'

bvh_files = [
    path+f for f in sorted(list(os.listdir(path)))
    if os.path.isfile(os.path.join(path,f))
    and f.endswith('.bvh') and f != 'skeleton.bvh']

rest, names, _ = BVH.load('./cmu/rest.bvh')
BVH.save('./edin_punching/rest.bvh', rest, names)

for i, bvh_file in enumerate(bvh_files):
    
    print('%i of %i Processing %s' % (i+1, len(bvh_files), bvh_file))
    
    bvhanim, bvhnames, _ = BVH.load(bvh_file)
    bvhanim.positions = bvhanim.positions * 0.16
    bvhanim.offsets = bvhanim.offsets * 0.16

    targets = Animation.positions_global(bvhanim)
    
    targets = interpolate.griddata(
        np.linspace(0, 1, len(targets)*1), targets,
        np.linspace(0, 1, len(targets)*4), method='linear')
    
    anim = rest.copy()
    anim.positions = anim.positions.repeat(len(targets), axis=0)
    anim.rotations.qs = anim.rotations.qs.repeat(len(targets), axis=0)
    
    across1 = targets[:,16] - targets[:,12]
    across0 = targets[:, 5] - targets[:, 9]
    across = across0 + across1
    across = across / np.sqrt((across**2).sum(axis=-1))[...,np.newaxis]

    forward = np.cross(across, np.array([[0,1,0]]))
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]
    target = np.array([[0,0,1]]).repeat(len(forward), axis=0)

    anim.positions[:,0] = targets[:,0]
    anim.rotations[:,0:1] = -Quaternions.between(forward, target)[:,np.newaxis]  
    
    mapping = {
         0: 0,
         1: 0,  2: 12,  3: 13,  4: 14,  5: 15,
         6: 0,  7: 16,  8: 17,  9: 18, 10: 19,
        11: 0, 15:  3,
        18: 5, 19:  6, 20:  7,
        25: 9, 26: 10, 27: 11}
    
    targetmap = {}
    
    for k in mapping:
        targetmap[k] = targets[:,mapping[k]]
    
    ik = JacobianInverseKinematics(anim, targetmap, iterations=10, damping=2.0, silent=False)
    ik()
    
    BVH.save('./edin_punching/'+os.path.split(bvh_file)[1], anim, names, 1.0/120)
    
    
