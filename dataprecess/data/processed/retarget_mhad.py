import os
import sys
import numpy as np

sys.path.append('../../motion')
import BVH as BVH
import Animation as Animation
from Quaternions import Quaternions
from Pivots import Pivots
from InverseKinematics import JacobianInverseKinematics

path='../external/MHAD/BerkeleyMHAD/Mocap/SkeletalData'

bvh_files = [path+'/'+f for f in sorted(list(os.listdir(path)))
    if os.path.isfile(os.path.join(path,f)) and f.endswith('.bvh')]

database = bvh_files

rest, names, _ = BVH.load('./cmu/rest.bvh')
BVH.save('./mhad/rest.bvh', rest, names)

for i, filename in enumerate(database):
    
    print('%i of %i Processing %s' % (i+1, len(database), filename))
    
    mhanim, mhnames, ftime = BVH.load(filename)
    mhanim = mhanim[::4]
    mhanim.positions = mhanim.positions * 0.19
    mhanim.offsets = mhanim.offsets * 0.19
    
    targets = Animation.positions_global(mhanim)
    
    anim = rest.copy()
    anim.positions = anim.positions.repeat(len(targets), axis=0)
    anim.rotations.qs = anim.rotations.qs.repeat(len(targets), axis=0)
    
    anim.positions[:,0] = targets[:,1]
    
    mapping = {
         0:  0,
         2: 24,  3: 26,  4: 28,  5: 29,
         7: 18,  8: 20,  9: 22, 10: 23,
        11:  1, 12:  2, 13:  3, 15:  4, 16: 5,
        18: 13, 19: 15, 20: 17,
        25:  7, 26:  9, 27: 11}
    
    targetmap = {}
    
    for k in mapping:
        targetmap[k] = targets[:,mapping[k]]
    
    ik = JacobianInverseKinematics(anim, targetmap, iterations=10, damping=2.0, silent=True)
    ik()
    
    BVH.save('./mhad/'+os.path.split(filename)[1], anim, names, 1.0/120)
    
