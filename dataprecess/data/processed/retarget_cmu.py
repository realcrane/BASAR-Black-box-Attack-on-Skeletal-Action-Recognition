import re
import sys
import numpy as np
import scipy.io as io

sys.path.append('../../motion')
import BVH as BVH
import Animation as Animation
from Quaternions import Quaternions
from Pivots import Pivots
from InverseKinematics import BasicJacobianIK, JacobianInverseKinematics

database = []

dbroot = '../external/cmu/'

info = open(dbroot+'cmu-mocap-index-text.txt', 'r', errors='ignore')

for line in info.readlines()[16:]:
    if line.strip() == '': continue
    m0 = re.match('Subject #\d+ \(([^)]*)\)\n', line)
    if m0:
        continue
        
    m1 = re.match('(\d+_\d+)\s+([^\n]*)\n', line)
    if m1:
        id0, id1 = m1.group(1).split('_')
        database.append((id0, id1))
    
info.close()

baddata = set([('90', '10')])

database = [data for data in database 
    if (data[0], data[1]) not in baddata]

""" Begin Processing """

rest, names, _ = BVH.load(dbroot+'01/01_01.bvh')
rest_targets = Animation.positions_global(rest)
rest_height = rest_targets[0,:,1].max()

skel = rest.copy()
skel.positions = rest.positions[0:1]
skel.rotations = rest.rotations[0:1]
skel.positions[:,0,0] = 0
skel.positions[:,0,2] = 0
skel.offsets[0,0] = 0
skel.offsets[0,2] = 0
skel.offsets = skel.offsets * 6.25
skel.positions = skel.positions * 6.25

BVH.save('./skel_motionbuilder.bvh', skel, names)

rest.positions = rest.offsets[np.newaxis]
rest.rotations.qs = rest.orients.qs[np.newaxis]

BVH.save('./cmu/rest.bvh', rest, names)

for i, data in enumerate(database):

    filename = dbroot+data[0]+'/'+data[0]+'_'+data[1]+'.bvh'
    
    print('%i of %i Processing %s' % (i+1, len(database), filename))
    
    anim, _, ftime = BVH.load(filename)
    anim_targets = Animation.positions_global(anim)
    anim_height = anim_targets[0,:,1].max()

    targets = (rest_height / anim_height) * anim_targets[1:]
    
    anim = anim[1:]
    anim.orients.qs = rest.orients.qs.copy()
    anim.offsets = rest.offsets.copy()
    anim.positions[:,0] = targets[:,0]
    anim.positions[:,1:] = rest.positions[:,1:].repeat(len(targets), axis=0)

    targetmap = {}
    
    for ti in range(targets.shape[1]):
        targetmap[ti] = targets[:,ti]
    
    ik = JacobianInverseKinematics(anim, targetmap, iterations=10, damping=2.0, silent=True)
    ik()
    
    BVH.save('./cmu/'+data[0]+'_'+data[1]+'.bvh', anim, names, ftime)    
    