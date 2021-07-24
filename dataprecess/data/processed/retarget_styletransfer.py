import os
import sys
import numpy as np
import scipy.io as io

sys.path.append('../../motion')
import BVH
import Animation
from Quaternions import Quaternions
from InverseKinematics import JacobianInverseKinematics

dbroot='../external/styletransfer/'

skel = io.loadmat(dbroot+'skeleton.mat')['skel']

names = [x.strip() for x in list(skel['bonenames'][0,0])]
offsets = skel['offset'][0][0]
parents = skel['parentid'][0][0][0]-1
orients = Quaternions.id(len(parents))

rest = Animation.Animation(
     orients[np.newaxis], offsets[np.newaxis], 
     orients, offsets, parents)

if not os.path.exists(dbroot+'bvh'): os.mkdir(dbroot+'bvh')
BVH.save(dbroot+'bvh/rest.bvh', rest, names, frametime=1.0/120.0)

db = io.loadmat(dbroot+'style_motion_database.mat')['motion_database']

takes = {}

for i in range(db.shape[1]):
    
    clip = db[0,i]
    
    style = clip['styleName'][0].lower().replace(' ', '_')
    content = clip['contentName'][0].lower().replace(' ', '_')
    
    if style+'_'+content in takes:
        id = takes[style+'_'+content] + 1
    else:
        id = 0
    
    takes[style+'_'+content] = id
    
    if not os.path.exists(dbroot+'bvh/'+style): os.mkdir(dbroot+'bvh/'+style)
    
    rotation_data = np.array([np.array(x) for x in clip['motion']['rotation'][0]])
    
    rotations = np.empty(rotation_data.shape)
    rotations[:,:,0] = rotation_data[:,:,-1]
    rotations[:,:,1:] = rotation_data[:,:,:-1]
    rotations = -Quaternions(rotations)
    
    positions = offsets[np.newaxis].repeat(len(rotations), axis=0)
    positions[:,0:1] = np.array([np.array(x) for x in clip['motion']['position'][0]])
    
    anim = Animation.Animation(
         rotations, positions, 
         orients, offsets, parents)
    
    BVH.save(dbroot+'bvh/%s/%s_%03i.bvh' % (style, content, id), anim, names, frametime=1.0/120.0)

styles = [
    'angry',
    'childlike',
    'depressed',
    'neutral',
    'old',
    'proud',
    'sexy',
    'strutting']

dbroot='../styletransfer/bvh/'

bvh_files = []

for style in styles:
    
    bvh_files += [dbroot+style+'/'+f for f in sorted(list(os.listdir(dbroot+style)))
        if  os.path.isfile(os.path.join(dbroot+style,f)) and f.endswith('.bvh')]

database = bvh_files

rest, names, _ = BVH.load('./cmu/rest.bvh')

if not os.path.exists('styletransfer'): os.mkdir('styletransfer')
BVH.save('./styletransfer/rest.bvh', rest, names)

for i, filename in enumerate(database):
    
    print('%i of %i Processing %s' % (i+1, len(database), filename))
    
    stanim, stnames, ftime = BVH.load(filename)
    stanim.positions = stanim.positions / 6.0
    stanim.offsets = stanim.offsets / 6.0
    
    targets = Animation.positions_global(stanim)
    
    anim = rest.copy()
    anim.positions = anim.positions.repeat(len(targets), axis=0)
    anim.rotations.qs = anim.rotations.qs.repeat(len(targets), axis=0)
    
    anim.positions[:,0] = targets[:,2] - np.array([0, 1.0, 0])
    anim.rotations[:,0] = stanim.rotations[:,0]

    mapping = {
         1:  1,  2: 20,  3: 21,  4: 22,  5: 23,
         6:  1,  7: 15,  8: 16,  9: 17, 10: 18,
        14:  3, 15: 14,  6: 14,
        17:  8, 18:  9, 19: 10, 20: 11, 21: 12, 22: 12, 23: 12,
        24:  3, 25:  4, 26:  5, 27:  6, 28:  7, 29:  7, 30:  7,  
    }
    
    targetmap = {}
    
    for k in mapping:
        anim.rotations[:,k] = stanim.rotations[:,mapping[k]]
        targetmap[k] = targets[:,mapping[k]]
    
    ik = JacobianInverseKinematics(anim, targetmap, iterations=10, damping=10.0, silent=True)
    ik()
    
    BVH.save('./styletransfer/'+filename.replace('../styletransfer/bvh/','').replace('/','_').replace('\\','_'), anim, names, ftime)
    
