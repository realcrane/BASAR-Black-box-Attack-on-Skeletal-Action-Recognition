import os
import sys
import numpy as np

sys.path.append('../../motion')
import BVH as BVH
import Animation as Animation
from Quaternions import Quaternions
from InverseKinematics import BasicJacobianIK
from InverseKinematics import JacobianInverseKinematics

path='../external/edin_locomotion/'

bvh_files = [path+f for f in sorted(list(os.listdir(path))) if 
    os.path.isfile(os.path.join(path,f)) and f.endswith('.bvh') and f != 'skeleton.bvh' and 'xsens' in f]

files = {
    path+'Locomotion_Zombies/cleaned/gorilla_run_xsens_001.bvh':  path+'Locomotion_Zombies/cleaned/gorilla_run_cmu_001.bvh',  
    path+'Locomotion_Zombies/cleaned/gorilla_walk_xsens_001.bvh': path+'Locomotion_Zombies/cleaned/gorilla_walk_cmu_001.bvh', 
    path+'Locomotion_Zombies/cleaned/monkey_run_xsens_001.bvh':   path+'Locomotion_Zombies/cleaned/monkey_run_cmu_001.bvh',   
    path+'Locomotion_Zombies/cleaned/zombie_run_xsens_001.bvh':   path+'Locomotion_Zombies/cleaned/zombie_run_cmu_001.bvh',   
    path+'Locomotion_Zombies/cleaned/zombie_run_xsens_002.bvh':   path+'Locomotion_Zombies/cleaned/zombie_run_cmu_002.bvh',   
    path+'Locomotion_Zombies/cleaned/zombie_walk_xsens_001.bvh':  path+'Locomotion_Zombies/cleaned/zombie_walk_cmu_001.bvh',  
    path+'Locomotion_Zombies/cleaned/zombie_walk_xsens_002.bvh':  path+'Locomotion_Zombies/cleaned/zombie_walk_cmu_002.bvh',
    
    path+'Aikido/cleaned/aikido_000_p0.bvh': path+'Aikido/cleaned/aikido_cmu_000_p0.bvh',
    path+'Aikido/cleaned/aikido_000_p1.bvh': path+'Aikido/cleaned/aikido_cmu_000_p1.bvh',
    path+'Aikido/cleaned/aikido_001_p0.bvh': path+'Aikido/cleaned/aikido_cmu_001_p0.bvh',
    path+'Aikido/cleaned/aikido_001_p1.bvh': path+'Aikido/cleaned/aikido_cmu_001_p1.bvh',
}

bvh_files = sorted(list(set(files.keys())))
cmu_files = [files[f] for f in bvh_files]

rest, names, _ = BVH.load('./cmu/rest.bvh')
BVH.save('./edin_misc/rest.bvh', rest, names)

for i, (bvh_file, cmu_file) in enumerate(zip(bvh_files, cmu_files)):
    
    print('%i of %i Processing %s' % (i+1, len(bvh_files), os.path.split(cmu_file)[1]))
    
    bvhanim, bvhnames, _ = BVH.load(bvh_file)
    bvhanim.positions = bvhanim.positions / 6.66
    bvhanim.offsets = bvhanim.offsets / 6.66

    targets = Animation.positions_global(bvhanim)
    
    anim = rest.copy()
    anim.positions = anim.positions.repeat(len(targets), axis=0)
    anim.rotations.qs = anim.rotations.qs.repeat(len(targets), axis=0)
    
    if os.path.exists(cmu_file):
        cmuanim, cmunames, cmuftime = BVH.load(cmu_file)
        cmuanim.positions = cmuanim.positions / 6.66
        cmuanim.offsets = cmuanim.offsets / 6.66
        anim.positions[:,0] = cmuanim.positions[:,0]
        anim.rotations = cmuanim.rotations
    else:
        if 'aikido' in  cmu_file:
            targets = targets / 0.9
    
        across1 = targets[:,12] - targets[:, 8]
        across0 = targets[:,19] - targets[:,15]
        across = across0 + across1
        across = across / np.sqrt((across**2).sum(axis=-1))[...,np.newaxis]

        forward = np.cross(across, np.array([[0,1,0]]))
        forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]
        target = np.array([[0,0,1]]).repeat(len(forward), axis=0)

        anim.positions[:,0] = targets[:,1]
        anim.rotations[:,0:1] = -Quaternions.between(forward, target)[:,np.newaxis]  
    
    mapping = {
         0: 0,
         1: 0,  2: 19,  3: 20,  4: 21,  5: 22,
         6: 0,  7: 15,  8: 16,  9: 17, 10: 18,
        11: 2, 12:  3, 13:  4, 15:  5, 16:  6,
        17: 4, 18: 12, 19: 13, 20: 14,
        24: 4, 25:  8, 26:  9, 27: 10}
    
    targetmap = {}
    
    for k in mapping:
        targetmap[k] = targets[:,mapping[k]]
    
    ik = JacobianInverseKinematics(anim, targetmap, iterations=10, damping=2.0, silent=False)
    ik()
    
    filename = './edin_misc/'+os.path.split(cmu_file)[1].replace('_cmu','')
    BVH.save(filename, anim, names, 1.0/120)
    
    
