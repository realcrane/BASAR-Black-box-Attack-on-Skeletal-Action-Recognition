import os
import sys
import numpy as np
import scipy.interpolate as interpolate

sys.path.append('../../motion')
import BVH as BVH
import Animation as Animation
from InverseKinematics import BasicJacobianIK
from InverseKinematics import JacobianInverseKinematics

path='../external/edin_locomotion/'

bvh_files = [path+'cleaned/'+f for f in sorted(list(os.listdir(path+'cleaned/'))) 
    if os.path.isfile(os.path.join(path+'cleaned/',f)) and f.endswith('.bvh') and f != 'skeleton.bvh']

cmu_filemap = {
    'Locomotion0004.bvh': 'locomotion_walk_001.bvh',
    'Locomotion0005.bvh': 'locomotion_jog_001.bvh',
    'Locomotion0006.bvh': 'locomotion_run_000.bvh',
    'Locomotion0007.bvh': 'locomotion_transitions_000.bvh',
    'Locomotion0008.bvh': 'locomotion_walk_sidestep_000.bvh',
    'Locomotion0009.bvh': 'locomotion_jog_sidestep_000.bvh',
    'Locomotion0010.bvh': 'locomotion_run_sidestep_000.bvh',
    'Locomotion0011.bvh': 'locomotion_jog_000.bvh',
    'Locomotion0012.bvh': 'locomotion_walk_sidestep_001.bvh',
    'Locomotion0013.bvh': 'locomotion_run_sidestep_001.bvh',
    'Locomotion0014.bvh': 'locomotion_jog_sidestep_001.bvh',
    'Locomotion0015.bvh': 'locomotion_walk_000.bvh',
}

footstep_files = [f.replace('.bvh','_footsteps.txt') for f in bvh_files]

cmu_files = [path+'cleaned_cmu/'+cmu_filemap[os.path.split(f)[1]] for f in bvh_files]

rest, names, _ = BVH.load('./cmu/rest.bvh')
BVH.save('./edin_locomotion/rest.bvh', rest, names)

for i, (bvh_file, cmu_file, footstep_file) in enumerate(zip(bvh_files, cmu_files, footstep_files)):
    
    print('%i of %i Processing %s' % (i+1, len(bvh_files), os.path.split(cmu_file)[1]))
    
    bvhanim, bvhnames, _ = BVH.load(bvh_file)
    bvhanim.positions = bvhanim.positions / 6.66
    bvhanim.offsets = bvhanim.offsets / 6.66
        
    cmuanim, cmunames, cmuftime = BVH.load(cmu_file)
    cmuanim.positions = cmuanim.positions / 6.66
    cmuanim.offsets = cmuanim.offsets / 6.66
    
    targets = Animation.positions_global(bvhanim)
    targets = interpolate.griddata(
        np.linspace(0, 1, len(bvhanim)), targets,
        np.linspace(0, 1, len(cmuanim)), method='linear')
    
    anim = rest.copy()
    anim.positions = anim.positions.repeat(len(targets), axis=0)
    anim.rotations.qs = anim.rotations.qs.repeat(len(targets), axis=0)
    anim.positions[:,0] = cmuanim.positions[:,0]
    anim.rotations = cmuanim.rotations
    
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
    
    footsteps = open(footstep_file, 'r')    
    lines = footsteps.readlines()
    footsteps.close()

    phase = np.zeros(len(anim))
    
    for fi in range(len(lines)-1):
        curr = lines[fi+0].strip().split(' ')
        next = lines[fi+1].strip().split(' ')
        if curr[-1] == '*': continue
        i0, i1, i2 = 4*(int(curr[0])-1), 4*(int(curr[1])-1), 4*(int(next[0])-1)
        if i1-i0+1 > 0: phase[i0:i1+1] = np.linspace(0, 0.5, i1-i0+1)[:len(phase[i0:i1+1])]
        if i2-i1+1 > 0: phase[i1:i2+1] = np.linspace(0.5, 1.0, i2-i1+1)[:len(phase[i1:i2+1])]
    
    exported=0
    start=-1
    for li in range(1, len(lines)):
        
        if lines[li].strip().endswith('*') and start != -1:
            end = 4*(int(lines[li-1].split(' ')[1])-1)
            filename = './edin_locomotion/'+os.path.split(cmu_file)[1].replace('.bvh', '_%03i.bvh' % exported)
            BVH.save(filename, anim[start:end], names, cmuftime)
            np.savetxt(filename.replace('.bvh','.phase'), phase[start:end], '%0.5f')
            start = -1
            exported += 1
        elif not lines[li].strip().endswith('*') and start == -1:
            start = 4*(int(lines[li].split(' ')[0])-1)
        
    end = 4*(int(lines[len(lines)-1].split(' ')[0])-1)
    
    filename = './edin_locomotion/'+os.path.split(cmu_file)[1].replace('.bvh', '_%03i.bvh' % exported)
    BVH.save(filename, anim[start:end], names, cmuftime)
    #np.savetxt(filename.replace('.bvh','.phase'), phase[start:end], '%0.5f')

    
