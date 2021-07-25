import os
import sys
import numpy as np

sys.path.append('../../motion')
import motion.BVH as BVH
import motion.Animation as Animation
from motion.InverseKinematics import JacobianInverseKinematics

classmap = {
    'cartwheelLHandStart1Reps': 'cartwheel',
    'cartwheelLHandStart2Reps': 'cartwheel',
    'cartwheelRHandStart1Reps': 'cartwheel',
    'clap1Reps': 'clap',
    'clap5Reps': 'clap',
    'clapAboveHead1Reps': 'clap',
    'clapAboveHead5Reps': 'clap',
    # 'depositFloorR': 'deposit',
    # 'depositHighR': 'deposit',
    # 'depositLowR': 'deposit',
    # 'depositMiddleR': 'deposit',
    'depositFloorR': 'grab',
    'depositHighR': 'grab',
    'depositLowR': 'grab',
    'depositMiddleR': 'grab',
    'elbowToKnee1RepsLelbowStart': 'elbow_to_knee',
    'elbowToKnee1RepsRelbowStart': 'elbow_to_knee',
    'elbowToKnee3RepsLelbowStart': 'elbow_to_knee',
    'elbowToKnee3RepsRelbowStart': 'elbow_to_knee',
    'grabFloorR': 'grab',
    'grabHighR': 'grab',
    'grabLowR': 'grab',
    'grabMiddleR': 'grab',
    #'hitRHandHead': 'hit',
    'hitRHandHead': 'grab',
    'hopBothLegs1hops': 'hop',
    'hopBothLegs2hops': 'hop',
    'hopBothLegs3hops': 'hop',
    'hopLLeg1hops': 'hop',
    'hopLLeg2hops': 'hop',
    'hopLLeg3hops': 'hop',
    'hopRLeg1hops': 'hop',
    'hopRLeg2hops': 'hop',
    'hopRLeg3hops': 'hop',
    'jogLeftCircle4StepsRstart': 'jog',
    'jogLeftCircle6StepsRstart': 'jog',
    'jogOnPlaceStartAir2StepsLStart': 'jog',
    'jogOnPlaceStartAir2StepsRStart': 'jog',
    'jogOnPlaceStartAir4StepsLStart': 'jog',
    'jogOnPlaceStartFloor2StepsRStart': 'jog',
    'jogOnPlaceStartFloor4StepsRStart': 'jog',
    'jogRightCircle4StepsLstart': 'jog',
    'jogRightCircle4StepsRstart': 'jog',
    'jogRightCircle6StepsLstart': 'jog',
    'jogRightCircle6StepsRstart': 'jog',
    'jumpDown': 'jump',
    'jumpingJack1Reps': 'jump',
    'jumpingJack3Reps': 'jump',
    'kickLFront1Reps': 'kick',
    'kickLFront2Reps': 'kick',
    'kickLSide1Reps': 'kick',
    'kickLSide2Reps': 'kick',
    'kickRFront1Reps': 'kick',
    'kickRFront2Reps': 'kick',
    'kickRSide1Reps': 'kick',
    'kickRSide2Reps': 'kick',
    'lieDownFloor': 'lie_down',
    'punchLFront1Reps': 'punch',
    'punchLFront2Reps': 'punch',
    'punchLSide1Reps': 'punch',
    'punchLSide2Reps': 'punch',
    'punchRFront1Reps': 'punch',
    'punchRFront2Reps': 'punch',
    'punchRSide1Reps': 'punch',
    'punchRSide2Reps': 'punch',
    'rotateArmsBothBackward1Reps': 'rotate_arms',
    'rotateArmsBothBackward3Reps': 'rotate_arms',
    'rotateArmsBothForward1Reps': 'rotate_arms',
    'rotateArmsBothForward3Reps': 'rotate_arms',
    'rotateArmsLBackward1Reps': 'rotate_arms',
    'rotateArmsLBackward3Reps': 'rotate_arms',
    'rotateArmsLForward1Reps': 'rotate_arms',
    'rotateArmsLForward3Reps': 'rotate_arms',
    'rotateArmsRBackward1Reps': 'rotate_arms',
    'rotateArmsRBackward3Reps': 'rotate_arms',
    'rotateArmsRForward1Reps': 'rotate_arms',
    'rotateArmsRForward3Reps': 'rotate_arms',
    # 'runOnPlaceStartAir2StepsLStart': 'run',
    # 'runOnPlaceStartAir2StepsRStart': 'run',
    # 'runOnPlaceStartAir4StepsLStart': 'run',
    # 'runOnPlaceStartFloor2StepsRStart': 'run',
    # 'runOnPlaceStartFloor4StepsRStart': 'run',
    'runOnPlaceStartAir2StepsLStart': 'jog',
    'runOnPlaceStartAir2StepsRStart': 'jog',
    'runOnPlaceStartAir4StepsLStart': 'jog',
    'runOnPlaceStartFloor2StepsRStart': 'jog',
    'runOnPlaceStartFloor4StepsRStart': 'jog',
    'shuffle2StepsLStart': 'shuffle',
    'shuffle2StepsRStart': 'shuffle',
    'shuffle4StepsLStart': 'shuffle',
    'shuffle4StepsRStart': 'shuffle',
    'sitDownChair': 'sit_down',
    'sitDownFloor': 'sit_down',
    'sitDownKneelTieShoes': 'sit_down',
    'sitDownTable': 'sit_down',
    'skier1RepsLstart': 'ski',
    'skier3RepsLstart': 'ski',
    'sneak2StepsLStart': 'sneak',
    'sneak2StepsRStart': 'sneak',
    'sneak4StepsLStart': 'sneak',
    'sneak4StepsRStart': 'sneak',
    'squat1Reps': 'squat',
    'squat3Reps': 'squat',
    'staircaseDown3Rstart': 'climb',
    'staircaseUp3Rstart': 'climb',
    'standUpKneelToStand': 'stand_up',
    'standUpLieFloor': 'stand_up',
    'standUpSitChair': 'stand_up',
    'standUpSitFloor': 'stand_up',
    'standUpSitTable': 'stand_up',
    'throwBasketball': 'throw',
    'throwFarR': 'throw',
    'throwSittingHighR': 'throw',
    'throwSittingLowR': 'throw',
    'throwStandingHighR': 'throw',
    'throwStandingLowR': 'throw',
    'turnLeft': 'turn',
    'turnRight': 'turn',
    'walk2StepsLstart': 'walk_forward',
    'walk2StepsRstart': 'walk_forward',
    'walk4StepsLstart': 'walk_forward',
    'walk4StepsRstart': 'walk_forward',
    'walkBackwards2StepsRstart': 'walk_backward',
    'walkBackwards4StepsRstart': 'walk_backward',
    'walkLeft2Steps': 'walk_left',
    'walkLeft3Steps': 'walk_left',
    'walkLeftCircle4StepsLstart': 'walk_left',
    'walkLeftCircle4StepsRstart': 'walk_left',
    'walkLeftCircle6StepsLstart': 'walk_left',
    'walkLeftCircle6StepsRstart': 'walk_left',
    'walkOnPlace2StepsLStart': 'walk_inplace',
    'walkOnPlace2StepsRStart': 'walk_inplace',
    'walkOnPlace4StepsLStart': 'walk_inplace',
    'walkOnPlace4StepsRStart': 'walk_inplace',
    'walkRightCircle4StepsLstart': 'walk_right',
    'walkRightCircle4StepsRstart': 'walk_right',
    'walkRightCircle6StepsLstart': 'walk_right',
    'walkRightCircle6StepsRstart': 'walk_right',
    'walkRightCrossFront2Steps': 'walk_right',
    'walkRightCrossFront3Steps': 'walk_right',
}

classes_names = list(sorted(list(set(classmap.values()))))

path='../external/hdm05/cuts/'

bvh_files = []

for cls in sorted(classmap.keys()):
    
    files = [path+cls+'/'+f for f in sorted(list(os.listdir(path+cls)))
        if  os.path.isfile(os.path.join(path+cls,f))]
    
    bvh_files += [f.replace('.amc', '.bvh').replace('.AMC', '.bvh') for f in files if f.lower().endswith('amc')]

database = bvh_files
rest, names, _ = BVH.load('./cmu/rest.bvh')
rest_scale = np.sqrt(np.mean(rest.offsets**2))
BVH.save('./hdm05/rest.bvh', rest, names)

for i, filename in enumerate(database):
    
    print('%i of %i Processing %s' % (i+1, len(database), filename))
    a = './hdm05/'+os.path.split(filename)[1].replace('/','_').replace('\\', '_')
    hdm05anim, names, ftime = BVH.load(a)
    #hdm05anim, names, ftime = BVH.load(filename)
    
    anim_scale = np.sqrt(np.mean(hdm05anim.offsets**2))

    targets = Animation.positions_global(hdm05anim)
    
    anim = rest.copy()
    anim.positions = anim.positions.repeat(len(targets), axis=0)
    anim.rotations.qs = anim.rotations.qs.repeat(len(targets), axis=0)
    anim.positions[:,0] = targets[:,0]
    anim.rotations[:,:] = hdm05anim.rotations[:,:]
    
    targetmap = {}
    for ti in range(targets.shape[1]):
        targetmap[ti] = targets[:,ti]
    
    ik = JacobianInverseKinematics(anim, targetmap, iterations=10, damping=2.0, silent=True)
    ik()
    BVH.save('./hdm05/'+os.path.split(filename)[1].replace('/','_').replace('\\', '_'), anim, names, 1.0/120)
    
