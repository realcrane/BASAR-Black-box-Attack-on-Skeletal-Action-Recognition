def forward_perturbation(epsilon, prev_sample, target_sample):
    perturb = (target_sample - prev_sample)
    perturb = perturb.permute(1,2,0)
    #perturb = np.transpose(perturb, [1, 2, 0])  # (3,300,25), (300,25,3)
    b = get_diff(target_sample, prev_sample)
    perturb /= b
    perturb *= epsilon
    #perturb = np.transpose(perturb, [2, 0, 1])
    perturb = perturb.permute(2,0,1)
    return perturb