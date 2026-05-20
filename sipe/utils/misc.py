import skimage, cv2, numpy as np, PIL

def make_name_from_list(data):
    if isinstance(data, str):
        return data
    return "+".join(data)

def patch_is_foreground(patch, frac=0.3, size=244):
    patch = patch.convert('L')
    gs_threshold_upper = 180
    gs_threshold_lower = 40
    thmbnl = np.bitwise_and(np.array(patch)<gs_threshold_upper, np.array(patch)>gs_threshold_lower)
    frc_in_patch = np.sum(thmbnl)/(size**2)
    return (frc_in_patch>=frac and frc_in_patch < 0.7)
    
