import skimage, cv2, numpy as np, PIL

def make_name_from_list(data):
    if isinstance(data, str):
        return data
    return "+".join(data)

def patch_is_foreground(patch, frac=0.3, size=244):
    patch = patch.convert('L')
    gs_threshold = 180
    thmbnl = np.array(patch)<gs_threshold
    frc_in_patch = np.sum(thmbnl)/(size**2)
    return (frc_in_patch>=frac and frc_in_patch < 0.7)
    
