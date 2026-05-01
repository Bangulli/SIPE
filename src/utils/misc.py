import skimage, cv2, numpy as np, PIL

def make_name_from_list(data):
    if isinstance(data, str):
        return data
    return "+".join(data)

def patch_is_foreground(patch, frac=0.7, size=244):
    patch = patch.convert('L')
    gs_threshold = 180
    thmbnl = np.array(patch)<gs_threshold
    thmbnl_se = skimage.morphology.disk(16)
    thmbnl = cv2.morphologyEx(thmbnl.astype(np.uint8), cv2.MORPH_CLOSE, thmbnl_se).astype(bool)
    thmbnl = cv2.morphologyEx(thmbnl.astype(np.uint8), cv2.MORPH_OPEN, thmbnl_se).astype(bool)
    return (np.sum(thmbnl)/(size**2))>=frac
    
