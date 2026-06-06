import numpy as np

from .bones import compute_bone_vectors, POSE_BONES, HAND_BONES
from .angles import compute_hand_angles
from ..constants import POSE_SIZE, FACE_SIZE, HAND_SIZE


def safe_norm(x, eps=1e-6): #it is used to avoid Nans and division by zero in norm
    return np.linalg.norm(x, axis=-1, keepdims=True) + eps 
#computes sqrt(x1^2+x2^2+...+xn^2) which is norm, it adds eps bec if norm is 0 so it becomes a very small no instead of 0 to avoid division by zero so error


def build_features(seq: np.ndarray) -> np.ndarray: 
    #takes np array which we got from the preprocessing and returns another array with extra features we add
    T = seq.shape[0] #seq is (T,438)
    x = seq #just using name x for simplicity

    valid = (np.abs(x).sum(axis=-1) > 0) #checks if frames is all 0, to be valid frame it must have any value > 0 because if all 0 so thats all empty so invalid

    



    # Multi-scale velocity (v2)
    v2 = np.zeros_like(x) #create an array of the same shape as x and fileld with 0s
    for t in range(2, T):
        if valid[t] and valid[t - 2]:
            v2[t] = x[t] - x[t - 2]
    #multi scale velocity measures the velocity between 2 frames not only 1 frame like normal velocity
    #so it is for longer window frame
    #it captures longer motion and is less sensitive to noise



    # Split parts    
    pose = x[:, :POSE_SIZE].reshape(T, 33, 4)[..., :3] #extract pose from the array and reshapes it and drops visibility
    lh   = x[:, POSE_SIZE + FACE_SIZE : POSE_SIZE + FACE_SIZE + HAND_SIZE].reshape(T, 21, 3) ##extract left hand from the array and reshapes it
    rh   = x[:, POSE_SIZE + FACE_SIZE + HAND_SIZE :].reshape(T, 21, 3) #extract right hand from the array and reshapes it
    #face is already clear enough so no need to feature engineer it for the model as it doesnt need bones or angles or anything, its good enough unlike hands and pose which need bones and angles


    
    # Bone features
    pose_bones = compute_bone_vectors(pose, POSE_BONES) #create the pose bones
    lh_bones   = compute_bone_vectors(lh, HAND_BONES) #create left hand bones
    rh_bones   = compute_bone_vectors(rh, HAND_BONES) #create right hand bones

    
    # Angles
    lh_angles = compute_hand_angles(lh) #compute left handle angles
    rh_angles = compute_hand_angles(rh) #compute right hand angles

    # Relative features
    nose = pose[:, 0:1, :] #nose keypoints
    l_sh = pose[:, 11:12, :] #left should kepoints
    r_sh = pose[:, 12:13, :] #right shoulder keypoints
    #this will help us know the relative position of keypoints according to each other
    #so we will know where is the hand compared to body parts
    #so if hand near face we will be able to detect that instead of treating hands and face separately

    #lh is left hand kepoints, rh is right hand keypoints, l_sh is left shoulder, r_sh is right shoulder, nose is nose keypoints
    rel_lh_sh = (lh - l_sh).reshape(T, -1) #left hand relativte to left shoudler
    rel_rh_sh = (rh - r_sh).reshape(T, -1) #right hand relative to right shoulder
    rel_lh_rh = (lh - rh).reshape(T, -1) #left hand relative to right hand
    rel_lh_nose = (lh - nose).reshape(T, -1) #left hand relative to nose
    rel_rh_nose = (rh - nose).reshape(T, -1) #right hand relative to nose

    #construct the relative features 
    relative_features = np.concatenate([
        rel_lh_sh,
        rel_rh_sh,
        rel_lh_rh,
        rel_lh_nose,
        rel_rh_nose
    ], axis=-1)

    # Distance features
    def dist(a, b):
        return np.linalg.norm(a - b, axis=-1, keepdims=True) #gets distane between 2 points as sqrt(x1^2+x2^2)

    lh_index = lh[:, 8] #index finger tip for left hand
    rh_index = rh[:, 8] #index finger tip for right hand

    d_lr = dist(lh_index, rh_index) #computes distance between the 2 hands
    d_lh_nose = dist(lh_index, nose[:, 0]) #computes distance between left hand and nose
    d_rh_nose = dist(rh_index, nose[:, 0]) #computes distance between right hand and nose
    #concatenate all distance features together
    distances = np.concatenate([
        d_lr,
        d_lh_nose,
        d_rh_nose
    ], axis=-1)

    # HANDSHAPE DISTANCES

    def hand_shape_feats(hand):
        wrist = hand[:, 0]
        index_tip = hand[:, 8]
        middle_tip = hand[:, 12]

        curl = np.linalg.norm(index_tip - wrist, axis=-1, keepdims=True) #curl means how bent the finger is
        spread = np.linalg.norm(index_tip - middle_tip, axis=-1, keepdims=True) #spread is sepration between index and middle fingers

        return np.concatenate([curl, spread], axis=-1)

    lh_shape = hand_shape_feats(lh) #compute the hand shape for the left hand
    rh_shape = hand_shape_feats(rh) #compute the hand shape for the right hand
    #concatenate them together in handshape features
    handshape_features = np.concatenate([lh_shape, rh_shape], axis=-1)

    #now we will concatenate all the features we computed in 1 variable to return them with T to the user
    features = np.concatenate([
        v2,#v2 is multiscale velocity, v2[t]=x[t]-x[t-2], it is for movement over a longer time window, across 2 frames not 2 like normal v, it also helps ignore noise better
        pose_bones, #pose bones
        lh_bones, #left hand bones
        rh_bones, #right hand bones
        lh_angles, #hand angles are used to know how much each finger is bent, which is important for sign language
        rh_angles, #lh_angles is left hand angles, rh_angles is right hand angles
        relative_features, #used to capture relationships instead of positions, we add for Hand<->Shoulder, Hand<->Hand, Hand<->Face
        distances, #captures closeness/touching, if hands are touching or near face so captures that
        handshape_features #curl > how much finger is bent, if bent small so closed, if large so extended, spread > to capture how far fingers are from each other
    ], axis=-1)

    return features.astype(np.float32) #so we return x and added features ie the original landmarks and added features on them too
    #x is 438, v is 438, a is 438, v2 is 438, direction is 438, pose bones are 18, left hand right hand bones each 60, left hand and right hand angles each 15, relative features are 315, distances are 3
    #hand shape features are spread and curl for both hands so 2x2 so 4
    #so final result:
    #v2             = 438
    #pose_bones     = 18
    #lh_bones       = 60
    #rh_bones       = 60
    #lh_angles      = 15
    #rh_angles      = 15
    #relative       = 315
    #distances      = 3
    #handshape      = 4
    #Total is 928 features