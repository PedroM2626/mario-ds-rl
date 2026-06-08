import cv2
import numpy as np

def match_images(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None:
        print(f"No descriptors found.")
        return 0
        
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    good_matches = [m for m in matches if m.distance < 50]
    
    print(f"Match {img1_path} vs {img2_path}: {len(good_matches)} good matches")
    return len(good_matches)

match_images('images/finish.png', 'images/finish.png')
match_images('images/timeout.png', 'images/timeout.png')
match_images('images/finish.png', 'images/timeout.png')
