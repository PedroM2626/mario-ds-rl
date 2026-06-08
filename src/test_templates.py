import cv2
import numpy as np

def test_template(template_path):
    print(f"\nTesting {template_path}")
    tmpl = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if tmpl is None:
        print("Failed to load")
        return
    print(f"Original shape: {tmpl.shape}")
    
    # We want to know what this template looks like roughly.
    # Print average pixel intensity
    print(f"Mean intensity: {np.mean(tmpl)}")
    
    # Since it's a cropped text, it probably has high variance (edges).
    # Let's see if we can extract ORB features.
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(tmpl, None)
    print(f"ORB keypoints: {len(kp) if kp is not None else 0}")

test_template('images/finish.png')
test_template('images/timeout.png')
test_template('images/death.png')
