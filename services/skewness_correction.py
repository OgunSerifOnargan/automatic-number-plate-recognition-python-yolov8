import cv2
import numpy as np
from scipy.ndimage import interpolation as inter

def correct_skew(vehicle, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(vehicle.licensePlate.img, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]
    (h, w) = vehicle.licensePlate.img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    if vehicle.licensePlate.img.shape[1] == 0:
        return 0, vehicle.licensePlate.img
    corrected = cv2.warpAffine(vehicle.licensePlate.img, M, (w, h), flags=cv2.INTER_CUBIC, \
            borderMode=cv2.BORDER_REPLICATE)

    return best_angle, corrected

# if __name__ == '__main__':
#     image = cv2.imread('/Users/onarganogun/Desktop/Screenshot 2024-03-25 at 13.42.59.png')
#     angle, corrected = correct_skew(image)
#     print('Skew angle:', angle)
#     cv2.imshow('corrected', corrected)
#     cv2.waitKey()