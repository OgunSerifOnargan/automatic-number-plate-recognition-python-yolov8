import cv2

# Load the image
image = cv2.imread('/Users/onarganogun/Desktop/desktop/1.jpg')

# Get the dimensions of the image
height, width, _ = image.shape

# Calculate the width of each part
part_width = width // 3

# Crop the image into three equal parts
part1 = image[:, :part_width]
part2 = image[:, part_width:2*part_width]
part3 = image[:, 2*part_width:]

# Save or process the cropped parts as needed
cv2.imwrite('part1.jpg', part1)
cv2.imwrite('part2.jpg', part2)
cv2.imwrite('part3.jpg', part3)
