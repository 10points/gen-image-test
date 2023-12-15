import cv2
import numpy as np

# Global variables to store the coordinates
coordinates = []

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        coordinates.append((x, y))
        print(f"Clicked at pixel coordinates: ({x}, {y})")

# Load an image
image_path = "frame_1702462022.1263134.jpg"
image = cv2.imread(image_path)

# Create a window and set the mouse callback function
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)

while True:
    # Display the image
    cv2.imshow("Image", image)

    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Create a binary mask based on the collected coordinates
mask = np.zeros_like(image[:, :, 0])
# for coord in coordinates:
#     mask[coord[1], coord[0]] = 255  # Set the pixel to white in the mask

# Convert the list of coordinates to NumPy array
polygon = np.array(coordinates, np.int32)
polygon = polygon.reshape((-1, 1, 2))

# Draw the polygon (line connecting points)
cv2.polylines(mask, [polygon], isClosed=True, color=(255, 255, 255), thickness=2)

# Fill the polygon with white color
cv2.fillPoly(mask, [polygon], color=(255, 255, 255))

# Save file
cv2.imwrite("mask_img", mask)

# Display the mask
cv2.imshow("Maskimg.jpg", mask)

cv2.waitKey(0)

# Print the collected coordinates
print("Collected Coordinates:", coordinates)

# Close all OpenCV windows
cv2.destroyAllWindows()
