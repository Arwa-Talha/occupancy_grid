import cv2

points = []
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Image', img)
        if len(points) == 4:
            cv2.destroyAllWindows()

img = cv2.imread('D:/grad project/YOLOP 21/frame_0010.png')
if img is None:
    print("Error: Could not load image.")
    exit()

cv2.imshow('Image', img)
cv2.setMouseCallback('Image', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Selected points:", points)