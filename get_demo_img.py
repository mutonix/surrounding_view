import cv2
import os

def main():

    cam = cv2.VideoCapture(4)
    i = 0
    while True:
        ret, img = cam.read()
        i += 1
        if i == 10:
            break

    camera_name = 'right_cam'
    image_file = os.path.join(os.getcwd(), "images", camera_name + ".png")
    cv2.imwrite(image_file, img)
    cam.release()

if __name__ == "__main__":
    main()

    