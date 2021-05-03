import cv2
cap = cv2.VideoCapture(0)

# while True:
#     ret,  f  = cam.read()
#     f = cv2.resize(f, (640, 480))   
#     cv2.imshow('haha', f)

#     if cv2.waitKey(0) & 0xff == ord('q'):
#         break

# cv2.destroyAllWindows()
# cam.release()

while (cap.isOpened()):
    ret, frame = cap.read()

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()   