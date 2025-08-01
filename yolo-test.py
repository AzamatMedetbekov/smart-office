from ultralytics import YOLO
import cv2 as cv


model = YOLO('yolo11l.pt')

results = model("office_4.jpg")


annotated_frame = results[0].plot()

cv.imshow("Detected Objects", annotated_frame)

cv.waitKey(0)

cv.destroyAllWindows()

print("Script finished. All windows closed.")

