#Python Script using OpenCV-Python (cv2) and YOLO11 to run Object Tracking on Video Frames and on Live Webcam Feed

#Import All the Required Libraries
import os
import cv2
from ultralytics import YOLO

#To avoid the error related to multiple libraries using the same OpenMP runtime
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


#Load the YOLO11 Model
model = YOLO("yolo11n.pt")

#Create a Video Capture Object
cap = cv2.VideoCapture("Resources/Videos/video5.mp4")

# Get video properties for saving
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create output directory
output_dir = "output_videos"
os.makedirs(output_dir, exist_ok=True)

# Setup video writer
output_path = os.path.join(output_dir, "tracked_video5.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

#Loop through Video Frames
while True:
    ret, frame = cap.read()
    if ret:
        #Run YOLO11 Tracking on the Video Frames
        results = model.track(frame, persist=True)
        #Visualize the results on the frame
        annotated_frame = results[0].plot()
        #Write the annotated frame to the output video
        out.write(annotated_frame)
        #Display the annotated frame
        cv2.imshow("YOLO11 Tracking", annotated_frame)
        #Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


cap.release()
out.release()
cv2.destroyAllWindows()







