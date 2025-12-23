#MultiThreaded Tracking Provides the Capability to run Object Tracking on Multiple Video Streams
#Import All the Required Libraries
import threading
import cv2
import os
from ultralytics import YOLO


#To avoid the error related to multiple libraries using the same OpenMP runtime
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Create model directory
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)

# Create output directory
output_dir = "output_videos"
os.makedirs(output_dir, exist_ok=True)

#Define Model Names and Video Source
MODEL_NAMES = ["yolo11n.pt", "yolo11n-seg.pt"]

SOURCES = ["Resources/Videos/video5.mp4", "Resources/Videos/video8.mp4"]

def run_tracker_in_thread(model_name, file_name, output_dir, model_dir):
    """Run YOLO Tracker in its own thread for concurrent processing"""
    
    # Construct full model path
    model_path = os.path.join(model_dir, model_name)

    # Load model
    model = YOLO(model_path)
    
    # Get video properties
    cap = cv2.VideoCapture(file_name)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    
    # Setup output path
    video_basename = os.path.splitext(os.path.basename(file_name))[0]
    model_basename = os.path.splitext(os.path.basename(model_name))[0]
    output_filename = f"tracked_{video_basename}_{model_basename}.mp4"
    output_path = os.path.join(output_dir, output_filename)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Run tracking
    results = model.track(source=file_name, persist=True, stream=True)
    
    for r in results:
        # Get annotated frame
        annotated_frame = r.plot()
        
        # Write to output video
        out.write(annotated_frame)
        
        # Display the frame (optional)
        cv2.imshow(f"Tracking: {video_basename} - {model_basename}", annotated_frame)
        
        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    out.release()
    print(f"Saved: {output_path}")

#Create and Start Tracker Threads using a for loop
tracker_threads = []
for video_file, model_name in zip(SOURCES, MODEL_NAMES):
    thread = threading.Thread(
        target=run_tracker_in_thread, 
        args=(model_name, video_file, output_dir, model_dir), 
        daemon=True
    )
    tracker_threads.append(thread)
    thread.start()

#Wait for all tracker threads to finish
for thread in tracker_threads:
    thread.join()

#Clean Up and Close Windows
cv2.destroyAllWindows()

print(f"\nAll tracking complete! Videos saved in '{output_dir}' directory")