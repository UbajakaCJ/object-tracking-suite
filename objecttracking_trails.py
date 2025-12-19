#Plotting Tracks Over Time

#Import All the Required Libraries
import cv2
import os
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

# Fix OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

#Load the YOLO Model
model = YOLO("yolo11n.pt")

#Create a Video Capture Object
cap = cv2.VideoCapture("Resources/Videos/video7.mp4")

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create output directory
output_dir = "output_videos"
os.makedirs(output_dir, exist_ok=True)

# Setup video writer
output_path = os.path.join(output_dir, "tracked_trails_video7.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

#Store the Track History
track_history = defaultdict(lambda: [])

# Define colors for trails (BGR format - bright and visible)
trail_colors = [
    (0, 255, 0),      # Bright Green
    (255, 0, 255),    # Bright Magenta
    (0, 255, 255),    # Bright Cyan
    (255, 255, 0),    # Bright Yellow
    (255, 0, 0),      # Bright Blue
    (0, 0, 255),      # Bright Red
    (255, 128, 0),    # Orange
    (128, 0, 255),    # Purple
]

#Loop through the Video Frames
frame_count = 0
print("Starting video processing with trails...")
print(f"Video: {frame_width}x{frame_height} @ {fps}fps")

while True:
    ret, frame = cap.read()
    if ret:
        #Run YOLO11 tracking on the frame
        results = model.track(source=frame, persist=True)
        
        # IMPORTANT: Get the annotated frame FIRST (outside the if block)
        annotated_frame = results[0].plot()
        
        # Check if there are any tracked objects
        if results[0].boxes.id is not None:
            #Get the bounding box coordinates and the track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            #Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                
                # Keep only last 30 positions
                if len(track) > 30:
                    track.pop(0)
                
                # Only draw if we have at least 2 points
                if len(track) > 1:
                    # Get color for this track (cycle through colors)
                    color = trail_colors[track_id % len(trail_colors)]
                    
                    # Draw the Tracking Lines
                    points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                    
                    # Draw thick trail line
                    cv2.polylines(annotated_frame, [points], isClosed=False, 
                                color=color, thickness=5)
                    
                    # Draw dots along the trail for better visibility
                    for point in track:
                        cv2.circle(annotated_frame, 
                                 (int(point[0]), int(point[1])), 
                                 3, color, -1)
        
        # Save the frame with trails
        out.write(annotated_frame)
        
        #Display the annotated frame
        cv2.imshow("YOLO11 Tracking with Trails", annotated_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames... ({len(track_history)} active tracks)")
        
        if cv2.waitKey(1) & 0xFF == ord('w'):
            print("Interrupted by user")
            break
    else:
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\n Processing complete!")
print(f"Total frames processed: {frame_count}")
print(f"Total unique objects tracked: {len(track_history)}")
print(f"Video saved to: {output_path}")

# Print tracking statistics
if track_history:
    print("\n Tracking Statistics:")
    for track_id, positions in track_history.items():
        print(f"  Object ID {track_id}: {len(positions)} positions tracked")
