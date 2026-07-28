import cv2
import numpy as np
import matplotlib.pyplot as plt
from stl import mesh
from mpl_toolkits import mplot3d
from stair_detector import StairDetector
import os

def render_stl_to_image(stl_path, output_path):
    print(f"Loading STL from {stl_path}...")
    stair_mesh = mesh.Mesh.from_file(stl_path)
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the mesh
    poly_collection = mplot3d.art3d.Poly3DCollection(stair_mesh.vectors, facecolors='lightgrey', edgecolors='black')
    ax.add_collection3d(poly_collection)
    
    # Auto scale to the mesh size
    scale = stair_mesh.points.flatten()
    ax.auto_scale_xyz(scale, scale, scale)
    
    # Set view angle to simulate robot camera looking at stairs
    # Azimuth around 30 degrees, elevation 20 degrees
    ax.view_init(elev=20, azim=-30) 
    
    # Hide axes for cleaner image
    ax.set_axis_off()
    
    # Save image
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()
    print(f"Saved rendered image to {output_path}")

def test_pipeline():
    stl_file = "Staircase.stl"
    test_image = "synthetic_stairs.png"
    output_image = "synthetic_stairs_annotated.png"
    
    if os.path.exists(stl_file):
        try:
            render_stl_to_image(stl_file, test_image)
        except Exception as e:
            print(f"Failed to render STL: {e}")
            return
    else:
        print(f"STL file {stl_file} not found. Generating basic synthetic image.")
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (100, 300), (540, 400), (200, 200, 200), -1) # bottom step
        cv2.rectangle(img, (150, 200), (490, 300), (150, 150, 150), -1) # mid step
        cv2.rectangle(img, (200, 100), (440, 200), (100, 100, 100), -1) # top step
        cv2.imwrite(test_image, img)
        
    image = cv2.imread(test_image)
    if image is None:
        print(f"Failed to load {test_image}")
        return
        
    detector = StairDetector(
        canny_thresh1=30, 
        canny_thresh2=150,
        hough_min_line_len=50,
        hough_max_line_gap=15,
        horizontal_angle_tolerance=20, # More tolerant for perspective distortion
        step_grouping_tolerance=30
    )
    
    print("Processing frame...")
    annotated_image, steps = detector.process_frame(image)
    
    print(f"Detected {len(steps)} stair steps.")
    for i, step in enumerate(steps):
        print(f"  Step {i+1}: Y-coord = {step.y_coord}, Lines = {len(step.lines)}")
        
    cv2.imwrite(output_image, annotated_image)
    print(f"Saved annotated image to {output_image}")

if __name__ == "__main__":
    test_pipeline()
