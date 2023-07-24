import cv2
import os
from ultralytics import YOLO

def predict_and_save_images(class_names_to_keep, output_folder):
    model = YOLO(f'/scratch/tz2518/runs/detect/{class_names_to_keep}/weights/best.pt')

    # The directory of images you want to make prediction
    image_folder = f'/scratch/tz2518/test_all_models/data/{class_names_to_keep}'
    
    # Create the output directory for each class
    class_output_folder = os.path.join(output_folder, class_names_to_keep)
    os.makedirs(class_output_folder, exist_ok=True)

    # Skip non-image files


    # Iterate over every image file in the image_folder
    for image_file_name in os.listdir(image_folder):
        if not image_file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            continue

        image_path = os.path.join(image_folder, image_file_name)

        # Run inference on the source
        results = model(image_path)

        # Draw the results on the image
        result_plotted = results[0].plot()

        # Save the visualized image to the output folder
        output_image_path = os.path.join(class_output_folder, image_file_name)
        cv2.imwrite(output_image_path, result_plotted)

features = ["WINDOW", "PC", "BRICK", "LIGHT-ROOF", "RC2-SLAB", "RC2-COLUMN",
            "RC-SLAB", "RC-JOIST", "RC-COLUMN", "TIMBER-COLUMN", "TIMBER-JOIST"]

output_folder = "/scratch/tz2518/test_all_models/testoutput-YOLOv8-Det"

for feature in features:
    
    predict_and_save_images(feature, output_folder)
