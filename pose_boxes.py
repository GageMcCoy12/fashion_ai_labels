import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from gpt4_vision_tagger import GPT4VisionTagger

# Dictionary mapping keypoint names to indices
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

def load_movenet():
    """Load the MoveNet model for pose detection."""
    model_path = os.path.join("movenet", "4.tflite")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def detect_pose(interpreter, image):
    """Detect human pose in an image using MoveNet Thunder TFLite model."""
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Calculate scale to maintain aspect ratio
    height, width = image.shape[:2]
    input_size = 256  # Thunder model input size
    scale = min(input_size / width, input_size / height)
    scaled_width = int(width * scale)
    scaled_height = int(height * scale)
    
    # Calculate padding
    pad_width = input_size - scaled_width
    pad_height = input_size - scaled_height
    pad_left = pad_width // 2
    pad_top = pad_height // 2
    
    # Resize and convert the image
    input_image = tf.image.resize_with_pad(image, input_size, input_size)
    input_image = tf.cast(input_image, dtype=tf.uint8)
    input_image = tf.expand_dims(input_image, axis=0)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    interpreter.invoke()
    
    # Get keypoints
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    keypoints_with_scores = keypoints_with_scores[0, 0, :, :]
    
    # Convert normalized coordinates to image coordinates
    keypoints_abs = keypoints_with_scores.copy()
    keypoints_abs[:, 0] = (keypoints_abs[:, 0] * input_size - pad_left) / scale
    keypoints_abs[:, 1] = (keypoints_abs[:, 1] * input_size - pad_top) / scale
    
    # Rotate keypoints 90 degrees counterclockwise around image center
    center_x, center_y = width / 2, height / 2
    for i in range(len(keypoints_abs)):
        x, y = keypoints_abs[i, 0], keypoints_abs[i, 1]
        x -= center_x
        y -= center_y
        new_x = -y
        new_y = x
        keypoints_abs[i, 0] = new_x + center_x
        keypoints_abs[i, 1] = new_y + center_y
        # Mirror along Y axis
        keypoints_abs[i, 0] = width - keypoints_abs[i, 0]
    
    return keypoints_abs

def get_clothing_boxes(keypoints, padding=30):
    """Create bounding boxes for shoes, pants, and shirt based on keypoints."""
    boxes = {}
    
    # Helper function to get box coordinates with padding
    def get_box_coords(points, is_shoes=False):
        valid_points = [(x, y) for x, y, conf in points if conf > 0.3]
        if not valid_points:
            return None
        xs, ys = zip(*valid_points)
        
        # For shoes, extend the box to the bottom of the image and make it 20% wider
        if is_shoes:
            width = max(xs) - min(xs) + 2*padding
            width = width * 1.2  # increase width by 20%
            return [
                max(0, min(xs) - padding - (width * 0.1)),  # x (adjusted for wider box)
                max(0, min(ys) - padding),  # y
                width,  # width (20% wider)
                float('inf')  # height (will be adjusted in visualization)
            ]
        else:
            return [
                max(0, min(xs) - padding),  # x
                max(0, min(ys) - padding),  # y
                max(xs) - min(xs) + 2*padding,  # width
                max(ys) - min(ys) + 2*padding   # height
            ]
    
    # Shoes box (using ankles)
    shoe_points = [keypoints[KEYPOINT_DICT['left_ankle']], 
                  keypoints[KEYPOINT_DICT['right_ankle']]]
    boxes['shoes'] = get_box_coords(shoe_points, is_shoes=True)
    
    # Pants box (using hips, knees, and ankles)
    pants_points = [keypoints[KEYPOINT_DICT['left_hip']], 
                   keypoints[KEYPOINT_DICT['right_hip']],
                   keypoints[KEYPOINT_DICT['left_knee']], 
                   keypoints[KEYPOINT_DICT['right_knee']],
                   keypoints[KEYPOINT_DICT['left_ankle']], 
                   keypoints[KEYPOINT_DICT['right_ankle']]]
    boxes['pants'] = get_box_coords(pants_points)
    
    # Shirt box (using shoulders, elbows, and wrists)
    shirt_points = [keypoints[KEYPOINT_DICT['left_shoulder']], 
                   keypoints[KEYPOINT_DICT['right_shoulder']],
                   keypoints[KEYPOINT_DICT['left_elbow']], 
                   keypoints[KEYPOINT_DICT['right_elbow']],
                   keypoints[KEYPOINT_DICT['left_wrist']], 
                   keypoints[KEYPOINT_DICT['right_wrist']]]
    boxes['shirt'] = get_box_coords(shirt_points)
    
    return boxes

def crop_box_region(image, box, add_padding=0.1):
    """Crop a region from the image based on the bounding box coordinates."""
    x, y, w, h = box
    height = image.shape[0]
    
    # Add padding
    pad_w = int(w * add_padding)
    pad_h = int(h * add_padding) if not np.isinf(h) else 0
    
    # Calculate coordinates with padding
    x1 = max(0, int(x - pad_w))
    y1 = max(0, int(y - pad_h))
    x2 = min(image.shape[1], int(x + w + pad_w))
    y2 = min(height, int(y + h + pad_h)) if not np.isinf(h) else height
    
    return image[y1:y2, x1:x2]

def analyze_clothing_regions(image, boxes):
    """Analyze each clothing region using GPT-4V."""
    print("Initializing GPT-4 Vision...")
    fashion_tagger = GPT4VisionTagger()
    analyses = {}
    
    # Create temporary directory for cropped images
    os.makedirs("temp_crops", exist_ok=True)
    
    # Prepare all images first
    temp_files = []
    for clothing_type, box in boxes.items():
        if box is not None:
            print(f"Preparing {clothing_type} for analysis...")
            
            # Crop the region
            cropped = crop_box_region(image, box)
            
            # Save cropped image
            temp_file = f"temp_crops/temp_{clothing_type}.jpg"
            plt.imsave(temp_file, cropped)
            temp_files.append((clothing_type, temp_file))
    
    # Batch analyze all items
    if temp_files:
        print(f"\nBatch analyzing {len(temp_files)} items...")
        image_paths = [f[1] for f in temp_files]
        results = fashion_tagger.tag_images_batch(image_paths)
        
        # Map results back to clothing types
        for (clothing_type, _), result in zip(temp_files, results):
            analyses[clothing_type] = {
                'type': result.get('type', 'similar clothing item'),
                'brand': result.get('brand', 'similar brand'),
                'color': result.get('color', 'neutral tone'),
                'material': result.get('material', 'similar fabric'),
                'aesthetic': result.get('aesthetic', 'versatile style'),
                'extra_details': result.get('extra_details', 'standard design and fit'),
                'confidence': result.get('confidence', 0.9)  # High confidence for GPT-4V
            }
            
            # Print the analysis
            print(f"\nAnalysis for {clothing_type}:")
            for key, value in analyses[clothing_type].items():
                if key != 'confidence':
                    print(f"{key.capitalize()}: {value}")
            print("-" * 50)
    
    # Clean up
    for _, temp_file in temp_files:
        os.remove(temp_file)
    os.rmdir("temp_crops")
    
    return analyses

def visualize_clothing_boxes(image, keypoints, boxes, analyses, output_path):
    """Visualize the clothing bounding boxes, pose skeleton, and GPT-4V analysis."""
    plt.figure(figsize=(15, 15))
    plt.imshow(image)
    
    # Draw pose skeleton
    connections = [
        ('nose', 'left_eye'), ('nose', 'right_eye'),
        ('left_eye', 'left_ear'), ('right_eye', 'right_ear'),
        ('left_shoulder', 'right_shoulder'), ('left_shoulder', 'left_elbow'),
        ('right_shoulder', 'right_elbow'), ('left_elbow', 'left_wrist'),
        ('right_elbow', 'right_wrist'), ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'), ('left_hip', 'right_hip'),
        ('left_hip', 'left_knee'), ('right_hip', 'right_knee'),
        ('left_knee', 'left_ankle'), ('right_knee', 'right_ankle')
    ]
    
    # Draw connections
    for start_name, end_name in connections:
        start_idx = KEYPOINT_DICT[start_name]
        end_idx = KEYPOINT_DICT[end_name]
        start_point = keypoints[start_idx]
        end_point = keypoints[end_idx]
        
        if start_point[2] > 0.3 and end_point[2] > 0.3:
            plt.plot([start_point[0], end_point[0]], 
                    [start_point[1], end_point[1]], 
                    'g-', linewidth=2, alpha=0.7)
    
    # Draw keypoints
    for name, idx in KEYPOINT_DICT.items():
        x, y, confidence = keypoints[idx]
        if confidence > 0.3:
            plt.plot(x, y, 'go', markersize=8, alpha=0.7)
            plt.text(x + 5, y - 5, name, fontsize=8, color='green', alpha=0.7)
    
    # Colors for each clothing type
    colors = {
        'shoes': 'red',
        'pants': 'blue',
        'shirt': 'green'
    }
    
    # Get image height for shoes box
    height = image.shape[0]
    
    # Draw clothing boxes and analysis
    for clothing_type, box in boxes.items():
        if box is not None:
            x, y, w, h = box
            color = colors[clothing_type]
            
            # For shoes, extend the box to the bottom of the image
            if clothing_type == 'shoes':
                h = height - y
            
            # Draw box
            rect = plt.Rectangle((x, y), w, h, fill=False, 
                               edgecolor=color, linewidth=2)
            plt.gca().add_patch(rect)
            
            # Add analysis text if available
            if clothing_type in analyses:
                analysis = analyses[clothing_type]
                info_text = [f"{clothing_type}:"]
                
                # Add type if different from clothing_type
                item_type = analysis.get('type', '')
                if item_type.lower() != clothing_type.lower():
                    info_text.append(f"Type: {item_type}")
                
                # Add brand if not generic
                brand = analysis.get('brand', '')
                if brand and brand.lower() not in ['unbranded', 'generic', 'similar brand']:
                    info_text.append(f"Brand: {brand}")
                
                # Add other details
                info_text.append(f"Color: {analysis.get('color', 'neutral tone')}")
                info_text.append(f"Material: {analysis.get('material', 'similar fabric')}")
                info_text.append(f"Style: {analysis.get('aesthetic', 'versatile style')}")
                
                # Add extra details if not standard
                extra_details = analysis.get('extra_details', '')
                if extra_details and extra_details != 'standard design and fit':
                    info_text.append(f"Details: {extra_details}")
                
                # Add confidence score
                info_text.append(f"Conf: {analysis.get('confidence', 0):.2f}")
                
                # Position text
                if y > 100:
                    text_y = y - 10 * len(info_text)  # Move up based on number of lines
                else:
                    text_y = y + h + 10
                
                plt.text(x, text_y, '\n'.join(info_text), 
                        color='white',
                        bbox=dict(facecolor=color, alpha=0.8),
                        fontsize=8,
                        verticalalignment='bottom')
    
    plt.title("Fashion Analysis")
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()

def process_image(image_path):
    """Process an image to detect pose, create clothing bounding boxes, and analyze with GPT-4V."""
    print(f"\nProcessing {os.path.basename(image_path)}...")
    
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load MoveNet and detect pose
    interpreter = load_movenet()
    print("Detecting human pose...")
    keypoints = detect_pose(interpreter, image)
    
    # Get clothing boxes
    print("Creating clothing bounding boxes...")
    boxes = get_clothing_boxes(keypoints)
    
    # Analyze clothing regions with GPT-4V
    analyses = analyze_clothing_regions(image, boxes)
    
    # Visualize results
    output_dir = "output_boxes"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"analyzed_{os.path.basename(image_path)}")
    visualize_clothing_boxes(image, keypoints, boxes, analyses, output_path)
    print(f"Analysis visualization saved to {output_path}")
    
    return boxes, analyses

def main():
    """Main function to process images."""
    # Process test image
    start_time = time.time()
    test_image = "testing_images/IMG_6658.jpg"
    boxes, analyses = process_image(test_image)
    
    # Print processing time
    elapsed = time.time() - start_time
    print(f"Processing took {elapsed:.2f} seconds")

if __name__ == "__main__":
    main() 