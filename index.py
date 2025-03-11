import os
import cv2
import numpy as np
import tensorflow as tf
from gpt4_vision_tagger import GPT4VisionTagger
import base64
from io import BytesIO
from PIL import Image

# Import our processing functions
from pose_boxes import (
    detect_pose,
    get_clothing_boxes,
    crop_box_region,
    analyze_clothing_regions
)

# Global caches
interpreter = None
fashion_tagger = None

def init_model():
    """Initialize and cache the MoveNet model."""
    global interpreter
    if interpreter is None:
        model_path = os.path.join("movenet", "4.tflite")
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
    return interpreter

def init_gpt4v():
    """Initialize and cache the GPT-4V tagger."""
    global fashion_tagger
    if fashion_tagger is None:
        # Check for API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        fashion_tagger = GPT4VisionTagger()
    return fashion_tagger

def process_image_bytes(image_bytes):
    """Process an image from bytes."""
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get cached models
    interpreter = init_model()
    fashion_tagger = init_gpt4v()
    
    # Detect pose
    keypoints = detect_pose(interpreter, image)
    
    # Get clothing boxes
    boxes = get_clothing_boxes(keypoints)
    
    # Create temporary directory for cropped images
    os.makedirs("temp_crops", exist_ok=True)
    
    # Prepare all images first
    temp_files = []
    for clothing_type, box in boxes.items():
        if box is not None:
            # Crop the region
            cropped = crop_box_region(image, box)
            
            # Save cropped image
            temp_file = f"temp_crops/temp_{clothing_type}.jpg"
            cv2.imwrite(temp_file, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
            temp_files.append((clothing_type, temp_file))
    
    # Batch analyze all items
    analyses = {}
    if temp_files:
        image_paths = [f[1] for f in temp_files]
        results = fashion_tagger.tag_images_batch(image_paths)
        
        # Map results back to clothing types
        for (clothing_type, _), result in zip(temp_files, results):
            analyses[clothing_type] = result
    
    # Clean up
    for _, temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    if os.path.exists("temp_crops"):
        os.rmdir("temp_crops")
    
    return {
        'boxes': boxes,
        'analyses': analyses
    }

def main(context):
    """Appwrite function entry point."""
    try:
        # Get image from request
        if not context.req.files or 'image' not in context.req.files:
            return context.res.json({
                'success': False,
                'error': 'No image file provided'
            }, 400)
        
        # Get image file
        image_file = context.req.files['image']
        image_bytes = image_file.read()
        
        # Process image
        results = process_image_bytes(image_bytes)
        
        return context.res.json({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return context.res.json({
            'success': False,
            'error': str(e)
        }, 500) 