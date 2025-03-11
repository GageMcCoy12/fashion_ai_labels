import os
import base64
import json
import requests
from PIL import Image
from dotenv import load_dotenv

class GPT4VisionTagger:
    def __init__(self):
        print("Initializing GPT-4 Vision Tagger...")
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
    def encode_image(self, image_path):
        """Load image, resize to 512x512, and encode to base64."""
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to 512x512 maintaining aspect ratio
            img.thumbnail((512, 512), Image.Resampling.LANCZOS)
            
            # Create new image with padding if needed
            new_img = Image.new('RGB', (512, 512), (255, 255, 255))  # white background
            
            # Paste resized image in center
            offset = ((512 - img.size[0]) // 2, (512 - img.size[1]) // 2)
            new_img.paste(img, offset)
            
            # Save to bytes
            import io
            buffer = io.BytesIO()
            new_img.save(buffer, format='JPEG', quality=85)  # slightly reduced quality to save size
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def tag_images_batch(self, image_paths):
        """Analyze clothing items using GPT-4 Vision API."""
        results = []
        
        system_prompt = """You are a fashion expert analyzing clothing items in images.
        For each item, identify:
        1. Type of clothing/accessory
        2. Brand (if visible or recognizable, otherwise suggest a similar brand that matches the style)
        3. Color (be specific with shades, or suggest closest matching color)
        4. Material (if visible, otherwise suggest most likely material based on appearance)
        5. Aesthetic/style (e.g. casual, formal, streetwear, etc.)
        6. Extra details (Include the specific clothing item if you know it, otherwise include a close alternative that fits the item. Be specific. Must be a real clothing item. i.e. "Converse Chuck Taylor All Star, Jordan 1, etc.")

        Keep responses concise. If you're unsure about any attribute provide educated suggestions based on visual cues and fashion knowledge.

        If you know the exact clothing item, include it in the extra_details field. (i.e. "Converse Chuck Taylor All Star, Jordan 1, etc.")
        If you can't identify the exact clothing item, include a close alternative that fits the item. Be specific. Must be a real clothing item.

        Format your response as a JSON array with one object per item. Each object should have these fields:
        {
            "type": "",
            "brand": "",
            "color": "",
            "material": "",
            "aesthetic": "",
            "extra_details": "",
            "confidence": 0.0
        }

        Return the results in the same order as the images provided. Never use 'unknown' - instead suggest similar alternatives."""
        
        print(f"\nAnalyzing {len(image_paths)} items in a single batch...")
        
        # Encode all images
        base64_images = [self.encode_image(path) for path in image_paths]
        
        # Create content with all images
        content = [
            {
                "type": "text",
                "text": f"Analyze these {len(image_paths)} clothing items. For EACH item, return a JSON object with these fields: type, brand, color, material, aesthetic, extra_details. Format your response as a JSON array with one object per item, in the same order as the images. Be specific but concise."
            }
        ]
        
        # Add each image to content
        for i, base64_image in enumerate(base64_images):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": 2000  # Limit response length for batch processing
        }
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            response_data = response.json()
            
            if 'error' in response_data:
                print(f"Error: {response_data['error']}")
                # Return default items for all images
                return [
                    {
                        "type": "similar clothing item",
                        "brand": "similar brand",
                        "color": "neutral tone",
                        "material": "similar fabric",
                        "aesthetic": "versatile style",
                        "extra_details": "standard design and fit",
                        "confidence": 0.0
                    }
                ] * len(image_paths)
            
            # Extract the response content
            response_content = response_data['choices'][0]['message']['content']
            print(f"Raw GPT-4V response: {response_content}")
            
            # Clean up markdown code blocks if present
            if "```json" in response_content:
                response_content = response_content.split("```json")[1].split("```")[0].strip()
            elif "```" in response_content:
                response_content = response_content.split("```")[1].split("```")[0].strip()
            
            # Parse the JSON response
            try:
                analyses = json.loads(response_content)
                
                # Make sure we got an array of results
                if not isinstance(analyses, list):
                    analyses = [analyses]  # Convert single object to list
                
                # Pad with default items if we got fewer results than images
                while len(analyses) < len(image_paths):
                    analyses.append({
                        "type": "similar clothing item",
                        "brand": "similar brand",
                        "color": "neutral tone",
                        "material": "similar fabric",
                        "aesthetic": "versatile style",
                        "extra_details": "standard design and fit",
                        "confidence": 0.0
                    })
                
                # Add confidence scores and clean up results
                for analysis in analyses:
                    analysis["confidence"] = 0.9  # High confidence for GPT-4V
                    
                    # Print the analysis
                    print("\nGPT-4V Analysis:")
                    for key, value in analysis.items():
                        if key != "confidence":
                            print(f"{key.capitalize()}: {value}")
                    print("-" * 50)
                
                return analyses[:len(image_paths)]  # Return only as many results as we had images
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                print(f"Raw content: {response_content}")
                # Return default items for all images
                return [
                    {
                        "type": "similar clothing item",
                        "brand": "similar brand",
                        "color": "neutral tone",
                        "material": "similar fabric",
                        "aesthetic": "versatile style",
                        "extra_details": "standard design and fit",
                        "confidence": 0.0
                    }
                ] * len(image_paths)
        
        except Exception as e:
            print(f"Error: {e}")
            # Return default items for all images
            return [
                {
                    "type": "similar clothing item",
                    "brand": "similar brand",
                    "color": "neutral tone",
                    "material": "similar fabric",
                    "aesthetic": "versatile style",
                    "extra_details": "standard design and fit",
                    "confidence": 0.0
                }
            ] * len(image_paths) 