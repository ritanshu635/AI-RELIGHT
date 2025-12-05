import base64
import os
from openai import OpenAI
from PIL import Image
import io
import numpy as np


class GPTRecommendationClient:
    def __init__(self, api_key=None):
        """
        Initialize GPT Vision API client
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env variable)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            print("Warning: OpenAI API key not provided. AI recommendations will use default suggestions.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
    
    def image_to_base64(self, image_array):
        """
        Convert numpy array to base64 string
        Args:
            image_array: numpy array (H, W, C) in uint8 format
        Returns:
            base64 encoded string
        """
        # Handle different input types
        if isinstance(image_array, np.ndarray):
            # Convert to PIL Image
            if image_array.dtype == np.float32 or image_array.dtype == np.float64:
                image_array = (image_array * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(image_array)
        elif isinstance(image_array, Image.Image):
            img = image_array
        else:
            raise ValueError("Input must be numpy array or PIL Image")
        
        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def get_lighting_recommendations(self, image_array, moods='', timeout=10):
        """
        Get 3 lighting recommendations from GPT-4o-mini
        Args:
            image_array: numpy array (H, W, C) or PIL Image
            moods: comma-separated mood/style keywords (e.g., "haunted,dramatic")
            timeout: API timeout in seconds
        Returns:
            List of 3 strings, each max 3 words
        """
        # If no API key, return default suggestions
        if not self.client:
            return self._get_default_suggestions()
        
        try:
            base64_image = self.image_to_base64(image_array)
            
            # Build prompt based on moods
            mood_context = ""
            if moods and moods.strip():
                mood_list = [m.strip() for m in moods.split(',') if m.strip()]
                if mood_list:
                    mood_context = f"\n\nUser wants a {', '.join(mood_list)} mood/style. Consider this when suggesting lighting."
            
            prompt_text = f"""Analyze this image and suggest 3 lighting styles that would enhance it.{mood_context}

Each suggestion must be EXACTLY 2-3 words. Format: one suggestion per line, no numbering, no punctuation.

Examples:
window light
haunted atmosphere
magic lit
warm golden
dramatic side
soft studio
cinematic glow
playful bright
mysterious shadow"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt_text
                            }
                        ]
                    }
                ],
                max_tokens=100,
                timeout=timeout
            )
            
            # Parse response
            content = response.choices[0].message.content
            suggestions = [s.strip() for s in content.split('\n') if s.strip()]
            
            # Ensure we have exactly 3 suggestions, each max 3 words
            suggestions = suggestions[:3]
            suggestions = [' '.join(s.split()[:3]) for s in suggestions]
            
            # Pad if less than 3
            while len(suggestions) < 3:
                suggestions.append("natural lighting")
            
            return suggestions
            
        except TimeoutError:
            print("GPT API timeout - returning default suggestions")
            return self._get_default_suggestions()
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Handle specific error types
            if "api_key" in error_msg or "authentication" in error_msg:
                print(f"GPT API Authentication Error: Invalid API key")
                raise ValueError("Invalid OpenAI API key. Please check your OPENAI_API_KEY.")
            elif "rate_limit" in error_msg:
                print(f"GPT API Rate Limit: {e}")
                print("Returning default suggestions. Please try again later.")
                return self._get_default_suggestions()
            elif "network" in error_msg or "connection" in error_msg:
                print(f"GPT API Network Error: {e}")
                print("Returning default suggestions. Check your internet connection.")
                return self._get_default_suggestions()
            else:
                print(f"GPT API Error: {e}")
                return self._get_default_suggestions()
    
    def _get_default_suggestions(self):
        """Return default lighting suggestions as fallback"""
        return [
            "window light",
            "magic lit",
            "soft studio"
        ]
