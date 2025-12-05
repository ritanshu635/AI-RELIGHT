import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from PIL import Image
import base64
import io
from gpt_recommendations import GPTRecommendationClient


class TestGPTRecommendationClient(unittest.TestCase):
    """Unit tests for GPTRecommendationClient class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Use a dummy API key for testing
        self.api_key = "test-api-key-12345"
        self.client = GPTRecommendationClient(api_key=self.api_key)
    
    def create_test_image(self, height=100, width=100):
        """Helper method to create a test image"""
        return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    def test_initialization_with_api_key(self):
        """Test client initialization with provided API key"""
        client = GPTRecommendationClient(api_key="test-key")
        self.assertEqual(client.api_key, "test-key")
    
    def test_initialization_without_api_key(self):
        """Test client initialization fails without API key"""
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaises(ValueError) as context:
                GPTRecommendationClient()
            self.assertIn("API key not provided", str(context.exception))
    
    def test_image_to_base64_numpy_uint8(self):
        """Test base64 encoding of uint8 numpy array"""
        test_image = self.create_test_image(50, 50)
        base64_str = self.client.image_to_base64(test_image)
        
        # Verify it's a valid base64 string
        self.assertIsInstance(base64_str, str)
        self.assertGreater(len(base64_str), 0)
        
        # Verify we can decode it back
        decoded = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(decoded))
        self.assertEqual(img.size, (50, 50))
    
    def test_image_to_base64_numpy_float32(self):
        """Test base64 encoding of float32 numpy array"""
        test_image = np.random.rand(50, 50, 3).astype(np.float32)
        base64_str = self.client.image_to_base64(test_image)
        
        # Verify it's a valid base64 string
        self.assertIsInstance(base64_str, str)
        self.assertGreater(len(base64_str), 0)
    
    def test_image_to_base64_pil_image(self):
        """Test base64 encoding of PIL Image"""
        test_array = self.create_test_image(50, 50)
        test_image = Image.fromarray(test_array)
        base64_str = self.client.image_to_base64(test_image)
        
        # Verify it's a valid base64 string
        self.assertIsInstance(base64_str, str)
        self.assertGreater(len(base64_str), 0)
    
    @patch('gpt_recommendations.OpenAI')
    def test_get_lighting_recommendations_success(self, mock_openai):
        """Test successful API call with mocked response"""
        # Mock the API response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "warm golden hour\ndramatic side lighting\nsoft studio light"
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Create new client with mocked OpenAI
        client = GPTRecommendationClient(api_key="test-key")
        
        # Test
        test_image = self.create_test_image()
        suggestions = client.get_lighting_recommendations(test_image)
        
        # Verify
        self.assertEqual(len(suggestions), 3)
        self.assertEqual(suggestions[0], "warm golden hour")
        self.assertEqual(suggestions[1], "dramatic side lighting")
        self.assertEqual(suggestions[2], "soft studio light")
    
    @patch('gpt_recommendations.OpenAI')
    def test_get_lighting_recommendations_truncates_long_suggestions(self, mock_openai):
        """Test that suggestions longer than 3 words are truncated"""
        # Mock response with long suggestions
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "warm golden hour sunset lighting\ndramatic side\nsoft"
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        client = GPTRecommendationClient(api_key="test-key")
        test_image = self.create_test_image()
        suggestions = client.get_lighting_recommendations(test_image)
        
        # Verify truncation
        self.assertEqual(suggestions[0], "warm golden hour")  # Truncated to 3 words
        self.assertEqual(suggestions[1], "dramatic side")
        self.assertEqual(suggestions[2], "soft")
    
    @patch('gpt_recommendations.OpenAI')
    def test_get_lighting_recommendations_pads_short_response(self, mock_openai):
        """Test that responses with less than 3 suggestions are padded"""
        # Mock response with only 1 suggestion
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "warm golden hour"
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        client = GPTRecommendationClient(api_key="test-key")
        test_image = self.create_test_image()
        suggestions = client.get_lighting_recommendations(test_image)
        
        # Verify padding
        self.assertEqual(len(suggestions), 3)
        self.assertEqual(suggestions[0], "warm golden hour")
        self.assertEqual(suggestions[1], "natural lighting")  # Padded
        self.assertEqual(suggestions[2], "natural lighting")  # Padded
    
    @patch('gpt_recommendations.OpenAI')
    def test_get_lighting_recommendations_timeout(self, mock_openai):
        """Test timeout handling"""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = TimeoutError("Request timeout")
        mock_openai.return_value = mock_client
        
        client = GPTRecommendationClient(api_key="test-key")
        test_image = self.create_test_image()
        suggestions = client.get_lighting_recommendations(test_image)
        
        # Should return default suggestions
        self.assertEqual(len(suggestions), 3)
        self.assertEqual(suggestions, ["warm golden light", "dramatic side lighting", "soft studio light"])
    
    @patch('gpt_recommendations.OpenAI')
    def test_get_lighting_recommendations_rate_limit(self, mock_openai):
        """Test rate limit error handling"""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("rate_limit exceeded")
        mock_openai.return_value = mock_client
        
        client = GPTRecommendationClient(api_key="test-key")
        test_image = self.create_test_image()
        suggestions = client.get_lighting_recommendations(test_image)
        
        # Should return default suggestions
        self.assertEqual(suggestions, ["warm golden light", "dramatic side lighting", "soft studio light"])
    
    @patch('gpt_recommendations.OpenAI')
    def test_get_lighting_recommendations_network_error(self, mock_openai):
        """Test network error handling"""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("network connection failed")
        mock_openai.return_value = mock_client
        
        client = GPTRecommendationClient(api_key="test-key")
        test_image = self.create_test_image()
        suggestions = client.get_lighting_recommendations(test_image)
        
        # Should return default suggestions
        self.assertEqual(suggestions, ["warm golden light", "dramatic side lighting", "soft studio light"])
    
    @patch('gpt_recommendations.OpenAI')
    def test_get_lighting_recommendations_invalid_api_key(self, mock_openai):
        """Test invalid API key error handling"""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("api_key authentication failed")
        mock_openai.return_value = mock_client
        
        client = GPTRecommendationClient(api_key="test-key")
        test_image = self.create_test_image()
        
        # Should raise ValueError for invalid API key
        with self.assertRaises(ValueError) as context:
            client.get_lighting_recommendations(test_image)
        self.assertIn("Invalid OpenAI API key", str(context.exception))
    
    def test_default_suggestions(self):
        """Test default suggestions fallback"""
        defaults = self.client._get_default_suggestions()
        self.assertEqual(len(defaults), 3)
        self.assertEqual(defaults[0], "warm golden light")
        self.assertEqual(defaults[1], "dramatic side lighting")
        self.assertEqual(defaults[2], "soft studio light")


if __name__ == '__main__':
    unittest.main()
