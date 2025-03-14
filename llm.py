"""
Text generation module for CSM Speech-to-Speech interface.
Uses local LLM models for generating responses.
"""

import os
import torch
from typing import List, Dict, Optional, Union, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("llm")

# Default models to try (in order of preference)
DEFAULT_MODELS = [
    "meta-llama/Llama-3-8B-Instruct",  # First choice - good quality, reasonable size
    "meta-llama/Llama-3-8B",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Fallback for low-resource systems
]

class LLMEngine:
    """Text generation using a local LLM."""
    
    def __init__(self, 
                 model_name: Optional[str] = None,
                 use_4bit: bool = True,
                 device: Optional[str] = None,
                 max_new_tokens: int = 256,
                 temperature: float = 0.7):
        """
        Initialize the LLM engine.
        
        Args:
            model_name: Name of the LLM model to use. If None, will try DEFAULT_MODELS.
            use_4bit: Whether to use 4-bit quantization to reduce memory usage.
            device: Device to run the model on ('cuda', 'cpu', etc.).
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Temperature for text generation.
        """
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Find an available model
        self.model_name = self._find_available_model(model_name)
        logger.info(f"Loading LLM model: {self.model_name}")
        
        # Configure quantization
        quantization_config = None
        if use_4bit and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        
        # Load model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                quantization_config=quantization_config
            )
            
            # Configure tokenizer for chat
            if not self.tokenizer.chat_template and "llama" in self.model_name.lower():
                logger.info("Setting default chat template for Llama model")
                self.tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + '\n' }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + '\n' }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + '\n' }}\n{% endif %}\n{% endfor %}\n{% if add_generation_prompt %}\n{{ '<|assistant|>\n' }}\n{% endif %}"
            
            logger.info("LLM model loaded successfully.")
            
            # Initialize conversation history
            self.conversation_history = []
            self.system_message = "You are a helpful, friendly AI assistant. Keep your responses concise and conversational."
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise ValueError(f"Failed to load model {self.model_name}: {str(e)}")
    
    def _find_available_model(self, model_name: Optional[str]) -> str:
        """Find an available LLM model to use."""
        if model_name is not None:
            return model_name
            
        # Try the default models in order
        from huggingface_hub import list_models
        
        available_models = set()
        try:
            # Try to get list of cached models
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            if os.path.exists(cache_dir):
                available_models = set(os.listdir(cache_dir))
        except Exception:
            pass
            
        # Check if any of our preferred models are available locally
        for model in DEFAULT_MODELS:
            model_id = model.split("/")[-1].lower()
            if any(model_id in cached_model.lower() for cached_model in available_models):
                logger.info(f"Found cached model: {model}")
                return model
                
        # If no cached models, use the first default
        logger.info(f"No cached models found, using: {DEFAULT_MODELS[0]}")
        return DEFAULT_MODELS[0]
    
    def generate_response(self, 
                        user_input: str, 
                        system_message: Optional[str] = None) -> str:
        """
        Generate a response to user input.
        
        Args:
            user_input: User's text input
            system_message: Optional system message to override the default
            
        Returns:
            Generated response text
        """
        if system_message is not None:
            self.system_message = system_message
            
        # Format messages for the model
        messages = [{"role": "system", "content": self.system_message}]
        
        # Add conversation history
        for message in self.conversation_history:
            messages.append(message)
            
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        # Generate response
        inputs = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            return_tensors="pt"
        ).to(self.device)
        
        outputs = self.model.generate(
            inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode and clean up response
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        response = response.strip()
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Keep conversation history to a reasonable size
        if len(self.conversation_history) > 10:
            # Remove oldest turn (2 messages)
            self.conversation_history = self.conversation_history[2:]
            
        return response
    
    def reset_conversation(self):
        """Clear the conversation history."""
        self.conversation_history = []
        
    def update_system_message(self, system_message: str):
        """Update the system message."""
        self.system_message = system_message


class FallbackLLM:
    """Enhanced rule-based fallback when no LLM is available."""
    
    def __init__(self):
        self.greeting_inputs = ["hello", "hi", "hey", "greetings"]
        self.farewell_inputs = ["goodbye", "bye", "see you", "farewell"]
        self.question_patterns = {
            "how are you": ["I'm doing well, thank you for asking!", "I'm functioning perfectly. How about you?", "All systems operational! How can I help you today?"],
            "your name": ["I'm CSM, a conversational speech model.", "You can call me CSM, I'm here to chat with you.", "I'm an AI assistant powered by CSM technology."],
            "about you": ["I'm an AI assistant that can chat with you using a voice interface.", "I'm a conversational AI using speech synthesis to communicate.", "I'm running on CSM, a conversational speech model designed for natural voice interactions."],
            "what can you do": ["I can have conversations, answer questions, and provide information.", "I'm able to chat with you in a natural voice and help with various topics.", "I can assist with information, casual conversation, and more - all through speech."],
            "help": ["I'd be happy to help! What do you need assistance with?", "I'm here to assist you. What would you like help with?", "Sure thing! How can I assist you today?"],
            "tell me about": ["That's an interesting topic! I can share what I know about it.", "I'd be happy to discuss that with you.", "That's a fascinating subject. Here's what I know."],
            "weather": ["I don't have access to current weather data, but I hope it's pleasant where you are!", "I can't check the weather in real-time, but I hope you're having nice weather.", "While I can't access current weather information, I'd be happy to discuss other topics."],
            "joke": ["Why don't scientists trust atoms? Because they make up everything!", "What did one wall say to the other wall? I'll meet you at the corner!", "Why did the scarecrow win an award? Because he was outstanding in his field!"],
            "thank": ["You're welcome! How else can I help?", "Happy to assist! Is there anything else you'd like to talk about?", "My pleasure! What else would you like to discuss?"]
        }
        
        self.general_responses = [
            "That's interesting. Tell me more about that.",
            "I understand. What would you like to discuss next?",
            "Thanks for sharing. Is there anything specific you'd like to know?",
            "I appreciate your input. How can I assist you further?",
            "Interesting perspective. Would you like to explore this topic more?",
            "I'm here to chat about whatever interests you."
        ]
        
        self.conversation_history = []
        self.last_response = ""
        
    def generate_response(self, user_input: str, system_message: Optional[str] = None) -> str:
        """Generate a more varied rule-based response."""
        import random
        user_input = user_input.lower().strip()
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Handle greetings
        if any(greeting in user_input for greeting in self.greeting_inputs):
            response = random.choice([
                "Hello there! How are you today?",
                "Hi! Nice to chat with you. What's on your mind?",
                "Hey! How can I help you today?",
                "Greetings! What would you like to talk about?"
            ])
            self.last_response = response
            self.conversation_history.append({"role": "assistant", "content": response})
            return response
        
        # Handle farewells
        if any(farewell in user_input for farewell in self.farewell_inputs):
            response = random.choice([
                "Goodbye! It was nice talking with you.",
                "Bye for now! Feel free to chat again anytime.",
                "Take care! It was a pleasure chatting with you.",
                "Farewell! Have a wonderful day!"
            ])
            self.last_response = response
            self.conversation_history.append({"role": "assistant", "content": response})
            return response
            
        # Look for pattern matches
        for pattern, responses in self.question_patterns.items():
            if pattern in user_input:
                response = random.choice(responses)
                self.last_response = response
                self.conversation_history.append({"role": "assistant", "content": response})
                return response
                
        # If nothing else matched, use a general response
        # Make sure we don't repeat the last response
        available_responses = [r for r in self.general_responses if r != self.last_response]
        response = random.choice(available_responses)
        self.last_response = response
        self.conversation_history.append({"role": "assistant", "content": response})
        return response
        
    def reset_conversation(self):
        """Clear the conversation history."""
        self.conversation_history = []


def test_llm():
    """Test the LLM functionality."""
    print("Testing LLM Response Generation")
    
    try:
        # Try to initialize the LLM
        llm = LLMEngine()
        
        # Generate some responses
        test_inputs = [
            "Hello! How are you?",
            "Tell me about the weather today.",
            "What's your favorite color?",
            "Tell me a short joke."
        ]
        
        for user_input in test_inputs:
            print(f"\nUser: {user_input}")
            response = llm.generate_response(user_input)
            print(f"AI: {response}")
            
    except Exception as e:
        print(f"Error loading LLM: {str(e)}")
        print("Falling back to rule-based responses.")
        
        # Use fallback
        fallback = FallbackLLM()
        test_inputs = [
            "hello",
            "how are you",
            "who are you",
            "goodbye"
        ]
        
        for user_input in test_inputs:
            print(f"\nUser: {user_input}")
            response = fallback.generate_response(user_input)
            print(f"AI: {response}")


if __name__ == "__main__":
    test_llm()
