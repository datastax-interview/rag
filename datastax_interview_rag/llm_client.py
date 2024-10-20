from copy import deepcopy
import json
import openai
import os
from typing import Dict, Any, List

from enum import Enum
from typing import List, Dict

class MessageRole(Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"

class ChatHistory:
    def __init__(self):
        self.messages: List[Dict[str, str]] = []

    def add_message(self, role: MessageRole, content: str):
        self.messages.append({"role": role.value, "content": content})

    def get_messages(self) -> List[Dict[str, str]]:
        return self.messages

    def clear(self):
        self.messages.clear()
    
    def copy(self):
        return deepcopy(self)

class OpenAIClient:
    def __init__(self, model_name: str = "gpt4o-mini", system_prompt: str = None):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("No OpenAI API key found in environment variables. Please set OPENAI_API_KEY.")
        openai.api_key = self.api_key
        self.default_model = model_name
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            # can change this to whatever is most helpful. 
            self.system_prompt = "You are a helpful assistant."
        self.default_chat_history = ChatHistory()
        self.default_chat_history.add_message(MessageRole.SYSTEM, self.system_prompt)
        

        
    def generate_chat_completion(self, history: ChatHistory, model: str = None, **kwargs) -> str:
        """
        Generate a chat completion using the OpenAI API.

        Args:
            messages (List): A list of message dictionaries representing the conversation history.
            model (str, optional): The model to use for generation. Defaults to self.default_model if not specified.
            **kwargs: Additional keyword arguments to pass to the OpenAI API.

        Returns:
            str: The generated response content from the assistant.

        Raises:
            openai.error.OpenAIError: If there's an error in the API call.
        """
        model = model or self.default_model
        response = openai.chat.completions.create(
            model=model,
            messages=history.get_messages(),
            **kwargs
        )
        return response.choices[0].message.content

    def generate_llm_completion(self, user_prompt: str, model: str = None, **kwargs) -> str:
        input_prompt = self.default_chat_history.copy()
        input_prompt.add_message(MessageRole.USER, user_prompt)
        model = model or self.default_model
        response = openai.chat.completions.create(
            model=model,
            messages=input_prompt.get_messages(),
            **kwargs
        )
        return response.choices[0].message.content

    def generate_json_completion(self, user_prompt: str, model: str = None, **kwargs) -> Dict[str, Any]:
        input_prompt = self.default_chat_history.copy()
        input_prompt.add_message(MessageRole.USER, user_prompt)
        model = model or self.default_model
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=input_prompt.get_messages(),
                response_format={"type": "json_object"},
                **kwargs
            )
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(f"Raw response content: {response.choices[0].message.content}")
            return {"error": "Failed to parse JSON response"}
        except openai.OpenAIError as e:
            print(f"OpenAI API error: {e}")
            return {"error": "OpenAI API error occurred"}
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {"error": "An unexpected error occurred"}

