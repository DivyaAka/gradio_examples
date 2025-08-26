
# Import required libraries
import os
import json
from dotenv import load_dotenv  # For loading environment variables from .env file
from openai import OpenAI       # OpenAI API client
import gradio as gr             # Gradio for building web UI

# Load environment variables from .env file
load_dotenv(override=True)


# Retrieve OpenAI API key from environment
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")


# Initialize OpenAI client
openai = OpenAI()

# System prompt for the AI assistant
system_message = (
    "you are a technical tutor who can answer technical questions in plain english."
    "if someone asks a tecnical question you can explain it in simple terms which can be understood by a non technical person."
    "you also provide easy examples of usage and the language of the code"
)


# Chat function to interact with the OpenAI API
# message: latest user message
# history: list of previous messages (as dicts)
# model: selected OpenAI model
def chat(message, history, model):
    # Build the conversation history for the API
    messages = [
        {"role": "system", "content": system_message}
    ] + history + [
        {"role": "user", "content": message}
    ]
    # Get response from OpenAI
    response = openai.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


# Dropdown for model selection in the Gradio UI
model_dropdown = gr.Dropdown(
    choices=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    value="gpt-4o-mini",
    label="select model"
)

# Launch the Gradio chat interface
gr.ChatInterface(
    fn=chat,
    type="messages",
    additional_inputs=[model_dropdown]
).launch()