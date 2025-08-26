#This is a simple airline booking assistant which validates the booking destination, accepts booking details from customer and confirms booking.
#This project showcases the use of gradio for building a quick and simple ui as well as use of multi modal agents via OpenAI function calling.

# --- Imports and environment setup ---
import os
import json
from dotenv import load_dotenv  # Load environment variables from .env
from openai import OpenAI       # OpenAI API client
import gradio as gr             # Gradio for web UI

# Load environment variables
load_dotenv(override=True)

# Retrieve OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

# Model and OpenAI client
MODEL = "gpt-4o-mini"
openai = OpenAI()

# System prompt for the assistant's behavior
system_message = "You are a helpful assistant for an Airline called OrbitAI. "
system_message += "Give short, courteous answers, no more than 1 sentence. "
system_message += "Always be accurate.If someone wants to book a flight ticket, follow this order:"
system_message += "1. Ensure that you ask the destination first"
system_message += "2. Check if destination is a valid using get_valid_destination. If destination is invalid, the call get_valid_destination and get the list of valid destination. Present the valid list to customer and request them to pick destination from the list"
system_message += "3. If valid, then call the right function to get the ticket price and ask for booking confirmation before proceeding"
system_message += "4. Only if they confirm then ask for the booking details using book_flight_tickets."
system_message += "5.  Never ask for booking confirmation before checking if the destination is available. If you don't know the answer, say so."

# --- Set variables: this example uses predefined ticket prices for simplicity ---
ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}
destination_city = None  # Tracks the current destination

# Get the price for a given destination
def get_ticket_price(destination_city):
    print(f"Tool get_ticket_price called for {destination_city}")
    city = destination_city.lower()
    return ticket_prices.get(city, "Unknown")

# Validate if the destination is supported
def get_valid_destination(destination_city):
    print("Tool get_valid_destination called")
    valid_destination = list(ticket_prices.keys())
    if destination_city.lower() not in valid_destination:
        return (f"please select destination from {valid_destination}")

# Book flight tickets for the customer
def book_flight_tickets(**kwargs):
    ticket_count = kwargs.get("ticket_count")
    customer_name = kwargs.get("customer_name")
    destination_city = kwargs.get("destination_city")

    if ticket_count is None:
        return ("please enter the number of required tickets")

    ticket_count = int(ticket_count)

    if customer_name is None:
        return ("please enter the name of primary booking holder")

    confirmation = f"{ticket_count} tickets for {destination_city} booked for {customer_name}"
    return (confirmation)

# --- OpenAI function calling schemas ---
price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city. Call this whenever you need to know the ticket price, for example when a customer asks 'How much is a ticket to this city'",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False
    }
}

booking_details_function = {
    "name": "book_flight_tickets",
    "description": "Get details from customer for booking tickets. If someone asks you to book a ticket, for example when a customer says 'Book my ticket to this city'",
    "parameters": {
        "type": "object",
        "properties": {
            "ticket_count": {
                "type": "integer",
                "description": "The number of tickets the customer wants to book"
            },
            "customer_name": {
                "type": "string",
                "description": "The name of the customer to make the booking"
            },
            "destination_city":{
                "type":"string",
                "description": "The name of city customer wants to book tickets to"
        }
        },
        "required": ["ticket_count", "customer_name"],
        "additionalProperties": False  # Optional: Set to False to restrict properties to ticket_count and customer_name only
    }
}

valid_destination_function={
    "name": "get_valid_destination",
    "description": "Use this tool to identify the list of valid destination. Call this function to validate of customer has provided request to book flight for a supported destination",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The name of city customer wants to book tickets to"
            }
        },
        "required":["destination_city"],
        "additionalProperties": False
    }
}


# List of tool schemas for OpenAI function calling
tools = [
    {"type": "function", "function": price_function},
    {"type": "function", "function": booking_details_function},
    {"type": "function", "function": valid_destination_function}
]

# Main chat function for Gradio interface
def chat(message, history):
    """
    Handles the chat interaction with the user and OpenAI API.
    Calls tools/functions as needed based on the assistant's response.
    """
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    print("response:", response)

    # If the assistant requests a tool call, handle it
    if response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        messages.append(message)  # Append the assistant message that requested tools
        print("message:", message)
        for tool_call in message.tool_calls:
            response = handle_tool_call(tool_call)
            messages.append(response)

        print("response Last:\n", response)
        response = openai.chat.completions.create(model=MODEL, messages=messages)

    return response.choices[0].message.content

# Handle tool calls from OpenAI
def handle_tool_call(tool_call):
    """
    Executes the appropriate tool/function based on the tool_call from OpenAI.
    """
    print("tool_call", tool_call)
    global destination_city
    arguments = json.loads(tool_call.function.arguments)
    if tool_call.function.name == "get_ticket_price":
        city = arguments.get('destination_city')
        destination_city = city
        price = get_ticket_price(city)
        return {
            "role": "tool",
            "content": json.dumps({"destination_city": city, "price": price}),
            "tool_call_id": tool_call.id
        }
    elif tool_call.function.name == "book_flight_tickets":
        booking_destination = arguments.get('destination_city') or destination_city
        ticket_count = arguments.get('ticket_count', None)
        customer_name = arguments.get('customer_name', None)

        tool_call_response = book_flight_tickets(destination_city=booking_destination, ticket_count=ticket_count, customer_name=customer_name)
        return {
            "role": "tool",
            "content": tool_call_response,
            "tool_call_id": tool_call.id
        }
    elif tool_call.function.name == "get_valid_destination":
        booking_destination = arguments.get('destination_city')
        tool_call_response = get_valid_destination(booking_destination)
        return {
            "role": "tool",
            "content": tool_call_response,
            "tool_call_id": tool_call.id
        }

# Launch the Gradio chat interface
gr.ChatInterface(fn=chat, type="messages").launch()