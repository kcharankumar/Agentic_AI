# pip install groq requests python-dotenv

# Import os to access environment variables
import os

# Import json to convert Python objects to JSON and vice versa
import json

# Import requests to make HTTP requests to external APIs
import requests

# Import Groq to connect Python with Groq LLM
from groq import Groq

# Import load_dotenv to load API keys from the .env file
from dotenv import load_dotenv


# Load API keys from the .env file
load_dotenv()

# Create a Groq client to communicate with the LLM
client = Groq(
    # Retrieve the Groq API key from environment variables
    api_key=os.getenv("GROQ_API_KEY")
)

# Retrieve the Google Places API key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")

# Specify the LLM model that Groq should use
MODEL = "openai/gpt-oss-120b"


# ---------------------------------------
# TOOL 1: SEARCH RESTAURANTS
# ---------------------------------------

# Define a function that accepts the restaurant search query
def search_restaurants(query):

    # Specify the Google Places Text Search API endpoint
    url = "https://places.googleapis.com/v1/places:searchText"

    # Define the HTTP headers required to communicate with Google
    headers = {
        # Specify that the request data will be in JSON format
        "Content-Type": "application/json",

        # Pass the Google Places API key for authentication
        "X-Goog-Api-Key": GOOGLE_API_KEY,

        # Specify which restaurant details should be returned
        "X-Goog-FieldMask": (
            # Retrieve the unique restaurant identifier
            "places.id,"

            # Retrieve the restaurant name
            "places.displayName,"

            # Retrieve the restaurant address
            "places.formattedAddress,"

            # Retrieve the restaurant rating
            "places.rating,"

            # Retrieve the number of customer reviews
            "places.userRatingCount,"

            # Retrieve the Google Maps URL
            "places.googleMapsUri"
        )
    }

    # Prepare the search query and maximum number of results
    data = {
        # Pass the restaurant search query to Google
        "textQuery": query,

        # Request a maximum of 10 restaurant results
        "maxResultCount": 10
    }

    # Send an HTTP POST request to Google Places API
    response = requests.post(
        # Specify the API endpoint
        url,

        # Pass the request headers
        headers=headers,

        # Pass the search data as JSON
        json=data,

        # Set the maximum request waiting time to 30 seconds
        timeout=30
    )

    # Raise an exception if Google returns an HTTP error
    response.raise_for_status()

    # Convert the response into JSON and return the restaurant list
    return response.json().get("places", [])


# ---------------------------------------
# TOOL DEFINITIONS
# ---------------------------------------

# Define the tools that Groq can use during the conversation
tools = [
    {
        # Specify that the tool is a function
        "type": "function",

        # Define the function details for the LLM
        "function": {
            # Specify the function name available to Groq
            "name": "search_restaurants",

            # Explain when the LLM should use this function
            "description": (
                "Search for real restaurants using Google Places. "
                "Use this whenever the customer asks to find, "
                "show, or discover restaurants."
            ),

            # Define the input parameters required by the function
            "parameters": {
                # Specify that the function arguments must be a JSON object
                "type": "object",

                # Define the properties accepted by the function
                "properties": {
                    # Define the restaurant search query
                    "query": {
                        # Specify that the query must be a string
                        "type": "string",

                        # Explain what information the query should contain
                        "description": (
                            "Restaurant type, cuisine and location. "
                            "Example: Indian restaurants in RR Nagar, Bangalore"
                        )
                    }
                },

                # Specify that the query parameter is mandatory
                "required": ["query"]
            }
        }
    }
]


# ---------------------------------------
# TOOL REGISTRY
# ---------------------------------------

# Map the function name provided to Groq to the actual Python function
available_functions = {
    # Associate search_restaurants with its Python implementation
    "search_restaurants": search_restaurants
}


# ---------------------------------------
# AGENT SYSTEM PROMPT
# ---------------------------------------

SYSTEM_PROMPT = """
You are an intelligent Restaurant Discovery AI Agent.

Your goal is to help customers discover restaurants.

INSTRUCTIONS:

1. Understand what type of restaurant the customer wants.
2. Ask for the location if it is not provided.
3. Ask about cuisine if it is not clear.
4. Use search_restaurants to find actual restaurants.
5. Display restaurant names, addresses and ratings
   returned by Google Places.
6. Never invent restaurant names or ratings.
7. Allow the customer to select a restaurant.
8. Remember the selected restaurant in the conversation.
9. Be polite and behave like a knowledgeable
   restaurant consultant.
10. Do not claim that a restaurant has a particular
    menu or food item unless verified by a reliable source.

For this stage, your responsibility is restaurant discovery.
Do not invent menus, prices or payment information.
"""


# ---------------------------------------
# MAIN AGENT PROGRAM
# ---------------------------------------

# Create the conversation history and initialize it with the system prompt
messages = [
    {
        # Identify this message as the agent's system instructions
        "role": "system",

        # Pass the system prompt to the LLM
        "content": SYSTEM_PROMPT
    }
]

# Display the welcome message to the customer
print("\nWelcome to Restaurant AI Agent!")

# Ask the customer how the agent can help
print("How can I help you today?")

# Inform the customer how to terminate the application
print("Type exit to stop.")

# Start an infinite loop to continuously accept customer requests
while True:

    # Accept the customer's input from the keyboard
    user_input = input("\nCustomer: ")

    # Check whether the customer wants to exit the application
    if user_input.lower() == "exit":

        # Display the exit confirmation message
        print("Thank you! Visit again.")

        # Terminate the while loop
        break

    # Add the customer's input to the conversation history
    messages.append({
        # Identify this message as coming from the user
        "role": "user",

        # Store the customer's actual request
        "content": user_input
    })

    # Start exception handling to manage runtime errors
    try:

        # Allow up to six LLM and tool execution iterations for one request
        for iteration in range(6):

            # Send the conversation and available tools to the Groq LLM
            response = client.chat.completions.create(

                # Specify the LLM model to be used
                model=MODEL,

                # Send the complete conversation history
                messages=messages,

                # Provide the available Python tools to the LLM
                tools=tools,

                # Allow the LLM to decide whether a tool is required
                tool_choice="auto",

                # Use a relatively low temperature for predictable responses
                temperature=0.2
            )

            # Extract the assistant's response from the Groq API result
            assistant_message = response.choices[0].message

            # Add the assistant's response to the conversation history
            messages.append(assistant_message)

            # Check whether the LLM requested a tool
            if not assistant_message.tool_calls:

                # Display the LLM's response to the customer
                print(
                    "\nRestaurant Agent:",
                    assistant_message.content
                )

                # Stop the current agent execution loop
                break

            # Execute each tool requested by the LLM
            for tool_call in assistant_message.tool_calls:

                # Extract the requested Python function name
                function_name = tool_call.function.name

                # Convert the LLM's JSON arguments into a Python dictionary
                arguments = json.loads(
                    tool_call.function.arguments
                )

                # Display the name of the function being executed
                print("\nExecuting:", function_name)

                # Handle errors that might occur during function execution
                try:

                    # Retrieve the corresponding Python function from the registry
                    function = available_functions[
                        function_name
                    ]

                    # Execute the function using the arguments provided by the LLM
                    result = function(**arguments)

                # Catch any exception raised during tool execution
                except Exception as error:

                    # Store the error as a dictionary to return to the LLM
                    result = {
                        "error": str(error)
                    }

                # Return the tool execution result to the LLM
                messages.append({
                    # Identify this message as a tool response
                    "role": "tool",

                    # Associate the response with the original tool call
                    "tool_call_id": tool_call.id,

                    # Specify which function generated this result
                    "name": function_name,

                    # Convert the function result into JSON text
                    "content": json.dumps(result)
                })

        # Execute this block if all six iterations finish without a break
        else:

            # Inform the customer that the request needs another attempt
            print("Agent: Please try a more specific request.")

    # Catch errors occurring anywhere in the main agent execution
    except Exception as error:

        # Display the error message for troubleshooting
        print("Error:", error)
