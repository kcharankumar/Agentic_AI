
import os
import json
import requests
from datetime import date
from dotenv import load_dotenv
from groq import Groq

# --------------------------------------------------
# 1. LOAD API CREDENTIALS
# --------------------------------------------------

# Load credentials from the .env file.
load_dotenv()

# Read the Groq API key.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Read the SerpApi key.
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Validate the API credentials.
if not GROQ_API_KEY or not SERPAPI_KEY:
    raise ValueError("Please configure both API keys in .env.")

# Create the Groq client.
client = Groq(api_key=GROQ_API_KEY)

# Select the Groq model.
MODEL = "openai/gpt-oss-120b"

# Define the SerpApi endpoint.
SERPAPI_URL = "https://serpapi.com/search.json"

# Store the actual flight offers returned by the API.
flight_offers = []


# --------------------------------------------------
# 2. DYNAMIC AIRPORT / CITY LOOKUP
# --------------------------------------------------

def get_airport_code(location_name):

    # Prepare the Google Flights autocomplete parameters.
    params = {
        "engine": "google_flights_autocomplete",
        "q": location_name,
        "api_key": SERPAPI_KEY,
        "gl": "in",
        "hl": "en"
    }

    # Call the live autocomplete API.
    response = requests.get(
        SERPAPI_URL,
        params=params,
        timeout=30
    )

    # Raise an exception for an unsuccessful API response.
    response.raise_for_status()

    # Convert the response into a Python dictionary.
    data = response.json()

    # Check whether the API returned an error.
    if data.get("error"):
        raise ValueError(data["error"])

    # Extract the location suggestions.
    suggestions = data.get("suggestions", [])

    # Stop if no location suggestions are available.
    if not suggestions:
        raise ValueError(f"No location found for {location_name}.")

    # Prefer a city result, which covers its associated airports.
    city_results = [
        item for item in suggestions
        if item.get("type") == "city" and item.get("id")
    ]

    # Return the city Knowledge Graph identifier when available.
    if city_results:
        return city_results[0]["id"]

    # Otherwise, use an airport identifier from the suggestions.
    for item in suggestions:
        airports = item.get("airports", [])

        if airports:
            return airports[0]["id"]

    # Raise an error if no usable location identifier is available.
    raise ValueError(f"No usable airport found for {location_name}.")


# --------------------------------------------------
# 3. SEARCH FLIGHTS USING LIVE API DATA
# --------------------------------------------------

def search_flights(
    origin: str,
    destination: str,
    departure_date: str,
    adults: int,
    cabin: str
):

    # Allow this function to update the global flight results.
    global flight_offers

    # Clear the previous flight results.
    flight_offers = []

    # Validate the departure date format.
    try:
        travel_date = date.fromisoformat(departure_date)
    except ValueError:
        return {
            "error": "Date must be in YYYY-MM-DD format."
        }

    # Reject departure dates in the past.
    if travel_date < date.today():
        return {
            "error": "Departure date cannot be in the past."
        }

    # Validate the number of adult passengers.
    if not isinstance(adults, int) or not 1 <= adults <= 9:
        return {
            "error": "Adult passenger count must be between 1 and 9."
        }

    # Define supported travel classes.
    cabin_classes = {
        "ECONOMY": 1,
        "PREMIUM ECONOMY": 2,
        "PREMIUM_ECONOMY": 2,
        "BUSINESS": 3,
        "FIRST": 4
    }

    # Normalize the cabin class.
    cabin = cabin.upper().replace("-", " ").strip()

    # Validate the cabin class.
    if cabin not in cabin_classes:
        return {
            "error": "Invalid cabin class."
        }

    # Dynamically resolve the departure location.
    origin_id = get_airport_code(origin)

    # Dynamically resolve the destination location.
    destination_id = get_airport_code(destination)

    # Reject identical origin and destination identifiers.
    if origin_id == destination_id:
        return {
            "error": "Origin and destination must be different."
        }

    # Prepare the live Google Flights search parameters.
    params = {
        "engine": "google_flights",
        "api_key": SERPAPI_KEY,
        "departure_id": origin_id,
        "arrival_id": destination_id,
        "outbound_date": departure_date,
        "type": 2,
        "adults": adults,
        "travel_class": cabin_classes[cabin],
        "currency": "INR",
        "gl": "in",
        "hl": "en",
        "no_cache": "true"
    }

    # Display the search operation.
    print("\nSearching live flight information...")

    # Call the Google Flights API.
    response = requests.get(
        SERPAPI_URL,
        params=params,
        timeout=60
    )

    # Raise an exception if the API request fails.
    response.raise_for_status()

    # Convert the API response into a Python dictionary.
    data = response.json()

    # Check whether SerpApi returned an error.
    if data.get("error"):
        return {
            "error": data["error"]
        }

    # Extract the best flight results.
    best_flights = data.get("best_flights", [])

    # Extract the remaining flight results.
    other_flights = data.get("other_flights", [])

    # Combine the available flight categories.
    all_offers = best_flights + other_flights

    # Check whether any flights were returned.
    if not all_offers:
        return {
            "message": "No flights found for these search criteria."
        }

    # Keep only offers containing flight segments.
    flight_offers = [
        offer for offer in all_offers
        if offer.get("flights")
    ]

    # Create a list for the formatted flight results.
    displayed_flights = []

    # Process each flight offer.
    for index, offer in enumerate(flight_offers, start=1):

        # Extract the flight segments.
        segments = offer.get("flights", [])

        # Get the first segment.
        first_segment = segments[0]

        # Get the last segment.
        last_segment = segments[-1]

        # Format the flight details.
        flight = {
            "flight_number": index,
            "airline": ", ".join(
                segment.get("airline", "Unknown")
                for segment in segments
            ),
            "flight_codes": ", ".join(
                segment.get("flight_number", "Unknown")
                for segment in segments
            ),
            "departure_airport": first_segment.get(
                "departure_airport", {}
            ).get("name", "Unknown"),
            "departure_code": first_segment.get(
                "departure_airport", {}
            ).get("id", "Unknown"),
            "departure_time": first_segment.get(
                "departure_airport", {}
            ).get("time", "Unknown"),
            "arrival_airport": last_segment.get(
                "arrival_airport", {}
            ).get("name", "Unknown"),
            "arrival_code": last_segment.get(
                "arrival_airport", {}
            ).get("id", "Unknown"),
            "arrival_time": last_segment.get(
                "arrival_airport", {}
            ).get("time", "Unknown"),
            "duration_minutes": offer.get("total_duration"),
            "stops": len(segments) - 1,
            "price": offer.get("price"),
            "currency": "INR"
        }

        # Add the formatted flight to the result list.
        displayed_flights.append(flight)

    # Return the actual flight information to the LLM.
    return {
        "origin": origin,
        "destination": destination,
        "departure_date": departure_date,
        "adults": adults,
        "cabin": cabin,
        "flight_count": len(displayed_flights),
        "flights": displayed_flights
    }


# --------------------------------------------------
# 4. DEFINE THE LLM TOOL
# --------------------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": (
                "Search live one-way flight information using "
                "Google Flights through SerpApi. Call only after "
                "collecting all required travel details."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": "Departure city or airport."
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination city or airport."
                    },
                    "departure_date": {
                        "type": "string",
                        "description": "Travel date in YYYY-MM-DD format."
                    },
                    "adults": {
                        "type": "integer",
                        "description": "Number of adult passengers."
                    },
                    "cabin": {
                        "type": "string",
                        "enum": [
                            "ECONOMY",
                            "PREMIUM ECONOMY",
                            "BUSINESS",
                            "FIRST"
                        ]
                    }
                },
                "required": [
                    "origin",
                    "destination",
                    "departure_date",
                    "adults",
                    "cabin"
                ]
            }
        }
    }
]


# --------------------------------------------------
# 5. TOOL REGISTRY
# --------------------------------------------------

# Connect the tool name to the actual Python function.
available_functions = {
    "search_flights": search_flights
}


# --------------------------------------------------
# 6. AGENT SYSTEM PROMPT
# --------------------------------------------------

SYSTEM_PROMPT = """
You are an intelligent Flight Search AI Agent.

YOUR RESPONSIBILITIES:

1. Welcome the user.
2. Understand the user's flight search request.
3. Collect departure city.
4. Collect destination city.
5. Collect departure date.
6. Collect number of adult passengers.
7. Collect cabin class.
8. Ask for missing details one question at a time.
9. Convert clear natural-language dates to YYYY-MM-DD.
10. Call search_flights after collecting all required details.
11. Read the actual flight results returned by the tool.
12. Display the available flights in a numbered list.

DISPLAY THESE DETAILS WHEN AVAILABLE:

- Airline name
- Flight number
- Departure airport and time
- Arrival airport and time
- Journey duration
- Number of stops
- Ticket price

IMPORTANT RULES:

- Never invent flight information or prices.
- Never display sample or hardcoded flight information.
- Display only information returned by the search tool.
- Do not call the tool before all required details are known.
- If no flights are found, explain that clearly.
- Do not claim that a flight has been booked.
- This application only searches and displays flights.
- Be polite and conversational.
"""


# --------------------------------------------------
# 7. MAIN AGENT EXECUTION
# --------------------------------------------------

# Initialize the conversation with the system prompt.
messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
]

# Display the application heading.
print("\n======================================")
print("       FLIGHT SEARCH AI AGENT")
print("======================================")

# Explain how to exit the application.
print("\nEnter your flight requirements in natural language.")
print("Type 'exit' to stop.")


# Start the user conversation loop.
while True:

    # Read the user's natural-language prompt.
    user_input = input("\nYou: ").strip()

    # Check whether the user wants to exit.
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("\nAgent: Thank you. Goodbye!")
        break

    # Ignore empty messages.
    if not user_input:
        continue

    # Add the user message to the conversation.
    messages.append({
        "role": "user",
        "content": user_input
    })

    # Start the tool-calling workflow.
    try:

        # Allow multiple LLM and tool interactions.
        for iteration in range(6):

            # Send the prompt and tool definitions to Groq.
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.2
            )

            # Extract the assistant's response.
            assistant_message = response.choices[0].message

            # Save the assistant message in conversation history.
            messages.append(
                assistant_message.model_dump(exclude_none=True)
            )

            # Check whether the LLM requested a tool.
            if not assistant_message.tool_calls:

                # Display the final response.
                print("\nFlight Agent:")
                print(assistant_message.content or "")
                break

            # Execute each requested tool call.
            for tool_call in assistant_message.tool_calls:

                # Get the requested function name.
                function_name = tool_call.function.name

                # Convert the JSON arguments into a Python dictionary.
                arguments = json.loads(
                    tool_call.function.arguments
                )

                # Display the tool being executed.
                print("\nExecuting:", function_name)

                # Execute the requested Python function.
                try:
                    function = available_functions[function_name]
                    result = function(**arguments)

                # Capture errors from the external API or Python function.
                except Exception as error:
                    result = {
                        "error": str(error)
                    }

                # Return the tool result to the LLM.
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": json.dumps(result, default=str)
                })

        # Handle the case where the tool loop reaches its limit.
        else:
            print("\nAgent: Maximum tool execution limit reached.")

    # Handle application errors.
    except Exception as error:
        print("\nApplication Error:", error)
