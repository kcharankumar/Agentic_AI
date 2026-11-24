
# Install required packages
# pip install -U langchain langchain-groq requests python-dotenv

# Import required libraries
import os
import requests
from datetime import datetime
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq

# Load API keys from the .env file
load_dotenv()

# Read API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Get today's date from the computer
today = datetime.now().strftime("%Y-%m-%d")

# Connect to the Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

# Define the custom flight-search tool
@tool
def search_flights(
    origin: str,
    destination: str,
    date: str,
    start_time: str = "00:00",
    end_time: str = "23:59",
    max_price: int = 0
):
    """Search flights using SerpApi."""

    # Resolve a city name into its Google Flights identifier
    def get_location(city):
        return requests.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google_flights_autocomplete",
                "q": city,
                "api_key": SERPAPI_KEY
            }
        ).json()["suggestions"][0]["id"]

    # Get departure and arrival identifiers
    departure_id = get_location(origin)
    arrival_id = get_location(destination)

    # Search for flights using SerpApi
    data = requests.get(
        "https://serpapi.com/search.json",
        params={
            "engine": "google_flights",
            "departure_id": departure_id,
            "arrival_id": arrival_id,
            "outbound_date": date,
            "type": 2,
            "api_key": SERPAPI_KEY,
            "currency": "INR",
            "gl": "in"
        }
    ).json()

    # Collect flight results
    results = []

    # Process each returned flight
    for flight in data.get("best_flights", []) + data.get("other_flights", []):

        # Read the flight segments
        segments = flight.get("flights", [])
        if not segments:
            continue

        # Extract departure, arrival and price information
        departure = segments[0].get("departure_airport", {})
        arrival = segments[-1].get("arrival_airport", {})
        price = flight.get("price", 0)
        flight_time = departure.get("time", "")[-5:]

        # Apply departure-time and budget filters
        if not flight_time or not (start_time <= flight_time <= end_time):
            continue

        # Skip flights exceeding the requested budget
        if max_price and (not isinstance(price, (int, float)) or price > max_price):
            continue

        # Prepare airline and flight-number information
        airline = ", ".join(
            dict.fromkeys(s.get("airline", "Unknown") for s in segments)
        )
        numbers = ", ".join(
            s.get("flight_number", "Unknown") for s in segments
        )

        # Add the flight information to the results
        results.append(
            f"Airline: {airline} | Flight: {numbers} | "
            f"Departure: {departure.get('time')} | "
            f"Arrival: {arrival.get('time')} | "
            f"Duration: {flight.get('total_duration')} minutes | "
            f"Price: INR {price}"
        )

        # Limit the results to five flights
        if len(results) == 5:
            break

    # Return flight details with a blank line between flights
    return "\n\n".join(results) if results else "No matching flights found."

# Create the LangChain agent with the flight-search tool
agent = create_agent(
    model=llm,
    tools=[search_flights],
    system_prompt=f"""
    You are an AI Travel Agent.

    Today's date is {today}.

    Understand the user's flight-search request.
    Convert tomorrow and other relative dates into YYYY-MM-DD
    using today's date.

    Collect origin, destination and travel date.
    Also collect departure time range and maximum budget.
    Ask for missing information one question at a time.

    Use 00:00 to 23:59 if no time range is specified.
    Use 0 if no maximum budget is specified.

    Call the search_flights tool when the required
    information is available.

    Display only the flight details returned by the tool.
    Never invent flight information or claim a flight is booked.
    """
)

# Initialize conversation history
messages = []

# Keep the Travel Agent running continuously
while True:

    # Accept the user's flight-search prompt
    prompt = input("\nEnter your flight request (or exit): ")

    # Stop the application when the user enters exit
    if prompt.lower() == "exit":
        break

    # Add the user's message to conversation history
    messages.append({"role": "user", "content": prompt})

    # Execute the LangChain agent with the conversation history
    response = agent.invoke({"messages": messages})

    # Update conversation history with the agent's messages
    messages = response["messages"]

    # Display the latest agent response
    print("\nTravel Agent:", messages[-1].content)
