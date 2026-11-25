# Import the required libraries
import os
import requests
from datetime import datetime
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

# Load environment variables
load_dotenv()

# Read API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERP_API_KEY = os.getenv("SERPAPI_KEY")

# Check whether API keys are available
if not GROQ_API_KEY or not SERP_API_KEY:
    raise ValueError("API keys are missing from the .env file")

# Get today's date
today = datetime.now().strftime("%Y-%m-%d")

# Connect to Groq LLM
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
    """Search flights using SerpApi based on route, date, time and budget."""

    # Resolve a city name into its Google Flights ID
    def get_location(city):
        response = requests.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google_flights_autocomplete",
                "q": city,
                "api_key": SERP_API_KEY
            },
            timeout=30
        )

        # Check for HTTP errors
        response.raise_for_status()

        # Read autocomplete suggestions
        suggestions = response.json().get("suggestions", [])

        # Check whether a location was found
        if not suggestions:
            raise ValueError(f"No location found for {city}")

        # Return the first matching location ID
        return suggestions[0]["id"]

    # Get departure and arrival IDs
    departure_id = get_location(origin)
    arrival_id = get_location(destination)

    # Search for flights using SerpApi
    response = requests.get(
        "https://serpapi.com/search.json",
        params={
            "engine": "google_flights",
            "departure_id": departure_id,
            "arrival_id": arrival_id,
            "outbound_date": date,
            "type": 2,
            "api_key": SERP_API_KEY,
            "currency": "INR",
            "gl": "in"
        },
        timeout=30
    )

    # Check for HTTP errors
    response.raise_for_status()

    # Read flight-search results
    data = response.json()

    # Collect flight results
    results = []

    # Process best and other flight results
    for flight in data.get("best_flights", []) + data.get("other_flights", []):

        # Read flight segments
        segments = flight.get("flights", [])

        # Skip flights without segments
        if not segments:
            continue

        # Extract departure, arrival and price information
        departure = segments[0].get("departure_airport", {})
        arrival = segments[-1].get("arrival_airport", {})
        price = flight.get("price", 0)
        flight_time = departure.get("time", "")[-5:]

        # Apply departure-time filter
        if not flight_time or not (start_time <= flight_time <= end_time):
            continue

        # Apply maximum-price filter
        if max_price and (
            not isinstance(price, (int, float)) or price > max_price
        ):
            continue

        # Collect airline names
        airline = ", ".join(
            dict.fromkeys(s.get("airline", "Unknown") for s in segments)
        )

        # Collect flight numbers
        numbers = ", ".join(
            s.get("flight_number", "Unknown") for s in segments
        )

        # Add flight information to results
        results.append(
            f"Airline: {airline} | Flight: {numbers} | "
            f"Departure: {departure.get('time')} | "
            f"Arrival: {arrival.get('time')} | "
            f"Duration: {flight.get('total_duration')} minutes | "
            f"Price: INR {price}"
        )

        # Limit results to five flights
        if len(results) == 5:
            break

    # Return matching flight details
    return "\n\n".join(results) if results else "No matching flights found."


# Register the available tools
tools = [search_flights]

# Connect the LLM with the tools
llm_with_tools = llm.bind_tools(tools)

# Define the system instructions
system_message = SystemMessage(
    content=f"""
    You are an AI Travel Agent.

    Today's date is {today}.

    Understand the user's flight-search request.
    Convert tomorrow and other relative dates into YYYY-MM-DD
    using today's date.

    Collect origin, destination and travel date.
    Also understand departure time range and maximum budget.

    Ask for missing information one question at a time.

    Use 00:00 to 23:59 if no time range is specified.
    Use 0 if no maximum budget is specified.

    Call the search_flights tool when the required information
    is available.

    Convert time ranges into 24-hour HH:MM format.
    Pass the date in YYYY-MM-DD format.

    Display only the flight details returned by the tool.
    Never invent flight information or claim a flight is booked.
    """
)

# Define the agent node
def agent_node(state: MessagesState):

    # Send conversation history to Groq
    response = llm_with_tools.invoke(
        [system_message] + state["messages"]
    )

    # Return the LLM response to graph state
    return {"messages": [response]}


# Create the StateGraph
graph = StateGraph(MessagesState)

# Add the agent node
graph.add_node("agent", agent_node)

# Add the tool execution node
graph.add_node("tools", ToolNode(tools))

# Connect START to the agent node
graph.add_edge(START, "agent")

# Route to tools if the LLM requests a tool call
graph.add_conditional_edges("agent", tools_condition)

# Return to the agent after tool execution
graph.add_edge("tools", "agent")

# Compile the graph
app = graph.compile()

# Initialize conversation history
messages = []

# Keep the Travel Agent running
while True:

    # Get the user's flight-search prompt
    prompt = input("\nWhere do you want to travel? ")

    # Exit the application when the user enters exit
    if prompt.lower().strip() == "exit":
        break

    # Add the user's prompt to conversation history
    messages.append(HumanMessage(content=prompt))

    # Execute the LangGraph application
    response = app.invoke({"messages": messages})

    # Update conversation history
    messages = response["messages"]

    # Display the latest agent response
    print("\nTravel Agent:", messages[-1].content)
