
# pip install -U langchain langchain-groq langgraph requests python-dotenv qrcode[pil]

# Import the required libraries
import os
import uuid
import requests
import qrcode

from datetime import datetime
from dotenv import load_dotenv

from langchain.tools import tool
from langchain_groq import ChatGroq

from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition


# Load environment variables
load_dotenv()

# Read API keys from the .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERP_API_KEY = os.getenv("SERPAPI_KEY")

# Validate API keys before starting the application
if not GROQ_API_KEY or not SERP_API_KEY:
    raise ValueError("API keys are missing from the .env file")


# Get today's date for interpreting relative travel dates
today = datetime.now().strftime("%Y-%m-%d")

# Initialize the Groq language model
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# Store the latest flight-search results for selection
available_flights = {}


# Define the flight-search tool
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

    # Resolve a city name into its Google Flights location ID
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

        # Read the autocomplete suggestions
        suggestions = response.json().get("suggestions", [])

        # Stop if the location cannot be found
        if not suggestions:
            raise ValueError(f"No location found for {city}")

        # Return the first matching location ID
        return suggestions[0]["id"]

    # Resolve departure and arrival locations
    departure_id = get_location(origin)
    arrival_id = get_location(destination)

    # Request Google Flights results through SerpApi
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

    # Check for HTTP errors and read the results
    response.raise_for_status()
    data = response.json()

    # Clear previous search results before storing new ones
    available_flights.clear()

    # Process the best and other flight results
    for flight in data.get("best_flights", []) + data.get("other_flights", []):

        # Read the flight segments
        segments = flight.get("flights", [])

        # Ignore flights without segment information
        if not segments:
            continue

        # Extract departure, arrival and price information
        departure = segments[0].get("departure_airport", {})
        arrival = segments[-1].get("arrival_airport", {})
        price = flight.get("price", 0)
        flight_time = departure.get("time", "")[-5:]

        # Apply the requested departure-time filter
        if not flight_time or not (start_time <= flight_time <= end_time):
            continue

        # Apply the maximum-price filter
        if max_price and (
            not isinstance(price, (int, float)) or price > max_price
        ):
            continue

        # Collect airline names from all flight segments
        airline = ", ".join(
            dict.fromkeys(s.get("airline", "Unknown") for s in segments)
        )

        # Collect flight numbers from all flight segments
        numbers = ", ".join(
            s.get("flight_number", "Unknown") for s in segments
        )

        # Assign a selection number to this flight
        flight_number = len(available_flights) + 1

        # Store the flight details for the booking tool
        available_flights[flight_number] = {
            "airline": airline,
            "flight_numbers": numbers,
            "departure": departure.get("time", ""),
            "arrival": arrival.get("time", ""),
            "duration": flight.get("total_duration", "Unknown"),
            "price": price
        }

        # Stop after collecting five matching flights
        if flight_number == 5:
            break

    # Return numbered flight options to the LLM
    if not available_flights:
        return "No matching flights found."

    # Format the flight details for display
    output = []

    for number, flight in available_flights.items():
        output.append(
            f"Flight {number}\n"
            f"Airline: {flight['airline']}\n"
            f"Flight Number: {flight['flight_numbers']}\n"
            f"Departure: {flight['departure']}\n"
            f"Arrival: {flight['arrival']}\n"
            f"Duration: {flight['duration']} minutes\n"
            f"Price: INR {flight['price']}"
        )

    # Return all matching flight options
    return "\n\n".join(output)


# Define the separate flight-booking tool
@tool
def book_flight(
    flight_number: int,
    passenger_name: str,
    age: int,
    gender: str,
    email: str,
    phone: str
):
    """Initiate a simulated booking for a user-selected flight."""

    # Verify that the selected flight exists in the latest search
    if flight_number not in available_flights:
        return "Invalid flight selection. Please search and select a valid flight."

    # Retrieve the selected flight details
    flight = available_flights[flight_number]

    # Validate the passenger information
    if age <= 0 or not passenger_name.strip():
        return "Invalid passenger details. Please check the information."

    if not email.strip() or not phone.strip():
        return "Passenger email and phone number are required."

    # Generate a unique demo booking reference
    booking_reference = "DEMO-" + uuid.uuid4().hex[:8].upper()

    # Use the displayed flight price as the simulated final amount
    final_amount = flight["price"]

    # Display the simulated booking summary
    print("\n========== DEMO BOOKING SUMMARY ==========")
    print("Booking Reference:", booking_reference)
    print("Passenger:", passenger_name)
    print("Age:", age)
    print("Gender:", gender)
    print("Email:", email)
    print("Phone:", phone)
    print("Airline:", flight["airline"])
    print("Flight:", flight["flight_numbers"])
    print("Departure:", flight["departure"])
    print("Arrival:", flight["arrival"])
    print("Final Demo Amount: INR", final_amount)
    print("==========================================")

    # Ask the user to approve the displayed amount
    approval = input(
        "\nDo you approve this demo booking amount? Type YES to continue: "
    ).strip().upper()

    # Stop if the user does not approve the amount
    if approval != "YES":
        return "Demo booking cancelled. The amount was not approved."

    # Create a demo payment reference
    payment_reference = "PAY-" + uuid.uuid4().hex[:8].upper()

    # Prepare a clearly labelled demo QR payload
    qr_payload = (
        f"DEMO PAYMENT ONLY\n"
        f"Booking: {booking_reference}\n"
        f"Payment: {payment_reference}\n"
        f"Amount: INR {final_amount}"
    )

    # Generate the demo QR code image
    qr = qrcode.make(qr_payload)

    # Save the QR image locally
    qr_filename = f"{payment_reference}.png"
    qr.save(qr_filename)

    # Display the QR code location and payment instructions
    print("\n========== DEMO PAYMENT ==========")
    print("Demo QR image:", os.path.abspath(qr_filename))
    print("Amount: INR", final_amount)
    print("This is NOT a UPI payment QR.")
    print("Do not scan it using a payment app.")
    print("==================================")

    # Open the QR image on Windows
    if os.name == "nt":
        os.startfile(os.path.abspath(qr_filename))

    # Ask the user to simulate successful payment
    payment_status = input(
        "\nFor this demo only, type DEMO PAID to simulate payment: "
    ).strip().upper()

    # Stop if the demo payment is not confirmed
    if payment_status != "DEMO PAID":
        return (
            f"Demo booking {booking_reference} is pending payment. "
            "No ticket has been issued."
        )

    # Simulate the airline ticket confirmation
    return (
        "\n========== DEMO TICKET CONFIRMED ==========\n"
        f"Booking Reference: {booking_reference}\n"
        f"Passenger: {passenger_name}\n"
        f"Airline: {flight['airline']}\n"
        f"Flight: {flight['flight_numbers']}\n"
        f"Amount: INR {final_amount}\n"
        "Payment Status: DEMO PAYMENT CONFIRMED\n"
        "Ticket Status: DEMO CONFIRMED\n"
        "==========================================="
    )


# Register both available tools
tools = [search_flights, book_flight]


# Connect the LLM to the registered tools
llm_with_tools = llm.bind_tools(tools)


# Define instructions for the AI Travel Agent
system_message = SystemMessage(
    content=f"""
    You are an AI Travel Agent.

    Today's date is {today}.

    FLIGHT SEARCH:
    1. Understand the user's flight-search request.
    2. Convert relative dates such as tomorrow into YYYY-MM-DD.
    3. Collect origin, destination and travel date.
    4. Understand departure time range and maximum budget.
    5. Ask for missing information one question at a time.
    6. Use 00:00 to 23:59 if no time range is specified.
    7. Use 0 if no maximum budget is specified.
    8. Call search_flights when the required information is available.
    9. Display the numbered flight options returned by the tool.

    FLIGHT BOOKING:
    1. Wait until the user explicitly selects a flight.
    2. Do not assume that displaying a flight means the user wants to book it.
    3. Collect passenger name, age, gender, email and phone number.
    4. Ask for missing passenger details one question at a time.
    5. Call book_flight only after the user has selected a flight
       and supplied all required passenger details.
    6. The booking tool will display the demo amount and ask for
       explicit approval before proceeding.
    7. Do not claim that an actual airline booking or payment occurred.
    8. Clearly describe the booking and payment as a simulation.

    IMPORTANT:
    The search tool provides flight-search results only.
    Never claim that a real flight has been booked or ticketed.
    Never invent flight details or prices.
    """
)


# Define the agent node
def agent_node(state: MessagesState):

    # Send system instructions and conversation history to Groq
    response = llm_with_tools.invoke(
        [system_message] + state["messages"]
    )

    # Return the LLM response to the graph state
    return {"messages": [response]}


# Create the LangGraph StateGraph
graph = StateGraph(MessagesState)

# Add the agent node to the graph
graph.add_node("agent", agent_node)

# Add the tool execution node to the graph
graph.add_node("tools", ToolNode(tools))

# Connect the START node to the agent
graph.add_edge(START, "agent")

# Route to tools when the LLM requests a tool call
graph.add_conditional_edges("agent", tools_condition)

# Return to the agent after tool execution
graph.add_edge("tools", "agent")

# Compile the graph into an executable application
app = graph.compile()


# Track whether this is the first user interaction
first_interaction = True

# Initialize conversation history
messages = []

# Start the interactive Travel Agent application
while True:

    # Display the appropriate prompt
    if first_interaction:
        prompt = input("\nWhere do you want to travel? ")
        first_interaction = False
    else:
        prompt = input("\nYou: ")

    # Exit when the user enters exit
    if prompt.lower().strip() == "exit":
        break

    # Add the new user message to conversation history
    messages.append(HumanMessage(content=prompt))

    # Execute the LangGraph application
    response = app.invoke({"messages": messages})

    # Update the conversation history with the graph response
    messages = response["messages"]

    # Display the latest response from the Travel Agent
    latest_message = messages[-1].content

    if latest_message:
        print("\nTravel Agent:", latest_message)
