
# Import the library components used in this program.
import os
# Import the library components used in this program.
import uuid
# Import the library components used in this program.
import requests
# Import the library components used in this program.
import qrcode

# Import the library components used in this program.
from datetime import datetime
# Import the library components used in this program.
from typing import List, TypedDict, Optional

# Import the library components used in this program.
from dotenv import load_dotenv

# Import the library components used in this program.
from langchain.tools import tool
# Import the library components used in this program.
from langchain_groq import ChatGroq

# Import the library components used in this program.
from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)

# Import the library components used in this program.
from langgraph.graph import (
    StateGraph,
    MessagesState,
    START
)

# Import the library components used in this program.
from langgraph.prebuilt import (
    ToolNode,
    tools_condition
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Execute load_dotenv.
load_dotenv()

# Load the required API key from the environment.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Load the required API key from the environment.
SERP_API_KEY = os.getenv("SERPAPI_KEY")

# Check this condition before choosing which block to execute.
if not GROQ_API_KEY or not SERP_API_KEY:
    # Stop execution by raising an error with an explanatory message.
    raise ValueError("Check GROQ_API_KEY and SERPAPI_KEY in .env")

# Get today's date in YYYY-MM-DD format for interpreting relative travel dates.
today = datetime.now().strftime("%Y-%m-%d")

# Initialize the Groq language model.
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

# Stores flight search results for this CLI demonstration.
# Create a dictionary to retain flight search results for later selection.
available_flights = {}


# ============================================================
# 2. SEARCH FLIGHTS TOOL
# ============================================================

@tool
# Define search_flights, which handles this part of the Travel Agent workflow.
def search_flights(
    origin: str,
    destination: str,
    date: str,
    start_time: str = "00:00",
    end_time: str = "23:59",
    max_price: int = 0
) -> str:
    """
    Search one-way flights using Google Flights through SerpApi.

    origin: Departure city or airport.
    destination: Arrival city or airport.
    date: Travel date in YYYY-MM-DD format.
    start_time: Earliest departure time in HH:MM.
    end_time: Latest departure time in HH:MM.
    max_price: Maximum budget in INR. Zero means no limit.
    """

    # Execute this Global statement.
    global available_flights

    # Run the API operations with error handling.
    try:

        # Get departure airport ID.
        # Request autocomplete matches for the departure location.
        origin_response = requests.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google_flights_autocomplete",
                "q": origin,
                "api_key": SERP_API_KEY
            },
            timeout=30
        )

        # Raise an exception if the HTTP response indicates a failure.
        origin_response.raise_for_status()

        # Convert the departure autocomplete response into a Python dictionary.
        origin_data = origin_response.json()

        # Assign the result of this operation to origin_suggestions.
        origin_suggestions = origin_data.get("suggestions", [])

        # Check this condition before choosing which block to execute.
        if not origin_suggestions:
            # Return the result to the caller or to LangGraph.
            return f"Departure location not found: {origin}"

        # Store the airport identifier returned for the departure location.
        departure_id = origin_suggestions[0]["id"]

        # Get destination airport ID.
        # Request autocomplete matches for the destination location.
        destination_response = requests.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google_flights_autocomplete",
                "q": destination,
                "api_key": SERP_API_KEY
            },
            timeout=30
        )

        # Raise an exception if the HTTP response indicates a failure.
        destination_response.raise_for_status()

        # Convert the destination autocomplete response into a Python dictionary.
        destination_data = destination_response.json()

        # Assign the result of this operation to destination_suggestions.
        destination_suggestions = destination_data.get(
            "suggestions", []
        )

        # Check this condition before choosing which block to execute.
        if not destination_suggestions:
            # Return the result to the caller or to LangGraph.
            return f"Destination not found: {destination}"

        # Store the airport identifier returned for the destination.
        arrival_id = destination_suggestions[0]["id"]

        # Search Google Flights.
        # Request flight listings from Google Flights through SerpApi.
        flight_response = requests.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google_flights",
                "departure_id": departure_id,
                "arrival_id": arrival_id,
                "outbound_date": date,
                "type": 2,
                "currency": "INR",
                "gl": "in",
                "hl": "en",
                "api_key": SERP_API_KEY
            },
            timeout=30
        )

        # Raise an exception if the HTTP response indicates a failure.
        flight_response.raise_for_status()

        # Convert the flight-search response into a Python dictionary.
        flight_data = flight_response.json()

        # Check this condition before choosing which block to execute.
        if "error" in flight_data:
            # Return the result to the caller or to LangGraph.
            return f"API Error: {flight_data['error']}"

        # Combine the best-flight and other-flight results returned by SerpApi.
        flights = (
            flight_data.get("best_flights", [])
            + flight_data.get("other_flights", [])
        )

        # Check this condition before choosing which block to execute.
        if not flights:
            # Return the result to the caller or to LangGraph.
            return "No flights found for this journey."

        # Clear the previous search results.
        # Remove previous search results so selections refer to the latest search.
        available_flights.clear()

        # Initialize the list of flights that pass the user's filters.
        results = []

        # Filter flights based on time and budget.
        # Loop through the returned items and process them one at a time.
        for flight in flights:

            # Extract the individual flight segments for this itinerary.
            segments = flight.get("flights", [])

            # Check this condition before choosing which block to execute.
            if not segments:
                # Skip the current item and continue with the next loop iteration.
                continue

            # Get the first segment to read departure details.
            first_segment = segments[0]
            # Get the last segment to read the final arrival details.
            last_segment = segments[-1]

            # Extract the departure timestamp from the first segment.
            departure = first_segment.get(
                "departure_airport", {}
            ).get("time", "")

            # Extract the arrival timestamp from the last segment.
            arrival = last_segment.get(
                "arrival_airport", {}
            ).get("time", "")

            # Check this condition before choosing which block to execute.
            if len(departure) < 16:
                # Skip the current item and continue with the next loop iteration.
                continue

            # Extract the HH:MM portion of the departure timestamp.
            departure_clock = departure[11:16]

            # Check this condition before choosing which block to execute.
            if not (start_time <= departure_clock <= end_time):
                # Skip the current item and continue with the next loop iteration.
                continue

            # Read the selected flight's price.
            price = flight.get("price", 0)

            # Check this condition before choosing which block to execute.
            if not isinstance(price, (int, float)):
                # Skip the current item and continue with the next loop iteration.
                continue

            # Check this condition before choosing which block to execute.
            if max_price > 0 and price > max_price:
                # Skip the current item and continue with the next loop iteration.
                continue

            # Read the airline name from the first segment.
            airline = first_segment.get("airline", "Unknown")

            # Collect the flight number from each segment.
            flight_numbers = [
                segment.get("flight_number", "")
                for segment in segments
            ]

            # Join the segment flight numbers into one display string.
            flight_number = ", ".join(
                number for number in flight_numbers if number
            )

            # Read the total journey duration returned by the API.
            duration = flight.get("total_duration", "Unknown")

            # Display a maximum of five matching flights.
            # Check this condition before choosing which block to execute.
            if len(results) >= 5:
                # Exit the current loop.
                break

            # Assign the next number that the user can use to select this flight.
            selection = len(results) + 1

            # Save the flight for the booking tool.
            # Assign the result of this operation to available_flights[selection].
            available_flights[selection] = {
                "airline": airline,
                "flight_number": flight_number,
                "departure": departure,
                "arrival": arrival,
                "duration": duration,
                "price": price
            }

            # Add the new item to the relevant list.
            results.append(
                f"""
Flight Selection: {selection}
Airline: {airline}
Flight Number: {flight_number}
Departure: {departure}
Arrival: {arrival}
Duration: {duration} minutes
Price: INR {price}
"""
            )

        # Check this condition before choosing which block to execute.
        if not results:
            # Return the result to the caller or to LangGraph.
            return (
                "No flights matched the requested time and budget. "
                "Ask the user whether they want to change the filters."
            )

        # Return the result to the caller or to LangGraph.
        return (
            f"Flights from {origin} to {destination} on {date}:\n"
            + "\n".join(results)
        )

    except requests.RequestException as error:
        # Return the result to the caller or to LangGraph.
        return f"Flight API error: {error}"

    except Exception as error:
        # Return the result to the caller or to LangGraph.
        return f"Unexpected error: {error}"


# ============================================================
# 3. BOOK FLIGHT TOOL
# ============================================================

@tool
# Define book_flight, which handles this part of the Travel Agent workflow.
def book_flight(
    flight_selection: int,
    passenger_name: str,
    age: int,
    gender: str,
    email: str,
    phone: str
) -> str:
    """
    Simulate booking a selected flight.

    Requires flight selection and passenger details.
    This is a demonstration, not a real airline booking.
    """

    # Execute this Global statement.
    global available_flights

    # Validate flight selection.
    # Check this condition before choosing which block to execute.
    if flight_selection not in available_flights:
        # Return the result to the caller or to LangGraph.
        return "Invalid flight selection. Please search again."

    # Validate passenger details.
    # Check this condition before choosing which block to execute.
    if not passenger_name.strip():
        # Return the result to the caller or to LangGraph.
        return "Passenger name is required."

    # Check this condition before choosing which block to execute.
    if age <= 0:
        # Return the result to the caller or to LangGraph.
        return "Please provide a valid age."

    # Check this condition before choosing which block to execute.
    if not email.strip() or not phone.strip():
        # Return the result to the caller or to LangGraph.
        return "Email and phone are required."

    # Retrieve the saved details of the flight selected by the user.
    selected_flight = available_flights[flight_selection]

    # Read the selected flight's price.
    price = selected_flight["price"]

    # Display booking information.
    # Display information in the terminal.
    print("\n========== BOOKING DETAILS ==========")

    # Display information in the terminal.
    print("Passenger:", passenger_name)
    # Display information in the terminal.
    print("Age:", age)
    # Display information in the terminal.
    print("Gender:", gender)
    # Display information in the terminal.
    print("Email:", email)
    # Display information in the terminal.
    print("Phone:", phone)

    # Display information in the terminal.
    print("Airline:", selected_flight["airline"])
    # Display information in the terminal.
    print("Flight:", selected_flight["flight_number"])
    # Display information in the terminal.
    print("Price: INR", price)

    # Display information in the terminal.
    print("=====================================")

    # Obtain explicit booking approval.
    # Ask the user to explicitly approve the demonstration booking.
    approval = input(
        "\nApprove this DEMO booking? Type YES: "
    )

    # Check this condition before choosing which block to execute.
    if approval.strip().upper() != "YES":
        # Return the result to the caller or to LangGraph.
        return "Booking cancelled."

    # Generate demonstration booking references.
    # Generate a unique demonstration booking reference.
    booking_reference = "DEMO-" + uuid.uuid4().hex[:8].upper()

    # Generate a unique demonstration payment reference.
    payment_reference = "PAY-" + uuid.uuid4().hex[:8].upper()

    # Create a demonstration QR code.
    # Prepare the demonstration-only text embedded in the QR code.
    qr_payload = (
        "DEMO PAYMENT ONLY - NOT A REAL UPI PAYMENT\n"
        f"Booking Reference: {booking_reference}\n"
        f"Amount: INR {price}\n"
        f"Payment Reference: {payment_reference}"
    )

    # Generate the demonstration QR-code image.
    qr = qrcode.make(qr_payload)

    # Set the filename for the generated QR-code image.
    qr_file = "demo_payment_qr.png"

    # Save the generated QR-code image to disk.
    qr.save(qr_file)

    # Display information in the terminal.
    print("\nDemo QR code saved:", qr_file)

    # Open QR image on Windows.
    # Check this condition before choosing which block to execute.
    if os.name == "nt":
        # Open the generated QR image using the Windows default image viewer.
        os.startfile(os.path.abspath(qr_file))

    # Simulate payment confirmation.
    # Display information in the terminal.
    print("\nThis QR code does not collect real payment.")

    # Ask the user to confirm the simulated payment.
    payment_status = input(
        "Type DEMO PAID to simulate payment: "
    )

    # Check this condition before choosing which block to execute.
    if payment_status.strip().upper() != "DEMO PAID":
        # Return the result to the caller or to LangGraph.
        return "Demo payment not confirmed. No ticket issued."

    # Return the result to the caller or to LangGraph.
    return f"""
DEMO BOOKING CONFIRMED

Booking Reference: {booking_reference}
Payment Reference: {payment_reference}

Passenger: {passenger_name}
Airline: {selected_flight['airline']}
Flight: {selected_flight['flight_number']}
Amount: INR {price}

Status: DEMO PAYMENT CONFIRMED

No real ticket has been booked or issued.
"""


# ============================================================
# 4. REGISTER TOOLS
# ============================================================

# Register the Python functions that the Travel Agent can call as tools.
tools = [
    search_flights,
    book_flight
]

# Bind tools to the LLM for execution.
# Bind the registered tools to the language model.
llm_with_tools = llm.bind_tools(tools)


# ============================================================
# 5. DEFINE LANGGRAPH STATE
# ============================================================

# Define the TravelState state structure used by LangGraph.
class TravelState(MessagesState):

    # Stores the plan generated by the Planner Node.
    # Execute this AnnAssign statement.
    plan: List[str]


# ============================================================
# 6. PLANNER NODE
# ============================================================

# Define planner_node, which handles this part of the Travel Agent workflow.
def planner_node(state: TravelState):

    # Display information in the terminal.
    print("\n========== PLANNER NODE ==========")

    # Read conversation history.
    # Retrieve the conversation history from graph state.
    conversation = state["messages"]

    # Read the previous plan.
    # Retrieve the plan from the previous graph invocation, if one exists.
    previous_plan = state.get("plan", [])

    # Build the system instructions for this node; the prompt text below is intentionally left uncommented.
    planner_prompt = f"""
You are the Planner Node of a Travel Agent.

Today's date is {today}.

Your job is to understand the user's travel goal
and create a numbered execution plan.

Previous Plan:
{previous_plan}

Create a plan covering the relevant tasks:

1. Collect origin and destination.
2. Collect travel date.
3. Collect departure time range and budget.
4. Search available flights.
5. Display flight results.
6. Wait for flight selection.
7. Collect passenger details.
8. Obtain explicit booking approval.
9. Execute the booking demonstration.
10. Confirm the result.

Rules:

- Consider the complete conversation history.
- Do not invent missing information.
- Do not execute tools.
- Ask for missing information one question at a time.
- Do not assume a flight has been selected.
- Do not assume booking approval.
- Revise the plan when the user changes requirements.
- Include only tasks relevant to the user's current goal.
- Return a numbered list of tasks only.
"""

    # Planner uses the LLM without tool binding.
    # Ask the LLM to generate the plan without giving it tools.
    response = llm.invoke(
        [SystemMessage(content=planner_prompt)] + conversation
    )

    # Convert the LLM response into plan steps.
    # Check this condition before choosing which block to execute.
    if isinstance(response.content, str):

        # Create or update the list of plan steps returned by the Planner.
        plan = [
            line.strip()
            for line in response.content.splitlines()
            if line.strip()
        ]

    else:

        # Create or update the list of plan steps returned by the Planner.
        plan = [
            str(block.get("text", "")).strip()
            for block in response.content
            if isinstance(block, dict) and block.get("text")
        ]

    # Display information in the terminal.
    print("\nGenerated Plan:")

    # Loop through the returned items and process them one at a time.
    for step in plan:
        # Display information in the terminal.
        print(step)

    # Display information in the terminal.
    print("=================================")

    # Save plan in LangGraph state.
    # Return the result to the caller or to LangGraph.
    return {
        "plan": plan
    }


# ============================================================
# 7. EXECUTOR NODE
# ============================================================

# Define execution_node, which handles this part of the Travel Agent workflow.
def execution_node(state: TravelState):

    # Display information in the terminal.
    print("\n========== EXECUTOR NODE ==========")

    # Retrieve the plan.
    # Create or update the list of plan steps returned by the Planner.
    plan = state.get("plan", [])

    # Convert the plan list into readable text for the Executor.
    plan_text = "\n".join(plan)

    # Retrieve conversation history.
    # Retrieve the conversation history from graph state.
    conversation = state["messages"]

    # Build the system instructions for this node; the prompt text below is intentionally left uncommented.
    execution_prompt = f"""
You are the Execution Node of a Travel Agent.

Your responsibility is to execute the Planner's plan.

CURRENT PLAN:

{plan_text}

AVAILABLE TOOLS:

1. search_flights:
   Searches one-way flights.
   Requires origin, destination, date, start_time,
   end_time and max_price.

2. book_flight:
   Simulates booking a selected flight.
   Requires flight_selection, passenger_name, age,
   gender, email and phone.

EXECUTION RULES:

1. Read the complete conversation history.
2. Identify completed tasks.
3. Determine the next pending task.
4. Ask the user for missing information one question at a time.
5. Call search_flights when all search details are available.
6. Show the search results to the user.
7. Wait for an explicit flight selection.
8. Collect all passenger details.
9. Obtain explicit booking approval.
10. Call book_flight only after approval.
11. Do not invent flight or passenger information.
12. Do not repeat completed tasks unnecessarily.
13. If the user changes requirements, follow the revised plan.

The Planner creates the plan.
You execute the plan using the available tools.
"""

    # Executor uses the LLM with tools bound.
    # Ask the tool-enabled LLM to determine the next action or tool call.
    response = llm_with_tools.invoke(
        [SystemMessage(content=execution_prompt)] + conversation
    )

    # Display information in the terminal.
    print("Executor determined the next action.")

    # Return the LLM response to LangGraph.
    # Return the result to the caller or to LangGraph.
    return {
        "messages": [response]
    }


# ============================================================
# 8. BUILD LANGGRAPH WORKFLOW
# ============================================================

# Create the LangGraph state machine using the custom TravelState schema.
graph = StateGraph(TravelState)

# Register graph nodes.
# Register this node and its Python function in the LangGraph workflow.
graph.add_node("planner", planner_node)

# Register this node and its Python function in the LangGraph workflow.
graph.add_node("executor", execution_node)

# Register this node and its Python function in the LangGraph workflow.
graph.add_node("tools", ToolNode(tools))

# Start with the Planner Node.
# Connect these two points in the LangGraph workflow.
graph.add_edge(START, "planner")

# Planner sends its plan to Executor.
# Connect these two points in the LangGraph workflow.
graph.add_edge("planner", "executor")

# Route to tools when the Executor requests them.
# Configure conditional routing based on whether the LLM requested a tool.
graph.add_conditional_edges(
    "executor",
    tools_condition
)

# Return to Executor after tool execution.
# Connect these two points in the LangGraph workflow.
graph.add_edge("tools", "executor")

# Compile the graph.
# Compile the graph into an executable application.
app = graph.compile()


# ============================================================
# 9. INTERACTIVE TRAVEL AGENT
# ============================================================

# Initialize the conversation history.
messages = []

# Initialize the variable that carries the plan between user turns.
current_plan = []

# Track whether the program is waiting for the user's first message.
first_interaction = True

# Display information in the terminal.
print("\n======================================")
# Display information in the terminal.
print("     PLAN AND EXECUTE TRAVEL AGENT")
# Display information in the terminal.
print("======================================")

# Display information in the terminal.
print("Search flights and simulate booking.")
# Display information in the terminal.
print("Type 'exit' to stop.")

# Keep the interactive Travel Agent running until the user exits.
while True:

    # Display the first interaction prompt.
    # Check this condition before choosing which block to execute.
    if first_interaction:

        # Read the user's next message from the terminal.
        prompt = input("\nWhere do you want to travel? ")

        # Track whether the program is waiting for the user's first message.
        first_interaction = False

    else:

        # Read the user's next message from the terminal.
        prompt = input("\nYou: ")

    # Exit the application.
    # Check this condition before choosing which block to execute.
    if prompt.strip().lower() in ["exit", "quit"]:

        # Display information in the terminal.
        print("Thank you for using Travel Agent.")

        # Exit the current loop.
        break

    # Add user input to conversation history.
    # Add the new item to the relevant list.
    messages.append(
        HumanMessage(content=prompt)
    )

    # Invoke the LangGraph workflow.
    # Run the LangGraph workflow with the current conversation and plan.
    result = app.invoke(
        {
            "messages": messages,
            "plan": current_plan
        }
    )

    # Update conversation history.
    # Initialize the conversation history.
    messages = result["messages"]

    # Update the execution plan.
    # Initialize the variable that carries the plan between user turns.
    current_plan = result.get("plan", current_plan)

    # Display the latest response.
    # Display information in the terminal.
    print("\nTravel Agent:", messages[-1].content)
