import os
import uuid
import requests
import qrcode

from datetime import datetime
from typing import List

from dotenv import load_dotenv

from langchain.tools import tool
from langchain_groq import ChatGroq

from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)

from langgraph.graph import (
    StateGraph,
    MessagesState,
    START
)

from langgraph.prebuilt import (
    ToolNode,
    tools_condition
)



load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERP_API_KEY = os.getenv("SERPAPI_KEY")

if not GROQ_API_KEY or not SERP_API_KEY:
    raise ValueError("Check GROQ_API_KEY and SERPAPI_KEY in .env")

today = datetime.now().strftime("%Y-%m-%d")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

available_flights = {}



@tool
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

    global available_flights

    try:

        origin_response = requests.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google_flights_autocomplete",
                "q": origin,
                "api_key": SERP_API_KEY
            },
            timeout=30
        )

        origin_response.raise_for_status()

        origin_data = origin_response.json()

        origin_suggestions = origin_data.get("suggestions", [])

        if not origin_suggestions:
            return f"Departure location not found: {origin}"

        departure_id = origin_suggestions[0]["id"]

        destination_response = requests.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google_flights_autocomplete",
                "q": destination,
                "api_key": SERP_API_KEY
            },
            timeout=30
        )

        destination_response.raise_for_status()

        destination_data = destination_response.json()

        destination_suggestions = destination_data.get(
            "suggestions", []
        )

        if not destination_suggestions:
            return f"Destination not found: {destination}"

        arrival_id = destination_suggestions[0]["id"]

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

        flight_response.raise_for_status()

        flight_data = flight_response.json()

        if "error" in flight_data:
            return f"API Error: {flight_data['error']}"

        flights = (
            flight_data.get("best_flights", [])
            + flight_data.get("other_flights", [])
        )

        if not flights:
            return "No flights found for this journey."

        available_flights.clear()

        results = []

        for flight in flights:

            segments = flight.get("flights", [])

            if not segments:
                continue

            first_segment = segments[0]
            last_segment = segments[-1]

            departure = first_segment.get(
                "departure_airport", {}
            ).get("time", "")

            arrival = last_segment.get(
                "arrival_airport", {}
            ).get("time", "")

            if len(departure) < 16:
                continue

            departure_clock = departure[11:16]

            if not (start_time <= departure_clock <= end_time):
                continue

            price = flight.get("price", 0)

            if not isinstance(price, (int, float)):
                continue

            if max_price > 0 and price > max_price:
                continue

            airline = first_segment.get("airline", "Unknown")

            flight_numbers = [
                segment.get("flight_number", "")
                for segment in segments
            ]

            flight_number = ", ".join(
                number for number in flight_numbers if number
            )

            duration = flight.get("total_duration", "Unknown")

            if len(results) >= 5:
                break

            selection = len(results) + 1

            available_flights[selection] = {
                "airline": airline,
                "flight_number": flight_number,
                "departure": departure,
                "arrival": arrival,
                "duration": duration,
                "price": price
            }

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

        if not results:
            return (
                "No flights matched the requested time and budget. "
                "Ask the user whether they want to change the filters."
            )

        return (
            f"Flights from {origin} to {destination} on {date}:\n"
            + "\n".join(results)
        )

    except requests.RequestException as error:
        return f"Flight API error: {error}"

    except Exception as error:
        return f"Unexpected error: {error}"



@tool
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

    global available_flights

    if flight_selection not in available_flights:
        return "Invalid flight selection. Please search again."

    if not passenger_name.strip():
        return "Passenger name is required."

    if age <= 0:
        return "Please provide a valid age."

    if not email.strip() or not phone.strip():
        return "Email and phone are required."

    selected_flight = available_flights[flight_selection]

    price = selected_flight["price"]

    print("\n========== BOOKING DETAILS ==========")

    print("Passenger:", passenger_name)
    print("Age:", age)
    print("Gender:", gender)
    print("Email:", email)
    print("Phone:", phone)

    print("Airline:", selected_flight["airline"])
    print("Flight:", selected_flight["flight_number"])
    print("Price: INR", price)

    print("=====================================")

    approval = input(
        "\nApprove this DEMO booking? Type YES: "
    )

    if approval.strip().upper() != "YES":
        return "Booking cancelled."

    booking_reference = "DEMO-" + uuid.uuid4().hex[:8].upper()

    payment_reference = "PAY-" + uuid.uuid4().hex[:8].upper()

    qr_payload = (
        "DEMO PAYMENT ONLY - NOT A REAL UPI PAYMENT\n"
        f"Booking Reference: {booking_reference}\n"
        f"Amount: INR {price}\n"
        f"Payment Reference: {payment_reference}"
    )

    qr = qrcode.make(qr_payload)

    qr_file = "demo_payment_qr.png"

    qr.save(qr_file)

    print("\nDemo QR code saved:", qr_file)

    if os.name == "nt":
        os.startfile(os.path.abspath(qr_file))

    print("\nThis QR code does not collect real payment.")

    payment_status = input(
        "Type DEMO PAID to simulate payment: "
    )

    if payment_status.strip().upper() != "DEMO PAID":
        return "Demo payment not confirmed. No ticket issued."

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



tools = [
    search_flights,
    book_flight
]

llm_with_tools = llm.bind_tools(tools)



class TravelState(MessagesState):
    # Stores the high-level plan generated by the Planner.
    plan: List[str]

    # Stores the smaller actionable tasks generated by the Decomposition Node.
    subtasks: List[str]



def planner_node(state: TravelState):
    """
    Planner Node: identifies the major activities needed to achieve
    the user's travel goal. It does not create detailed subtasks.
    """

    print("\n========== PLANNER NODE ==========")

    conversation = state["messages"]
    previous_plan = state.get("plan", [])

    planner_prompt = f"""
        You are the Planner Node of a Travel Agent.

        Today's date is {today}.

        Understand the user's complete travel goal and create a high-level plan.

        Previous high-level plan:
        {previous_plan}

        For a flight search and booking request, identify the relevant major
        activities, such as:

        1. Understand travel requirements.
        2. Search for available flights.
        3. Select a suitable flight.
        4. Collect passenger information.
        5. Obtain booking approval.
        6. Complete the booking process.
        7. Confirm the result.

        Rules:
        - Create only high-level tasks.
        - Do not break tasks into smaller subtasks.
        - Do not execute tools.
        - Use the conversation history.
        - Do not invent missing information.
        - Revise the plan when the user changes requirements.
        - Include only tasks relevant to the current goal.
        - Return a numbered list of major tasks only.
    """

    # Generate the high-level plan without tool access.
    response = llm.invoke(
        [SystemMessage(content=planner_prompt)] + conversation
    )

    # Convert the model response into a list of plan steps.
    if isinstance(response.content, str):
        plan = [
            line.strip()
            for line in response.content.splitlines()
            if line.strip()
        ]
    else:
        plan = [
            str(block.get("text", "")).strip()
            for block in response.content
            if isinstance(block, dict) and block.get("text")
        ]

    print("\nHIGH-LEVEL PLAN:")
    for step in plan:
        print(step)

    # Return the high-level plan to LangGraph state.
    return {"plan": plan}


def decomposition_node(state: TravelState):
    """
    Task Decomposition Node: breaks each high-level plan item
    into smaller, actionable subtasks.
    """

    print("\n========== TASK DECOMPOSITION NODE ==========")

    conversation = state["messages"]
    plan = state.get("plan", [])
    previous_subtasks = state.get("subtasks", [])

    plan_text = "\n".join(plan)

    decomposition_prompt = f"""
        You are the Task Decomposition Node of a Travel Agent.

        Your responsibility is to break the Planner's high-level plan
        into smaller, actionable subtasks.

        HIGH-LEVEL PLAN:
        {plan_text}

        PREVIOUS SUBTASKS:
        {previous_subtasks}

        CONVERSATION HISTORY:
        Use the supplied conversation history to understand what the user
        has already provided and which activities have already been completed.

        For a flight search and booking request, possible subtasks include:
        - Ask for the departure city if it is missing.
        - Ask for the destination city if it is missing.
        - Confirm the travel date.
        - Collect the preferred departure-time range.
        - Collect the maximum budget.
        - Call search_flights when all search details are available.
        - Present the returned flight options.
        - Ask the user to select a flight.
        - Collect passenger name, age, gender, email and phone.
        - Obtain explicit booking approval.
        - Call book_flight only after approval.
        - Present the booking result.

        RULES:
        1. Break each relevant high-level task into clear, actionable subtasks.
        2. Put the subtasks in logical execution order.
        3. Do not execute any tools.
        4. Do not invent travel or passenger information.
        5. Do not assume the user has selected a flight.
        6. Do not assume the user has approved a booking.
        7. Exclude subtasks that are already completed, where clear from history.
        8. Include only subtasks relevant to the current user goal.
        9. Return a numbered list of subtasks only.
        """

    # Ask the LLM to decompose the high-level plan.
    response = llm.invoke(
        [SystemMessage(content=decomposition_prompt)] + conversation
    )

    # Convert the response into a list of actionable subtasks.
    if isinstance(response.content, str):
        subtasks = [
            line.strip()
            for line in response.content.splitlines()
            if line.strip()
        ]
    else:
        subtasks = [
            str(block.get("text", "")).strip()
            for block in response.content
            if isinstance(block, dict) and block.get("text")
        ]

    print("\nDECOMPOSED SUBTASKS:")
    for task in subtasks:
        print(task)

    # Save the decomposed subtasks in LangGraph state.
    return {"subtasks": subtasks}


def execution_node(state: TravelState):

    print("\n========== EXECUTOR NODE ==========")

    plan = state.get("plan", [])
    subtasks = state.get("subtasks", [])

    plan_text = "\n".join(plan)
    subtasks_text = "\n".join(subtasks)

    conversation = state["messages"]

    execution_prompt = f"""
        You are the Execution Node of a Travel Agent.

        HIGH-LEVEL PLAN:
        {plan_text}

        DECOMPOSED SUBTASKS:
        {subtasks_text}

        AVAILABLE TOOLS:

        1. search_flights:
        Searches one-way flights.
        Requires origin, destination, date, start_time, end_time and max_price.

        2. book_flight:
        Simulates booking a selected flight.
        Requires flight_selection, passenger_name, age, gender, email and phone.

        EXECUTION RULES:
        1. Read the complete conversation history.
        2. Use the decomposed subtasks as the execution guide.
        3. Identify which subtasks are already completed.
        4. Determine the next pending subtask.
        5. Ask for missing information one question at a time.
        6. Call search_flights only when all required search details are available.
        7. Present the returned flight options and wait for the user's selection.
        8. Collect all required passenger details before booking.
        9. Obtain explicit user approval before calling book_flight.
        10. Never invent flight results or passenger details.
        11. Do not repeat completed work unnecessarily.
        12. If the user changes the requirements, follow the updated plan and subtasks.

        The Planner creates the high-level plan.
        The Decomposition Node breaks it into smaller tasks.
        You execute those subtasks using the tools when appropriate.
        """

    response = llm_with_tools.invoke(
        [SystemMessage(content=execution_prompt)] + conversation
    )

    print("Executor determined the next action.")

    return {
        "messages": [response]
    }



graph = StateGraph(TravelState)

# Register the Planner, Decomposition, Executor and Tool nodes.
graph.add_node("planner", planner_node)
graph.add_node("decomposer", decomposition_node)
graph.add_node("executor", execution_node)
graph.add_node("tools", ToolNode(tools))

# Run planning first, then decompose the plan, then execute subtasks.
graph.add_edge(START, "planner")
graph.add_edge("planner", "decomposer")
graph.add_edge("decomposer", "executor")

# Route tool calls from the Executor to the ToolNode.
graph.add_conditional_edges("executor", tools_condition)

# Return to the Executor after a tool has finished running.
graph.add_edge("tools", "executor")

# Compile the graph into an executable application.
app = graph.compile()



messages = []
current_plan = []
current_subtasks = []
first_interaction = True

print("\n======================================")
print("     TASK DECOMPOSITION TRAVEL AGENT")
print("======================================")
print("Search flights and simulate booking.")
print("Type 'exit' to stop.")

while True:

    if first_interaction:
        prompt = input("\nWhere do you want to travel? ")
        first_interaction = False
    else:
        prompt = input("\nYou: ")

    if prompt.strip().lower() in ["exit", "quit"]:
        print("Thank you for using Travel Agent.")
        break

    # Add the user's message to the conversation history.
    messages.append(HumanMessage(content=prompt))

    # Invoke the workflow with messages, high-level plan and subtasks.
    result = app.invoke(
        {
            "messages": messages,
            "plan": current_plan,
            "subtasks": current_subtasks
        }
    )

    # Update the conversation history returned by LangGraph.
    messages = result["messages"]

    # Preserve the latest high-level plan and decomposed subtasks.
    current_plan = result.get("plan", current_plan)
    current_subtasks = result.get("subtasks", current_subtasks)

    # Display the latest response from the Travel Agent.
    print("\nTravel Agent:", messages[-1].content)
