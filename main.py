import streamlit as st
import json
from chatbot import predict, get_session_history, analyze_chat_and_rate
from db import init_db, insert_analysis, fetch_analysis
import pandas as pd

# Initialize the database
init_db()

# Streamlit page setup
st.title("Employee Satisfaction Chatbot")
st.subheader("Monitor employee mood and challenges with ease.")

# Session state for chat history, user name, and other flags
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "session_id" not in st.session_state:
    st.session_state["session_id"] = "employee_chat"

if "name" not in st.session_state:
    st.session_state["name"] = None

if "greeting_sent" not in st.session_state:
    st.session_state["greeting_sent"] = False

if "initial_prompt_sent" not in st.session_state:
    st.session_state["initial_prompt_sent"] = False


def clean_ai_response(response: str) -> str:
    """Clean the AI response to remove unintended text."""
    if "Assistant:" in response:
        response = response.replace("Assistant:", "").strip()
    return response


def clear_chat():
    """Clear chat history and reset session state."""
    st.session_state["messages"] = []
    st.session_state["name"] = None
    st.session_state["greeting_sent"] = False
    st.session_state["initial_prompt_sent"] = False


# Add the Clear Chat button
if st.button("Clear Chat"):
    clear_chat()

# Step 1: Prompt for user name with a greeting exchange
if st.session_state["name"] is None:
    st.write("### Enter Your Name")

    # Display conversation
    for msg in st.session_state["messages"]:
        if "Human" in msg:
            st.write(f"**You:** {msg['Human']}")
        if "AI" in msg:
            st.write(f"**Bot:** {msg['AI']}")

    # Text input for user name
    user_name = st.text_input("Your Name", key="user_name")

    if st.button("Check Name"):
        if user_name.strip():
            # Add user's response to the conversation
            st.session_state["messages"].append({"Human": user_name, "AI": "Checking name..."})

            # Fetch existing analysis data from the database
            analysis_data = fetch_analysis()

            # Check if the name exists in the database
            existing_user = any(row[1] == user_name for row in analysis_data)

            if existing_user:
                ai_response = f"Welcome back, {user_name}! Your previous records exist in the database."
                st.success(ai_response)
            else:
                ai_response = f"Nice to meet you, {user_name}! Let's get started."
                st.info(ai_response)

            # Store the name in session state
            st.session_state["name"] = user_name
        else:
            st.warning("Please enter your name.")

# Step 2: Chat interface after name is provided
if st.session_state["name"]:
    st.write(f"### Chat Interface for {st.session_state['name']}")

    # AI's initial question if not already asked
    if not st.session_state["initial_prompt_sent"]:
        initial_message = "Could you tell us your feelings about XYZ Corp work environment?"

    # Display chat messages
    st.write("### Conversation")
    for msg in st.session_state["messages"]:
        if "Human" in msg:
            st.write(f"**You:** {msg['Human']}")
        if "AI" in msg:
            st.write(f"**Bot:** {msg['AI']}")

    # Text input for user input
    user_input = st.text_input("You:", key="user_input")

    if st.button("Send"):
        if user_input.strip():
            # Add user's message to chat history
            st.session_state["messages"].append({"Human": user_input, "AI": "Typing..."})

            # Get chatbot response
            response = predict(user_input, session_id=st.session_state["session_id"])

            # Clean and add AI response to chat history
            cleaned_response = clean_ai_response(response)
            if cleaned_response:
                st.session_state["messages"].append({"AI": cleaned_response, "Human": user_input})
            else:
                st.warning("The chatbot response was empty or invalid.")

            # Clear user input
            user_input = ""

    st.write("### Conversation in JSON")
    st.json(st.session_state["messages"])

    if st.button("Analyze Satisfaction"):
        if st.session_state["messages"]:
            chat_json = st.session_state["messages"]
            analysis = analyze_chat_and_rate(chat_json)

            if "error" not in analysis:
                name = st.session_state["name"]
                satisfaction = analysis.get("satisfaction", "")
                insert_analysis(name, satisfaction)  # Save to DB

                st.success("Analysis saved to database!")
                st.write("### Satisfaction Analysis")
                st.json(analysis)
            else:
                st.error(analysis["error"])
        else:
            st.warning("No conversation data available to analyze!")

col1, col2 = st.columns(2)

with col1:
    st.write("### Stored Analysis Data")

    analysis_data = fetch_analysis()

    if analysis_data:
        df = pd.DataFrame(analysis_data, columns=["ID", "Name", "Satisfaction"])
        st.dataframe(df)
    else:
        st.warning("No analysis data available!")

with col2:
    if analysis_data:
        st.write("### Satisfaction Visualization")
        satisfaction_counts = df["Satisfaction"].value_counts()
        st.bar_chart(satisfaction_counts)