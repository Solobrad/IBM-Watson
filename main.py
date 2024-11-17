import streamlit as st
import json
from chatbot import predict, get_session_history, analyze_chat_and_rate
# Import the database functions
from db import init_db, insert_analysis, fetch_analysis
import pandas as pd

# Initialize the database
init_db()

# Streamlit page setup
st.title("Employee Satisfaction Chatbot")
st.subheader("Monitor employee mood and challenges with ease.")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "session_id" not in st.session_state:
    st.session_state["session_id"] = "employee_chat"

# Chat interface
with st.container():
    st.write("### Chat Interface")
    user_input = st.text_input("You:", key="user_input")

    if st.button("Send"):
        if user_input:
            # Get chatbot response
            response = predict(user_input, session_id=st.session_state["session_id"])
            st.session_state["messages"].append({"Human": user_input, "AI": response})
            user_input = ""

# Display chat messages
st.write("### Conversation")
for msg in st.session_state["messages"]:
    st.write(f"**You:** {msg['Human']}")
    st.write(f"**Bot:** {msg['AI']}")

# Display JSON output
st.write("### Conversation in JSON")
st.json(st.session_state["messages"])


# Analyze conversation and rate satisfaction
if st.button("Analyze Satisfaction"):
    if st.session_state["messages"]:
        chat_json = st.session_state["messages"]
        analysis = analyze_chat_and_rate(chat_json)

        if "error" not in analysis:
            name = analysis.get("name_of_employee", "")
            satisfaction = analysis.get("satisfaction", "")
            # conversation = json.dumps(chat_json)
            insert_analysis(name, satisfaction)  # Save to DB

            st.success("Analysis saved to database!")
            st.write("### Satisfaction Analysis")
            st.json(analysis)
        else:
            st.error(analysis["error"])
    else:
        st.warning("No conversation data available to analyze!")


st.write("### Stored Analysis Data")

# Fetch data from the database
analysis_data = fetch_analysis()

if analysis_data:
    # Display data in tabular form
    df = pd.DataFrame(analysis_data, columns=["ID", "Name", "Satisfaction"])
    st.dataframe(df)

    # Plot bar chart
    satisfaction_counts = df["Satisfaction"].value_counts()
    st.bar_chart(satisfaction_counts)
else:
    st.warning("No analysis data available!")
