import os
import json
from dotenv import load_dotenv
from langchain_ibm import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.agents import Tool, initialize_agent

# Load environment variables
load_dotenv()

credentials = {
    "url": os.getenv("WATSON_URL", "https://us-south.ml.cloud.ibm.com"),
    "apikey": os.getenv("WATSON_API_KEY"),
}
project_id = os.getenv("WATSON_PROJECT_ID")

# Initialize the WatsonxLLM
llm = WatsonxLLM(
    # model_id="ibm/granite-3-8b-instruct",
    model_id="ibm/granite-13b-chat-v2",
    url=credentials.get("url"),
    apikey=credentials.get("apikey"),
    project_id=project_id,
    params={
        GenParams.DECODING_METHOD: "greedy",
        GenParams.TEMPERATURE: 0,
        GenParams.MIN_NEW_TOKENS: 5,
        GenParams.MAX_NEW_TOKENS: 70,
        # GenParams.STOP_SEQUENCES: ["Human:", "Observation"],
    },
)

# Output Parser
parser = StrOutputParser()

# Chat history store
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


config = {"configurable": {"session_id": "default_session"}}

# Create the prompt and chain
chatBot_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
            Guidelines for the conversation:

            1. Always respond **only as the assistant**. Do **not** speak for or continue the user's message. Your response should only reflect your role as an assistant and not represent the user's thoughts, feelings, or intentions.
            2. Respond to each **user message with a single, clear response**. Do not continue the conversation on behalf of the user.
            3. Your responses should be empathetic, supportive, and non-judgmental, but always as the assistant.
            4. Do not make assumptions or guesses about the user's emotions, experiences, or thoughts. Focus on providing helpful, direct responses to their input.
        """),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
chain = chatBot_prompt | llm
model_with_memory = RunnableWithMessageHistory(chain, get_session_history)


def predict(human_msg: str, session_id: str = "default_session"):
    """Generate a response for a given user message."""
    # Add user message to the chat history
    chat_history = get_session_history(session_id)
    chat_history.add_message(HumanMessage(content=human_msg))

    # Format the prompt with the conversation history, ensuring the model can distinguish messages
    prompt = chatBot_prompt.format_messages(messages=chat_history.messages)

    # Process the model response
    response = llm.invoke(prompt)

    # Clean up AI response
    cleaned_response = response.replace(
        "AI:", "").replace("Human:", "").strip()

    # Store AI response in chat history
    chat_history.add_message(AIMessage(content=cleaned_response))

    return cleaned_response


def sanitize_and_parse_json(response: str):
    if not response or not response.strip():
        return {"error": "Empty response from the LLM", "raw_response": response}

    try:
        # Attempt to find the JSON object in the response
        start = response.find("{")
        end = response.rfind("}") + 1
        json_str = response[start:end]
        parsed_json = json.loads(json_str)

        # Ensure it contains required keys
        if "name_of_employee" in parsed_json and "satisfaction" in parsed_json:
            return parsed_json
        else:
            raise ValueError("Missing required keys in JSON")

    except (json.JSONDecodeError, ValueError) as e:
        return {
            "error": f"JSON parsing failed: {str(e)}",
            "raw_response": response,
        }

# Function for analysis
def analyze_conversation_tool(conversation):
    """Analyze the conversation JSON to determine satisfaction."""
    # Format the conversation as input text
    messages = [
        f"Human: {msg.get('Human', '')} AI: {msg.get('AI', '')}" for msg in conversation
    ]
    input_text = "\n".join(messages)

    # Combine the prompt and input text
    request = f"""
        You are an HR assistant bot tasked with analyzing employee satisfaction based on their conversation history.
        Your job is to:
        1. Review the conversation between Human and AI to assess the tone, mood, and overall sentiment.
        2. Determine the satisfaction level as one of the following: Bad, Average, Good.
        3. Extract the employee's name if mentioned, otherwise leave it blank.

        Pay special attention to words and phrases that indicate dissatisfaction or negative emotions, such as:
        - "not feeling good"
        - "stuck"
        - "unsure"
        - "worried"
        - "not doing enough"
        - "not good enough"

        Below are examples to guide you:

        Example 1:
        Conversation:
        "Human: I'm not feeling good about my current job. I'm worried about my future."
        "AI: I'm sorry to hear that. Can I help with career advice?"
        Output:
        {{
            "name_of_employee": "",
            "satisfaction": "Bad"
        }}

        Example 2:
        Conversation:
        "Human: Good evening, I'm feeling optimistic about my new role."
        "AI: That's wonderful to hear. Keep up the great work!"
        Output:
        {{
            "name_of_employee": "",
            "satisfaction": "Good"
        }}

        JUST Output a **SINGLE** JSON with the following format:
        {{
            "name_of_employee": "<employee_name>",
            "satisfaction": "<Bad, Average, or Good>"
        }}

        Conversation:
        {input_text}

        IMPORTANT: Ensure the output is a valid JSON object with no additional text or comments.
    """

    try:
        # Invoke the LLM
        response = llm.invoke(request)

        # Log raw response for debugging
        print("Debug: Raw LLM Response:", response)

        # Sanitize and parse the response
        result = sanitize_and_parse_json(response)

        # Ensure the result contains required keys
        if "name_of_employee" in result and "satisfaction" in result:
            return result
        else:
            raise ValueError("Missing required keys in the JSON response.")
    except Exception as e:
        # Handle any exceptions gracefully
        return {
            "error": f"Failed to analyze conversation: {str(e)}",
            "raw_response": response if 'response' in locals() else None,
        }


# Function to analyze and generate satisfaction JSON
def analyze_chat_and_rate(chat_json):
    """Takes the conversation JSON and returns satisfaction rating."""
    try:
        analysis = analyze_conversation_tool(chat_json)
        return analysis
    except Exception as e:
        return {"error": str(e)}
