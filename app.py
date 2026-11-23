
import streamlit as st

from chatbot_backend import (
    process_text,
    chatbot_response
)

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="NLP Chatbot",
    page_icon="🤖",
    layout="centered"
)

# =====================================
# DESIGN - SCREENSHOT MATCH
# =====================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background: #FFFCF6 !important;
}

/* Header */
header[data-testid="stHeader"] {
    background: #FFFCF6 !important;
}

/* Main chat width */
.block-container {
    max-width: 1050px !important;
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
}

/* User and chatbot cards */
div[data-testid="stChatMessage"] {
    padding: 22px 26px !important;
    border-radius: 24px !important;
    margin-bottom: 28px !important;
    box-shadow: 0 8px 24px rgba(100, 90, 70, 0.05) !important;
}

/* USER - PEACH */
div[data-testid="stChatMessage"]:has(
    [data-testid="chatAvatarIcon-user"]
) {
    background: #F9EBDD !important;
    border: 1px solid #F0D5B7 !important;
    margin-left: 8% !important;
}

/* CHATBOT - WHITE */
div[data-testid="stChatMessage"]:has(
    [data-testid="chatAvatarIcon-assistant"]
) {
    background: #FFFFFF !important;
    border: 1px solid #DDE8D8 !important;
    margin-right: 8% !important;
}

/* Hide avatars */
div[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"],
div[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] {
    display: none !important;
}

/* Message text */
div[data-testid="stChatMessage"] p {
    color: #485849 !important;
    font-size: 17px !important;
    line-height: 1.9 !important;
}

/* Input box */
div[data-testid="stChatInput"] {
    background: #F0F4E9 !important;
    border: 1px solid #D0DCC9 !important;
    border-radius: 14px !important;
    padding: 8px !important;
    box-shadow: 0 8px 22px rgba(100, 90, 70, 0.05) !important;
}

/* Input text */
div[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #465448 !important;
    font-size: 16px !important;
}

/* Placeholder */
div[data-testid="stChatInput"] textarea::placeholder {
    color: #96A397 !important;
}

/* Send button */
div[data-testid="stChatInput"] button {
    background: #E5EBD8 !important;
    border-radius: 12px !important;
    color: #80917D !important;
    border: none !important;
}

/* Clear chat button */
.stButton button {
    background: #F9EBDD !important;
    color: #76513C !important;
    border: 1px solid #E6C8AD !important;
    border-radius: 12px !important;
}

/* NLP details expander - pastel blue */
div[data-testid="stExpander"] {
    background: #DCEAF7 !important;
    border: 1px solid #C8DCEB !important;
    border-radius: 14px !important;
}

/* Expander internal area */
div[data-testid="stExpander"] details,
div[data-testid="stExpander"] details > div,
div[data-testid="stExpander"] details div[data-testid="stExpanderDetails"] {
    background-color: #DCEAF7 !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-thumb {
    background: #D9BCA3;
    border-radius: 10px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

</style>
""", unsafe_allow_html=True)


# =====================================
# HEADER
# =====================================

st.title("🤖 NLP Based Chatbot")

st.caption(
    "An interactive chatbot using Natural Language Processing, "
    "NLTK and corpus-based text retrieval."
)

st.markdown(
    "Ask questions, explore NLP concepts, or chat with the bot."
)


# =====================================
# SESSION STATE
# =====================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_context" not in st.session_state:
    st.session_state.chat_context = {}


# =====================================
# CLEAR CHAT
# =====================================

if st.button("🗑️ Clear Chat"):

    st.session_state.messages = []
    st.session_state.chat_context = {}

    st.rerun()


# =====================================
# DISPLAY CHAT HISTORY
# =====================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # NLP details for user messages
        if message["role"] == "user" and "analysis" in message:

            with st.expander("🔍 NLP Processing Details"):

                analysis = message["analysis"]

                st.write("**Normalized Text**")
                st.code(analysis["Normalized Text"])

                st.write("**Tokens**")
                st.code(str(analysis["Tokens"]))

                st.write("**Stopword Removal**")
                st.code(str(analysis["Stopword Removal"]))

                st.write("**POS Tagging**")
                st.code(str(analysis["POS Tagging"]))

                st.write("**Stemming**")
                st.code(str(analysis["Stemming"]))

                st.write("**Lemmatization**")
                st.code(str(analysis["Lemmatization"]))


# =====================================
# CHAT INPUT
# =====================================

user_input = st.chat_input("Write a message...")

if user_input:

    # NLP processing
    analysis = process_text(user_input)

    # Chatbot response
    response, updated_context = chatbot_response(
        user_input,
        analysis,
        st.session_state.chat_context
    )

    st.session_state.chat_context = updated_context

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "analysis": analysis
    })

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    st.rerun()
