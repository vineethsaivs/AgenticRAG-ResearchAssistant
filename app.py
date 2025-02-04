import streamlit as st
import warnings
# Optionally suppress deprecation warnings:
warnings.filterwarnings("ignore", category=DeprecationWarning)

from agent import Agent
from config import OLLAMA_MODEL

st.title("Research Assistant")
st.write("Ask questions about research papers or request summaries, comparisons, and citation analyses.")

# A text input for the user query.
user_query = st.text_input("Enter your query:")

# When the user clicks "Submit", run the agent and display the answer.
if st.button("Submit"):
    if user_query:
        # Create an instance of the Agent.
        agent = Agent(model_name=OLLAMA_MODEL)
        # Run the agent's processing (this uses your run() method that returns a final answer).
        final_answer = agent.run(user_query)
        st.markdown("### Final Answer")
        st.write(final_answer)
    else:
        st.write("Please enter a query.")
