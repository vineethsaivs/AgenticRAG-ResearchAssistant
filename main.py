# main.py

from agent import Agent
from config import OLLAMA_MODEL

def main():
    agent = Agent(model_name=OLLAMA_MODEL)
    agent.converse()

if __name__ == "__main__":
    main()
