# agent.py

import json
import re
import ast
from ollama_client import chat
from tools import get_search_results, summarize_paper, compare_papers, analyze_citations
from config import OLLAMA_MODEL

def extract_json(text: str) -> dict:
    """
    Attempt to extract a JSON object from the provided text.
    
    1. Try json.loads directly.
    2. If that fails, remove extraneous quotes/backticks and then
       repeatedly append a closing brace ("}") (up to 5 times) and try again.
    3. If still unsuccessful, try ast.literal_eval as a fallback.
    
    Returns an empty dict if all attempts fail.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    text = text.strip("`\"")
    attempt = text
    for i in range(5):
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            attempt += "}"
    try:
        result = ast.literal_eval(attempt)
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    return {}

class Agent:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.conversation = []  # List of messages (each with "role" and "content")
        
        # System instruction to tell the assistant how to handle tool calls.
        system_instructions = (
            "You are a research assistant. When you need to call a tool (e.g., to retrieve research papers, "
            "summarize a paper, compare papers, or analyze citations), output a JSON object with a key 'tool_calls' "
            "that includes the function name and arguments. Do not include any extra text or formatting."
        )
        self.conversation.append({"role": "system", "content": system_instructions})
        
        # Map tool names (and common synonyms) to their corresponding functions.
        self.tool_mapping = {
            "get_search_results": get_search_results,
            "summarize_paper": summarize_paper,
            "compare_papers": compare_papers,
            "analyze_citations": analyze_citations,
            "retrieve_paper": get_search_results,
            "get_research_papers": get_search_results,
            "search_papers": get_search_results,
            "retrieve_research_papers": get_search_results,
        }
        
        # Define tool schemas so the model knows which tools are available.
        self.tools_schema = [
            {
                'type': 'function',
                'function': {
                    'name': 'get_search_results',
                    'description': 'Retrieve research papers relevant to a query.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'query': {
                                'type': 'string',
                                'description': 'The search query for research papers.',
                            },
                        },
                        'required': ['query'],
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'summarize_paper',
                    'description': 'Summarize a research paper text.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'text': {
                                'type': 'string',
                                'description': 'The full text or excerpt of a research paper.',
                            },
                        },
                        'required': ['text'],
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'compare_papers',
                    'description': 'Compare two research paper texts.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'text1': {
                                'type': 'string',
                                'description': 'Text from the first research paper.',
                            },
                            'text2': {
                                'type': 'string',
                                'description': 'Text from the second research paper.',
                            },
                        },
                        'required': ['text1', 'text2'],
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'analyze_citations',
                    'description': 'Analyze the citations within a research paper text.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'text': {
                                'type': 'string',
                                'description': 'The text containing citations to analyze.',
                            },
                        },
                        'required': ['text'],
                    },
                },
            },
        ]

    def run(self, user_query: str) -> str:
        """
        One-shot method: processes a single query and returns a clean, final answer.
        This method appends the user's query to the conversation, processes any internal tool calls,
        and finally instructs the assistant to generate a polished answer.
        """
        self.conversation.append({"role": "user", "content": user_query})
        response = chat(model=self.model_name, messages=self.conversation, tools=self.tools_schema)
        content = response.get("message", {}).get("content", "")
        structured = extract_json(content)
        tool_calls = response.get("tool_calls", structured.get("tool_calls", []))
        if isinstance(tool_calls, dict):
            tool_calls = [tool_calls]
        if not tool_calls and "tools" in response:
            tool_calls = []
            for tool in response["tools"]:
                if isinstance(tool, dict) and "function_name" in tool:
                    tool_calls.append(tool)
                elif isinstance(tool.get("function"), dict) and "name" in tool["function"]:
                    tool_calls.append(tool["function"])
        while tool_calls:
            for tool_call in tool_calls:
                if "function_name" in tool_call:
                    tool_name = tool_call["function_name"]
                    tool_args = tool_call.get("arguments", {})
                elif isinstance(tool_call.get("function"), dict):
                    tool_name = tool_call["function"].get("name")
                    tool_args = tool_call["function"].get("arguments", {})
                elif "name" in tool_call:
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("parameters", {})
                else:
                    tool_name = tool_call.get("function")
                    tool_args = tool_call.get("arguments", {})
                
                # Fallback for required parameters: if "text" is missing for functions that need it, use the original query.
                if tool_name in ["analyze_citations", "summarize_paper"]:
                    if "text" not in tool_args or not tool_args["text"]:
                        tool_args["text"] = user_query
                
                if tool_name in self.tool_mapping:
                    tool_function = self.tool_mapping[tool_name]
                    try:
                        tool_result = tool_function(**tool_args)
                    except TypeError as e:
                        # If an error occurs (e.g., missing argument), use the original query as fallback.
                        tool_args["text"] = user_query
                        tool_result = tool_function(**tool_args)
                    self.conversation.append({"role": "tool", "content": tool_result})
                # Skip unknown tools silently.
            response = chat(model=self.model_name, messages=self.conversation)
            content = response.get("message", {}).get("content", "")
            structured = extract_json(content)
            tool_calls = response.get("tool_calls", structured.get("tool_calls", []))
            if isinstance(tool_calls, dict):
                tool_calls = [tool_calls]
            if not tool_calls and "tools" in response:
                tool_calls = []
                for tool in response["tools"]:
                    if isinstance(tool, dict) and "function_name" in tool:
                        tool_calls.append(tool)
                    elif isinstance(tool.get("function"), dict) and "name" in tool["function"]:
                        tool_calls.append(tool["function"])
        final_instruction = (
            "Based on all the information gathered so far, please now generate a final, polished answer "
            "to the original question in plain text. The answer should be concise, well-organized, "
            "and visually appealing, without any internal processing details."
        )
        self.conversation.append({"role": "system", "content": final_instruction})
        final_response = chat(model=self.model_name, messages=self.conversation)
        final_answer = (final_response.get("message", {}).get("content", "") or 
                        final_response.get("content", "")).strip()
        return final_answer

    def converse(self):
        """
        Interactive conversational loop. The conversation history is maintained so that the assistant's
        responses are context-aware.
        """
        print("Welcome to the Conversational Research Assistant!")
        print("Type your questions below (or type 'exit' to quit).")
        while True:
            user_input = input("You: ")
            if user_input.strip().lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            self.conversation.append({"role": "user", "content": user_input})
            response = chat(model=self.model_name, messages=self.conversation, tools=self.tools_schema)
            content = response.get("message", {}).get("content", "")
            structured = extract_json(content)
            tool_calls = response.get("tool_calls", structured.get("tool_calls", []))
            if isinstance(tool_calls, dict):
                tool_calls = [tool_calls]
            if not tool_calls and "tools" in response:
                tool_calls = []
                for tool in response["tools"]:
                    if isinstance(tool, dict) and "function_name" in tool:
                        tool_calls.append(tool)
                    elif isinstance(tool.get("function"), dict) and "name" in tool["function"]:
                        tool_calls.append(tool["function"])
            while tool_calls:
                for tool_call in tool_calls:
                    if "function_name" in tool_call:
                        tool_name = tool_call["function_name"]
                        tool_args = tool_call.get("arguments", {})
                    elif isinstance(tool_call.get("function"), dict):
                        tool_name = tool_call["function"].get("name")
                        tool_args = tool_call["function"].get("arguments", {})
                    elif "name" in tool_call:
                        tool_name = tool_call.get("name")
                        tool_args = tool_call.get("parameters", {})
                    else:
                        tool_name = tool_call.get("function")
                        tool_args = tool_call.get("arguments", {})
                    
                    if tool_name in ["analyze_citations", "summarize_paper"]:
                        if "text" not in tool_args or not tool_args["text"]:
                            tool_args["text"] = user_input
                    if tool_name in self.tool_mapping:
                        tool_function = self.tool_mapping[tool_name]
                        try:
                            tool_result = tool_function(**tool_args)
                        except TypeError:
                            tool_args["text"] = user_input
                            tool_result = tool_function(**tool_args)
                        self.conversation.append({"role": "tool", "content": tool_result})
                    # Skip unknown tools silently.
                response = chat(model=self.model_name, messages=self.conversation)
                content = response.get("message", {}).get("content", "")
                structured = extract_json(content)
                tool_calls = response.get("tool_calls", structured.get("tool_calls", []))
                if isinstance(tool_calls, dict):
                    tool_calls = [tool_calls]
                if not tool_calls and "tools" in response:
                    tool_calls = []
                    for tool in response["tools"]:
                        if isinstance(tool, dict) and "function_name" in tool:
                            tool_calls.append(tool)
                        elif isinstance(tool.get("function"), dict) and "name" in tool["function"]:
                            tool_calls.append(tool["function"])
            final_instruction = (
                "Based on everything so far, please now provide a final, conversational answer to the original question. "
                "The answer should be clear, engaging, and free of any internal processing details."
            )
            self.conversation.append({"role": "system", "content": final_instruction})
            final_response = chat(model=self.model_name, messages=self.conversation)
            answer = (final_response.get("message", {}).get("content", "") or 
                      final_response.get("content", "")).strip()
            if not answer:
                answer = "I'm sorry, I didn't quite catch that. Could you please rephrase?"
            print("Assistant:", answer)
            self.conversation.append({"role": "assistant", "content": answer})
