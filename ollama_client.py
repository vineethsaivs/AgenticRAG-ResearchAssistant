# ollama_client.py

import json
import subprocess

def chat(model: str, messages: list, tools: list = None) -> dict:
    """
    Calls the Ollama model (e.g., Qwen2.5) via the command line.
    
    Parameters:
      - model: the model name (e.g., "qwen2.5")
      - messages: a list of message dictionaries (each with a role and content)
      - tools: (optional) a list of tool schema dictionaries

    Returns:
      A dictionary representing the response.
    """
    payload = {
        "model": model,
        "messages": messages,
    }
    if tools is not None:
        payload["tools"] = tools

    # Convert the payload to a JSON string.
    input_str = json.dumps(payload)

    try:
        # Run the Ollama CLI command. Make sure “ollama” is in your PATH.
        result = subprocess.run(
            ["ollama", "run", model],
            input=input_str,
            text=True,
            capture_output=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print("Error calling Ollama:", e.stderr)
        raise

    # Try parsing the output as JSON.
    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError:
        # Fallback: if output is plain text.
        response = {"message": {"content": result.stdout}}
    
    return response
