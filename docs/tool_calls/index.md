# Tool Calls


## Simple call from OpenRouter to Qwen

```python 
import json
from openai import OpenAI

# It's good practice to use environment variables for API keys
# For this example, we'll use a placeholder.
# from dotenv import load_dotenv
# import os
# load_dotenv()
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="", # Replace with your key
)

# --- Step 1: Define your tool and the function it calls ---

# This is the actual Python function that will be executed.
def get_current_weather(location, unit="celsius"):
    """Get the current weather in a given location."""
    # In a real application, you would call a weather API here.
    # For this example, we'll return mock data.
    if "astana" in location.lower():
        weather_info = {
            "location": location,
            "temperature": "-15",
            "unit": unit,
            "forecast": ["snowy", "windy", "cold"],
        }
    else:
        weather_info = {
            "location": location,
            "temperature": "22",
            "unit": unit,
            "forecast": ["sunny", "mild"],
        }
    return json.dumps(weather_info)

# --- Step 2: Make the first API call with the tools defined ---

# The user's prompt that should trigger the tool.
user_prompt = "What's the weather like in Astana?"
print(f"👤 User: {user_prompt}\n")

messages = [{"role": "user", "content": user_prompt}]

# Describe your tool in the JSON schema format the model expects.
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g., San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    },
                },
                "required": ["location"],
            },
        },
    }
]

# First API call
completion = client.chat.completions.create(
  model="qwen/qwen3-235b-a22b-2507",
  messages=messages,
  tools=tools,
  tool_choice="auto", # 'auto' lets the model decide, or you can force a tool call.
)

response_message = completion.choices[0].message
tool_calls = response_message.tool_calls

# --- Step 3: Check if the model wants to call a tool and execute it ---

if tool_calls:
    print("🤖 Model wants to call a tool...")
    print(f"Tool calls: {tool_calls}\n")
    
    # Append the assistant's message with tool calls to the conversation history
    messages.append(response_message)
    
    # In this example, we'll use a mapping to find the correct function.
    available_functions = {
        "get_current_weather": get_current_weather,
    }
    
    # Loop through each tool call the model requested
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_functions[function_name]
        function_args = json.loads(tool_call.function.arguments)
        
        print(f"Executing function: {function_name}({function_args})")
        
        # Call the actual function with the arguments provided by the model
        function_response = function_to_call(
            location=function_args.get("location"),
            unit=function_args.get("unit"),
        )
        
        print(f"Function response: {function_response}\n")

        # --- Step 4: Send the function's response back to the model in a second call ---
        
        # Append the tool's response to the conversation history
        messages.append(
            {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            }
        )

    print("📢 Sending tool response back to the model...")
    
    # Second API call
    second_response = client.chat.completions.create(
        model="qwen/qwen3-235b-a22b-2507",
        messages=messages, # Send the whole conversation history
    )
    
    final_message = second_response.choices[0].message.content
    print(f"\n✅ Final Model Response:\n{final_message}")

else:
    # If the model didn't call a tool, just print its response
    print(f"🤖 Model (no tool call):\n{response_message.content}")
```