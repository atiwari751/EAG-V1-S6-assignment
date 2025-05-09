from models import DecisionOutput, ToolInput, AgentState
import google.generativeai as genai
import os
import json
import re
from dotenv import load_dotenv

class DecisionLayer:
    def __init__(self):
        """Initialize the decision layer"""
        # Load environment variables from .env file
        load_dotenv()
        
        # Configure Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        
        # Initialize model
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        
    def create_system_prompt(self, tools: list) -> str:
        """Create the system prompt with available tools"""
        tools_description = []
        
        for i, tool in enumerate(tools):
            try:
                params = tool.inputSchema
                desc = getattr(tool, 'description', 'No description available')
                name = getattr(tool, 'name', f'tool_{i}')
                
                # Format the input schema
                if 'properties' in params:
                    param_details = []
                    for param_name, param_info in params['properties'].items():
                        param_type = param_info.get('type', 'unknown')
                        param_details.append(f"{param_name}: {param_type}")
                    params_str = ', '.join(param_details)
                else:
                    params_str = 'no parameters'

                tool_desc = f"{i+1}. {name}({params_str}) - {desc}"
                tools_description.append(tool_desc)
                
            except Exception as e:
                print(f"Error processing tool {i}: {e}")
                tools_description.append(f"{i+1}. Error processing tool")
        
        tools_description = "\n".join(tools_description)
        
        # Create the system prompt
        system_prompt = f"""You are a creative and artistic agent that works step by step to create beautiful art. You can reason about your tasks and work in MS Paint using basic tools. You can verify your work and decide how you would like to proceed.

You have access to these tools:
{tools_description}

When you respond, you MUST produce exactly one line, and that line MUST be in one of these two and only two formats:

  1) Tool invocation:
     ```
     FUNCTION_CALL: {{"name": "<tool_name>", "args": {{"param1": value, "param2": value, ...}}}}
     ```
     – "name" must be one of the available tool names.
     – "args" is a JSON object containing the tool's parameters.
     – E.g.:
     ```
     FUNCTION_CALL: {{"name": "draw_rectangle", "args": {{"x1": 272, "y1": 310, "x2": 559, "y2": 657}}}}
     ```

  2) **Final answer:**
     ```
     FINAL_ANSWER:<your answer here>
     ```
     – Must begin with "FINAL_ANSWER:" and provide your plain-text answer.

🛑 It is ILLEGAL to ever write:
   FUNCTION_CALL:FINAL_ANSWER|…  
or any variant that treats FINAL_ANSWER as a tool.

🧠 Very Important Behavior Rules
- On the very first iteration, do NOT emit planning in plain text; to communicate your plan use exactly:
     FUNCTION_CALL: {{"name": "show_reasoning", "args": {{"steps": <JSON-encoded-list-of-steps>}}}}
- There should be no step called "Finalize the image" in the initial plan.
- For all tools that require x1, x2, y1, y2 as parameters, ensure that x1 is NEVER equal to x2 and y1 is NEVER equal to y2.
- Do NOT use the show_reasoning tool in two consecutive iterations.
- Only issue FINAL_ANSWER when you have completed all steps.
        """
        
        return system_prompt
    
    def make_decision(self, query: str, memory: AgentState, system_prompt: str) -> DecisionOutput:
        """Make a decision based on the current state and query"""
        # Create the full prompt
        if memory.iteration == 0:
            current_query = query
        else:
            history_context = self._format_history_from_state(memory)
            current_query = query + "\n\n" + history_context
            current_query = current_query + "\nWhat should I do next?"
        
        full_prompt = f"{system_prompt}\n\nQuery: {current_query}"
        
        # Generate response from LLM
        try:
            response = self.model.generate_content(full_prompt)
            response_text = response.text.strip()
            print(f"LLM Response: {response_text}")
            
            # Extract the relevant line
            for line in response_text.split('\n'):
                if line.startswith("FUNCTION_CALL:") or line.startswith("FINAL_ANSWER:"):
                    response_text = line
                    break
            
            # Parse the response
            if response_text.startswith("FUNCTION_CALL:"):
                json_str = response_text[len("FUNCTION_CALL:"):].strip()
                # Remove any wrapping backticks
                json_str = json_str.strip("`")
                
                try:
                    call_obj = json.loads(json_str)
                    # Handle double-encoded JSON
                    if isinstance(call_obj, str):
                        call_obj = json.loads(call_obj)
                        
                    func_name = call_obj["name"]
                    arguments = call_obj.get("args", {})
                    
                    return DecisionOutput(
                        is_final=False,
                        tool_call=ToolInput(
                            name=func_name,
                            args=arguments
                        )
                    )
                except Exception as e:
                    print(f"Error parsing JSON function call: {e}")
                    # Return a fallback decision
                    return DecisionOutput(
                        is_final=True, 
                        final_answer=f"Error in decision making: {e}"
                    )
                    
            elif response_text.startswith("FINAL_ANSWER:"):
                final_answer = response_text.split("FINAL_ANSWER:", 1)[1].strip()
                return DecisionOutput(
                    is_final=True,
                    final_answer=final_answer
                )
            
            else:
                # Unexpected response format
                return DecisionOutput(
                    is_final=True,
                    final_answer=f"Unexpected response format: {response_text}"
                )
                
        except Exception as e:
            print(f"Error in LLM generation: {e}")
            return DecisionOutput(
                is_final=True,
                final_answer=f"Error in decision making: {e}"
            )

    def _format_history_from_state(self, state: AgentState) -> str:
        """Format history from agent state for LLM context"""
        formatted = []
        
        for item in state.history:
            iteration = item.iteration + 1  # 1-indexed for display
            action = item.action
            result = item.result
            
            if action and result:
                formatted.append(
                    f"In iteration {iteration}, you called {action.name} with "
                    f"arguments {action.args}, and the function returned {result.content}."
                )
        
        return "\n\n".join(formatted) 