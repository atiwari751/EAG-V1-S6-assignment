from models import AgentState, MemoryItem, ToolInput, ToolResult
from typing import List, Optional
import json
from datetime import datetime
import os

class MemoryLayer:
    def __init__(self):
        """Initialize the memory layer"""
        self.state = AgentState()
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
    def get_state(self) -> AgentState:
        """Get the current agent state"""
        return self.state
        
    def record_action(self, tool_call: ToolInput) -> None:
        """Record an action being taken"""
        self.state.history.append(
            MemoryItem(
                iteration=self.state.iteration,
                action=tool_call
            )
        )
        
    def record_result(self, result: ToolResult) -> None:
        """Record the result of an action"""
        # Update the last memory item with the result
        if self.state.history:
            self.state.history[-1].result = result
            
    def increment_iteration(self) -> None:
        """Increment the iteration counter"""
        self.state.iteration += 1
        
    def set_task_complete(self, final_answer: str) -> None:
        """Mark the task as complete with final answer"""
        self.state.task_complete = True
        self.state.final_answer = final_answer
        
    def format_history_for_context(self) -> str:
        """Format the history for LLM context"""
        formatted = []
        
        for item in self.state.history:
            iteration = item.iteration + 1  # 1-indexed for display
            action = item.action
            result = item.result
            
            if action and result:
                formatted.append(
                    f"In iteration {iteration}, you called {action.name} with "
                    f"arguments {action.args}, and the function returned {result.content}."
                )
        
        return "\n\n".join(formatted)
        
    def reset(self) -> None:
        """Reset the memory state"""
        self.state = AgentState() 

    def save_state_to_file(self, filename=None) -> str:
        """Save the current agent state to a text file and return the filename"""
        if filename is None:
            # Generate a filename with timestamp if not provided
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"logs/agent_state_{timestamp}.txt"
        
        with open(filename, "w") as f:
            # Write a header
            f.write("=== AGENT STATE LOG ===\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Current Iteration: {self.state.iteration}\n")
            f.write(f"Task Complete: {self.state.task_complete}\n\n")
            
            if self.state.final_answer:
                f.write(f"Final Answer: {self.state.final_answer}\n\n")
            
            f.write("=== HISTORY ===\n")
            
            # Write each history item in a readable format
            for idx, item in enumerate(self.state.history):
                f.write(f"\n--- ITERATION {item.iteration} ---\n")
                
                if item.action:
                    f.write(f"Action: {item.action.name}\n")
                    f.write(f"Arguments: {json.dumps(item.action.args, indent=2)}\n")
                
                if item.result:
                    f.write(f"Success: {item.result.success}\n")
                    
                    # Truncate very long content for readability
                    content = item.result.content
                    if len(content) > 500:
                        content = content[:500] + "... [truncated]"
                    f.write(f"Result: {content}\n")
                    
                    if item.result.error:
                        f.write(f"Error: {item.result.error}\n")
            
            # Also include the raw JSON for programmatic analysis if needed
            f.write("\n\n=== RAW STATE (JSON) ===\n")
            f.write(json.dumps(self.state.model_dump(), indent=2))
            
        return filename 