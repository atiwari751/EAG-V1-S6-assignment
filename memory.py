from models import AgentState, MemoryItem, ToolInput, ToolResult
from typing import List, Optional

class MemoryLayer:
    def __init__(self):
        """Initialize the memory layer"""
        self.state = AgentState()
        
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