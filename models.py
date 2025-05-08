from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Any

class ToolInput(BaseModel):
    """Input parameters for tool execution"""
    name: str
    args: Dict[str, Any] = {}

class ToolResult(BaseModel):
    """Result from tool execution"""
    success: bool
    content: str
    error: Optional[str] = None

class MemoryItem(BaseModel):
    """An item in agent memory"""
    iteration: int
    action: Optional[ToolInput] = None
    result: Optional[ToolResult] = None
    
class AgentState(BaseModel):
    """Current state of the agent"""
    iteration: int = 0
    history: List[MemoryItem] = []
    task_complete: bool = False
    final_answer: Optional[str] = None

class DecisionOutput(BaseModel):
    """Output from the decision layer"""
    is_final: bool = False
    final_answer: Optional[str] = None
    tool_call: Optional[ToolInput] = None 