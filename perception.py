from models import ToolResult
import json
from rich.console import Console
from rich.panel import Panel
import re
from typing import Any

console = Console()

class PerceptionLayer:
    def __init__(self):
        """Initialize the perception layer"""
        pass
        
    def process_user_query(self, query: str) -> str:
        """Process the initial user query"""
        # Simple processing, could be expanded to extract key information
        return query.strip()
    
    def process_tool_result(self, result: Any, tool_name: str) -> ToolResult:
        """Process the result from a tool execution"""
        try:
            # If we already have a ToolResult, just return it
            if isinstance(result, ToolResult):
                # Special case for show_reasoning - don't reprocess
                if tool_name == "show_reasoning":
                    return result
                    
                # For other tools, maybe add additional processing if needed
                return result
                
            # Otherwise extract content as string
            if hasattr(result, 'content'):
                if isinstance(result.content, list):
                    content_str = " ".join([
                        item.text if hasattr(item, 'text') else str(item)
                        for item in result.content
                    ])
                else:
                    content_str = str(result.content)
            else:
                content_str = str(result)
                
            return ToolResult(
                success=True,
                content=content_str
            )
        except Exception as e:
            console.print(f"[bold red]Error in perception layer: {e}[/]")
            import traceback
            traceback.print_exc()
            return ToolResult(
                success=False,
                content="",
                error=str(e)
            )
    
    def _render_reasoning_steps(self, steps):
        """Render reasoning steps with rich panels"""
        if isinstance(steps, str):
            try:
                steps_list = json.loads(steps)
            except json.JSONDecodeError:
                steps_list = [
                    s.strip() for s in re.split(r"[;,]", steps)
                    if s.strip()
                ]
        else:
            steps_list = steps
            
        for idx, step in enumerate(steps_list, start=1):
            console.print(
                Panel(
                    step,
                    title=f"Step {idx}",
                    border_style="cyan",
                    expand=False,
                )
            ) 