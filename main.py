from perception import PerceptionLayer
from memory import MemoryLayer
from decision import DecisionLayer
from action import ActionLayer
from models import ToolInput
from rich.console import Console
import time

console = Console()

def main():
    """Main function to run the agent"""
    # Initialize all layers
    console.print("[bold cyan]Initializing cognitive layers...[/]")
    perception = PerceptionLayer()
    memory = MemoryLayer()
    decision = DecisionLayer()
    action = ActionLayer()
    
    try:
        # Start the MCP server
        console.print("[bold cyan]Starting MCP server...[/]")
        if not action.start_mcp_server():
            console.print("[bold red]Failed to start MCP server. Exiting.[/]")
            return
            
        # Wait for tools to be loaded
        console.print("[bold cyan]Waiting for tools to load...[/]")
        time.sleep(2)  # Give a bit of time for initialization
        tools = action.get_tools()
        if not tools:
            console.print("[bold red]Failed to get tools. Exiting.[/]")
            return
            
        console.print(f"[green]Successfully loaded {len(tools)} tools[/]")
        
        # Create system prompt
        system_prompt = decision.create_system_prompt(tools)
        
        # Get user query
        query = """Get creative with shapes! Open paint and draw a rectangle with corner points (272,310) and (559, 657). Then draw some more ovals, rectangles, and arrows in the canvas. The final image should represent an abstract idea from Greek mythology. Add text in the canvas to describe the idea you've drawn."""
        processed_query = perception.process_user_query(query)
        
        console.print(f"[bold magenta]User Query:[/] {processed_query}")
        console.print("[bold cyan]Beginning agent execution loop...[/]")
        
        # Agent execution loop
        while not memory.get_state().task_complete:
            # Get decision from decision layer
            decision_output = decision.make_decision(
                processed_query, 
                memory.get_state(),
                system_prompt
            )
            
            if decision_output.is_final:
                # Task complete, store final answer
                console.print(f"[bold green]Task Complete:[/] {decision_output.final_answer}")
                memory.set_task_complete(decision_output.final_answer)
            else:
                # Execute the tool
                tool_call = decision_output.tool_call
                console.print(f"[cyan]Executing tool:[/] {tool_call.name}")
                
                # Record the action in memory
                memory.record_action(tool_call)
                
                # Execute in action layer
                result = action.execute_tool(tool_call)
                
                # Process the result
                processed_result = perception.process_tool_result(result, tool_call.name)
                
                # Store result in memory
                memory.record_result(processed_result)
                
                # Increment iteration counter
                memory.increment_iteration()
                
                # Only print result if it's not from show_reasoning (already printed)
                if tool_call.name != "show_reasoning":
                    console.print(f"[green]Result:[/] {processed_result.content}")
                
                # Add a waiting message for next decision
                console.print("[cyan]Waiting for next action decision from LLM...[/]")
                
        console.print("[bold green]=== Agent Execution Complete ===[/]")
        
    except KeyboardInterrupt:
        console.print("[yellow]User interrupted execution[/]")
    except Exception as e:
        console.print(f"[bold red]Error in main execution: {str(e)}[/]")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        action.stop()
        memory.reset()
        console.print("[bold cyan]Agent resources cleaned up[/]")

if __name__ == "__main__":
    main() 