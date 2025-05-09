from perception import PerceptionLayer
from memory import MemoryLayer
from decision import DecisionLayer
from action import ActionLayer
from models import ToolInput, UserQuery
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
        
        # Get user input interactively
        console.print("[bold magenta]Welcome to Paint Agent![/]")
        console.print("[bold cyan]Please answer the following questions to begin:[/]")
        
        # Get style preference
        console.print("[yellow]What is your preference for the style of image to create? (simple, experimental, abstract, regular, etc.)[/]")
        style_preference = input("> ")
        
        # Get image description
        console.print("[yellow]Describe the image you want to create:[/]")
        description = input("> ")
        
        # Create structured query
        user_query = UserQuery(
            description=description,
            style_preference=style_preference
        )
        
        # Process the query
        processed_query = perception.process_user_query(user_query)
        
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
                
                # Log the state to file
                state_file = memory.save_state_to_file()
                console.print(f"[dim]Agent state logged to {state_file}[/]")
                
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