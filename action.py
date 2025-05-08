from models import ToolInput, ToolResult
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import threading
import queue
import time
import subprocess
import json
import asyncio
import sys
from rich.console import Console
from rich.panel import Panel

console = Console()

class ActionLayer:
    def __init__(self):
        """Initialize the action layer"""
        self.tools = []
        self._session = None
        self._server_process = None
        self.result_queue = queue.Queue()
        
    def start_mcp_server(self):
        """Start the MCP server"""
        try:
            # Start the server process
            server_params = StdioServerParameters(
                command="python",
                args=["paint_mcp_tools.py"]
            )
            
            # Start the MCP server in a separate thread
            self.server_thread = threading.Thread(
                target=self._run_server_thread,
                args=(server_params,)
            )
            self.server_thread.daemon = True
            self.server_thread.start()
            
            # Wait for server to be ready
            time.sleep(2)  # Increased wait time
            return True
        except Exception as e:
            console.print(f"[bold red]Error starting MCP server: {e}[/]")
            return False
    
    def _run_server_thread(self, server_params):
        """Run the server in a thread by creating and running an event loop"""
        asyncio.run(self._run_server(server_params))
    
    async def _run_server(self, server_params):
        """Run the MCP server asynchronously"""
        try:
            console.print("[cyan]Starting MCP server connection...[/]")
            # This function runs in a separate thread with its own event loop
            async with stdio_client(server_params) as (read, write):
                console.print("[cyan]Established stdio connection with MCP server[/]")
                # Create the session
                async with ClientSession(read, write) as session:
                    self._session = session
                    
                    # Initialize the session
                    console.print("[cyan]Initializing MCP session...[/]")
                    init_result = await session.initialize()
                    console.print(f"[green]MCP session initialized: {init_result}[/]")
                    
                    # Get the tools
                    console.print("[cyan]Requesting available tools...[/]")
                    tools_result = await session.list_tools()
                    self.tools = tools_result.tools
                    console.print(f"[green]Received {len(self.tools)} tools from MCP server[/]")
                    
                    # Signal that initialization is complete
                    self.result_queue.put(("init_complete", self.tools))
                    
                    # Keep the session alive
                    console.print("[cyan]MCP server session ready and waiting for commands[/]")
                    while True:
                        await asyncio.sleep(1)
                        
        except Exception as e:
            console.print(f"[bold red]Error in MCP server thread: {e}[/]")
            import traceback
            traceback.print_exc()
            self.result_queue.put(("init_error", str(e)))
            
    def get_tools(self):
        """Get the available tools"""
        # Wait for tools to be loaded
        if not self.tools:
            try:
                console.print("[cyan]Waiting for tools to be loaded from MCP server...[/]")
                result_type, result_value = self.result_queue.get(timeout=10)
                if result_type == "init_complete":
                    self.tools = result_value
                    console.print(f"[green]Tools loaded successfully: {len(self.tools)} tools available[/]")
                else:
                    console.print(f"[bold red]Error initializing tools: {result_value}[/]")
            except queue.Empty:
                console.print("[bold red]Timeout waiting for tools to load[/]")
        
        return self.tools
            
    def execute_tool(self, tool_call: ToolInput) -> ToolResult:
        """Execute a tool and return the result"""
        try:
            # Special case for show_reasoning to handle it directly
            if tool_call.name == "show_reasoning":
                return self._handle_show_reasoning(tool_call)
                
            # Create a new event loop for each tool execution
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Use wait_for with timeout to prevent hanging
            tool_name = tool_call.name
            console.print(f"[cyan]Starting execution of tool: [bold]{tool_name}[/][/]")
            
            try:
                result = loop.run_until_complete(
                    asyncio.wait_for(
                        self._execute_tool_async(tool_call), 
                        timeout=30  # 30 second timeout
                    )
                )
                loop.close()
                console.print(f"[green]Tool [bold]{tool_name}[/] completed successfully[/]")
                return result
            except asyncio.TimeoutError:
                console.print(f"[bold yellow]Tool [bold]{tool_name}[/] execution timed out after 30 seconds[/]")
                # Generic timeout handler for all tools
                return ToolResult(
                    success=False,
                    content=f"Tool {tool_name} execution timed out but the action may have completed",
                    error="No response received from tool within timeout period"
                )
            
        except Exception as e:
            console.print(f"[bold red]Error executing tool: {e}[/]")
            import traceback
            traceback.print_exc()
            return ToolResult(
                success=False,
                content="",
                error=f"Error executing tool: {str(e)}"
            )
    
    def _handle_show_reasoning(self, tool_call: ToolInput) -> ToolResult:
        """Special handler for show_reasoning tool"""
        try:
            steps = tool_call.args.get("steps", [])
            
            # Only display steps once in the main console
            console.print("[bold cyan]Plan:[/]")
            for idx, step in enumerate(steps, start=1):
                console.print(Panel(
                    step,
                    title=f"Step {idx}",
                    border_style="cyan"
                ))
            
            # Return the result without rendering it again
            return ToolResult(
                success=True,
                content="Reasoning steps displayed successfully."
            )
        except Exception as e:
            console.print(f"[bold red]Error in show_reasoning: {e}[/]")
            return ToolResult(
                success=False,
                content="",
                error=str(e)
            )
    
    async def _execute_tool_async(self, tool_call: ToolInput) -> ToolResult:
        """Execute a tool asynchronously"""
        func_name = tool_call.name
        arguments = tool_call.args
        
        try:
            # Find the tool
            tool = next((t for t in self.tools if t.name == func_name), None)
            if not tool:
                console.print(f"[bold red]Unknown tool: {func_name}[/]")
                return ToolResult(
                    success=False,
                    content="",
                    error=f"Unknown tool: {func_name}"
                )
            
            console.print(f"[cyan]Calling tool [bold]{func_name}[/] with args: {arguments}[/]")
            
            # Process arguments according to schema
            schema_properties = tool.inputSchema.get('properties', {})
            processed_args = {}
            
            for param_name, param_info in schema_properties.items():
                if param_name in arguments:
                    expected_type = param_info.get('type', 'string')
                    value = arguments[param_name]
                    
                    console.print(f"[dim cyan]Processing argument {param_name} with expected type {expected_type}[/]")
                    
                    if expected_type == 'integer':
                        processed_args[param_name] = int(value)
                    elif expected_type == 'number':
                        processed_args[param_name] = float(value)
                    elif expected_type == 'array' and isinstance(value, str):
                        processed_args[param_name] = [
                            int(x.strip()) for x in value.strip('[]').split(',')
                        ]
                    else:
                        processed_args[param_name] = str(value)
            
            console.print(f"[cyan]Processed arguments: {processed_args}[/]")
            console.print(f"[cyan]Sending request to MCP server for tool: [bold]{func_name}[/][/]")
            
            # Detailed logging around the critical MCP server call
            try:
                console.print(f"[dim cyan]Starting MCP tool call: {func_name}[/]")
                result = await self._session.call_tool(func_name, arguments=processed_args)
                console.print(f"[green]MCP server responded for tool: [bold]{func_name}[/][/]")
                console.print(f"[dim]Raw MCP response: {result}[/]")
                console.print(f"[dim]Response type: {type(result)}[/]")
                
                # More detailed inspection of what we got back
                if hasattr(result, '__dict__'):
                    console.print(f"[dim]Response attributes: {result.__dict__}[/]")
            except Exception as call_error:
                console.print(f"[bold red]Error during MCP call: {call_error}[/]")
                import traceback
                traceback.print_exc()
                raise call_error
            
            # Process the result
            if hasattr(result, 'content'):
                console.print(f"[dim]Result has content attribute of type: {type(result.content)}[/]")
                if isinstance(result.content, list):
                    content_str = " ".join([
                        item.text if hasattr(item, 'text') else str(item)
                        for item in result.content
                    ])
                else:
                    content_str = str(result.content)
            else:
                console.print(f"[dim]Result has no content attribute, using string representation[/]")
                content_str = str(result)
                
            console.print(f"[green]Processed result for tool [bold]{func_name}[/]: {content_str[:100]}...[/]")
            
            return ToolResult(
                success=True,
                content=content_str
            )
            
        except Exception as e:
            console.print(f"[bold red]Error in _execute_tool_async for {func_name}: {e}[/]")
            import traceback
            traceback.print_exc()
            return ToolResult(
                success=False,
                content="",
                error=str(e)
            )
            
    def stop(self):
        """Stop the action layer and clean up resources"""
        # Just letting the thread terminate naturally is enough
        # since async context managers will be closed properly
        pass 