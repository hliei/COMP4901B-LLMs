"""
Personal Assistant Agent - Part II Option C

This agent coordinates multiple tools for personal assistance tasks:
- Google Calendar (schedule management)
- Weather API (weather forecasting)
- Google Maps (navigation and location services)

Unlike SearchAgent which focuses on finding information, this agent
performs actions and coordinates real-world APIs.
"""

from typing import Dict, Any, List, Optional
import json
import os
from datetime import datetime

from src.llm_client import DeepSeekClient
from src.assistaant_tools.calendar_tool import GoogleCalendarTool
from src.assistaant_tools.weather_tool import WeatherTool
from src.assistaant_tools.maps_tool import GoogleMapsTool
from src.assistant_tool_definitions import ASSISTANT_TOOLS

# Load environment variables
from src.utils import load_env
load_env()


class PersonalAssistantAgent:
    """
    Personal Assistant Agent that coordinates multiple real-world APIs.
    
    This agent uses DeepSeek's Function Calling capability to:
    1. Understand natural language requests
    2. Break down complex tasks into steps
    3. Call appropriate tools (Calendar, Weather, Maps)
    4. Coordinate multiple tools to complete tasks
    
    Example tasks:
    - "Schedule outdoor lunch if weather is good tomorrow"
    - "Find Italian restaurants near HKUST and get directions"
    - "Check weather before planning my weekend trip"
    """
    
    def __init__(
        self,
        llm_client: Optional[DeepSeekClient] = None,
        google_credentials = None,
        max_steps: int = 15,
        verbose: bool = False
    ):
        """
        Initialize Personal Assistant Agent.
        
        Args:
            llm_client: DeepSeek LLM client (if None, creates new one)
            google_credentials: Google OAuth credentials for Calendar API
            max_steps: Maximum number of tool calls (default: 15)
            verbose: Whether to print detailed progress (default: False)
        """
        self.llm_client = llm_client or DeepSeekClient()
        self.max_steps = max_steps
        self.verbose = verbose
        
        # Initialize tools
        if google_credentials is None:
            raise ValueError(
                "google_credentials required. "
                "Use: from src.assistaant_tools.auth_helper import get_google_credentials"
            )
        
        self.calendar = GoogleCalendarTool(google_credentials)
        self.weather = WeatherTool()  # Auto-loads API key from env
        self.maps = GoogleMapsTool()   # Auto-loads API key from env
        
        # Tool dispatcher - maps function names to handlers
        self.tool_handlers = {
            'create_calendar_event': self._handle_create_event,
            'list_calendar_events': self._handle_list_events,
            'get_current_weather': self._handle_current_weather,
            'get_weather_forecast': self._handle_weather_forecast,
            'get_directions': self._handle_directions,
            'search_nearby_places': self._handle_nearby_search,
            'google_search': self._handle_google_search,  # Additional tool from Part 1
        }
    
    # ==========================================
    # Tool Handler Methods
    # ==========================================
    
    def _handle_create_event(self, **kwargs) -> Dict[str, Any]:
        """Handle create_calendar_event tool call."""
        return self.calendar.create_event(**kwargs)
    
    def _handle_list_events(self, **kwargs) -> Dict[str, Any]:
        """Handle list_calendar_events tool call."""
        events = self.calendar.list_events(**kwargs)
        # Wrap in dict for consistent return format
        return {"success": True, "events": events} if events else {"success": False, "error": "No events found"}
    
    def _handle_current_weather(self, **kwargs) -> Dict[str, Any]:
        """Handle get_current_weather tool call."""
        return self.weather.get_current_weather(**kwargs)
    
    def _handle_weather_forecast(self, **kwargs) -> Dict[str, Any]:
        """Handle get_weather_forecast tool call."""
        return self.weather.get_forecast(**kwargs)
    
    def _handle_directions(self, **kwargs) -> Dict[str, Any]:
        """Handle get_directions tool call."""
        return self.maps.get_directions(**kwargs)
    
    def _handle_nearby_search(self, **kwargs) -> Dict[str, Any]:
        """Handle search_nearby_places tool call."""
        return self.maps.search_nearby(**kwargs)
    
    def _handle_google_search(self, **kwargs) -> Dict[str, Any]:
        """Handle google_search tool call (from Part 1)."""
        from src.search_tools.search_tool import google_search
        results = google_search(**kwargs)
        # Wrap in dict for consistent return format
        return {"success": True, "results": results} if results else {"success": False, "error": "No search results found"}
    
    # ==========================================
    # System Prompt
    # ==========================================
    
    def _create_system_prompt(self) -> str:
        """
        Create system prompt for the agent.
        
        This prompt defines the agent's:
        - Role and capabilities
        - Available tools
        - Task completion criteria
        - Behavior guidelines
        """
        return """You are a helpful personal assistant with access to real-world tools.

**Available Tools:**

1. **Google Calendar** - Manage schedule
   - create_calendar_event: Schedule events with time, location, description
   - list_calendar_events: Check upcoming events and availability

2. **Weather API** - Weather information
   - get_current_weather: Get current conditions for a city
   - get_weather_forecast: Get multi-day forecast (up to 5 days)

3. **Google Maps** - Location and navigation
   - get_directions: Get routes, travel time, and directions
   - search_nearby_places: Find restaurants, shops, services near a location

4. **Google Search** - General information lookup
   - google_search: Search the web for facts, information, or web content

**Your Capabilities:**
- Schedule management based on weather conditions
- Find and recommend locations with directions
- Coordinate multiple tools for complex tasks
- Make intelligent decisions based on weather/location data

**Important Instructions:**

1. **Break down complex requests** into logical steps
   Example: "Schedule outdoor lunch if weather is good"
   → Check weather → Decide if suitable → Check calendar schedule → Create event if suitable and no time conflict

2. **Use weather to inform decisions**
   - Only schedule outdoor activities if weather is suitable
   - Consider rain, temperature, and conditions
   - Suggest alternatives if weather is bad

3. **Check calendar availability BEFORE creating events**
    - ALWAYS use list_calendar_events to check existing events before creating new ones
    - Compare the proposed event time with existing event times to detect conflicts
    - Two events conflict if their time ranges overlap (event A ends after event B starts AND event A starts before event B ends)
    - If conflict detected, inform user and suggest alternative times
    - Only create the event if there are NO conflicts

4. **Coordinate location + calendar**
   - When scheduling at a location, include address in event
   - Get directions when planning travel
   - Find nearby places when location-dependent

5. **Complete task indication**
   When task is fully completed, respond with:
   "TASK COMPLETE: <brief summary of what was accomplished>"

6. **Handle errors gracefully**
   - If a tool fails, try alternative approaches
   - Inform about issues but continue with available data
   - Don't give up unless truly impossible

**Examples:**

Task: "Schedule outdoor lunch if weather is good tomorrow at noon"
Steps:
1. get_weather_forecast(city="Hong Kong", days=1) → Check tomorrow's weather
2. Analyze: Is it suitable? (Not raining, reasonable temperature)
3. If yes: list_calendar_events(days_ahead=1) → Get all events for tomorrow
4. Check for conflicts: Compare proposed time (12:00-13:00) with existing event times
   - If existing event is 11:00-12:30 → CONFLICT! (overlaps with 12:00-13:00)
   - If existing event is 14:00-15:00 → No conflict (no overlap)
5. If no conflicts: create_calendar_event(summary="Outdoor Lunch", start_time="...", end_time="...")
6. "TASK COMPLETE: Scheduled outdoor lunch for tomorrow at noon. Weather will be sunny, 25°C. No calendar conflicts."

Task: "Find Italian restaurants near HKUST and get directions"
Steps:
1. search_nearby_places(location="HKUST", keyword="Italian restaurant")
2. get_directions(origin="HKUST", destination="<best restaurant>", mode="walking")
3. "TASK COMPLETE: Found 5 Italian restaurants. Closest is Paisano's (10min walk, 800m)."

Be proactive, helpful, and coordinate tools effectively!"""
    
    # ==========================================
    # Main Execution Method
    # ==========================================
    
    def execute_task(self, user_request: str) -> Dict[str, Any]:
        """
        Execute a personal assistant task using multiple tools.
        
        This method:
        1. Sends user request to LLM with tool definitions
        2. LLM decides which tools to call
        3. Executes tool calls and returns results to LLM
        4. Continues until task is complete or max_steps reached
        
        Args:
            user_request: Natural language task description
            
        Returns:
            Dict containing:
            - task: Original request
            - steps: List of tool calls with arguments and results
            - final_summary: Task completion summary
            - success: Whether task completed successfully
            - tools_used: List of unique tools used
            
        Example:
            >>> agent = PersonalAssistantAgent(creds)
            >>> result = agent.execute_task(
            ...     "Check weather and schedule bike ride if sunny"
            ... )
            >>> print(result['success'])
            True
        """
        
        # Initialize conversation with system prompt and user request
        messages = [
            {"role": "system", "content": self._create_system_prompt()},
            {"role": "user", "content": user_request}
        ]
        
        # Track execution
        trajectory = {
            "task": user_request,
            "steps": [],
            "final_summary": "",
            "success": False,
            "tools_used": set()  # Will convert to list at end
        }
        
        # Main execution loop
        for step_num in range(1, self.max_steps + 1):
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"Step {step_num}/{self.max_steps}")
                print(f"{'='*70}")
            
            # Get LLM response with tool calling
            try:
                # Use the underlying OpenAI client directly for function calling
                response = self.llm_client.client.chat.completions.create(
                    model=self.llm_client.model,
                    messages=messages,
                    tools=ASSISTANT_TOOLS,
                    tool_choice="auto",
                    temperature=0.0  # Deterministic for reproducibility
                )
            except Exception as e:
                if self.verbose:
                    print(f"❌ LLM Error: {e}")
                trajectory["final_summary"] = f"Error: {str(e)}"
                break
            
            message = response.choices[0].message
            
            # Check if task is complete
            # Accept both explicit "TASK COMPLETE" and finish_reason="stop" without tool calls
            if message.content and "TASK COMPLETE" in message.content:
                trajectory["final_summary"] = message.content
                trajectory["success"] = True
                if self.verbose:
                    print(f"\n✅ {message.content}")
                break
            
            # If LLM responds with text but no tool calls and finish_reason is stop, consider task complete
            if message.content and not message.tool_calls and response.choices[0].finish_reason == "stop":
                trajectory["final_summary"] = f"TASK COMPLETE: {message.content}"
                trajectory["success"] = True
                if self.verbose:
                    print(f"\n✅ Task completed with answer: {message.content}")
                break
            
            # Process tool calls
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    
                    # Parse arguments
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError as e:
                        if self.verbose:
                            print(f"⚠️  JSON parsing error: {e}")
                        arguments = {}
                    
                    if self.verbose:
                        print(f"\n🔧 Tool: {function_name}")
                        print(f"📝 Args: {json.dumps(arguments, indent=2, ensure_ascii=False)}")
                    
                    # Execute tool
                    try:
                        handler = self.tool_handlers.get(function_name)
                        if not handler:
                            result = {
                                "success": False,
                                "error": f"Unknown tool: {function_name}"
                            }
                        else:
                            result = handler(**arguments)
                        
                        # Track tool usage
                        trajectory["tools_used"].add(function_name)
                        
                        # Record step
                        step_record = {
                            "step_number": step_num,
                            "tool": function_name,
                            "arguments": arguments,
                            "result": result,
                            "success": result.get("success", True)
                        }
                        
                        if self.verbose:
                            # Pretty print result
                            result_str = json.dumps(result, indent=2, ensure_ascii=False)
                            # Truncate very long results
                            if len(result_str) > 500:
                                result_str = result_str[:500] + "\n... (truncated)"
                            print(f"✅ Result: {result_str}")
                        
                    except Exception as e:
                        result = {"success": False, "error": str(e)}
                        step_record = {
                            "step_number": step_num,
                            "tool": function_name,
                            "arguments": arguments,
                            "result": result,
                            "success": False
                        }
                        
                        if self.verbose:
                            print(f"❌ Error executing {function_name}: {e}")
                    
                    trajectory["steps"].append(step_record)
                    
                    # Add tool result to conversation
                    # This follows OpenAI's function calling format
                    messages.append({
                        "role": "assistant",
                        "tool_calls": [tool_call.model_dump() if hasattr(tool_call, 'model_dump') else tool_call.dict()]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps(result, ensure_ascii=False)
                    })
            
            # Handle assistant message without tool calls
            elif message.content:
                messages.append({
                    "role": "assistant",
                    "content": message.content
                })
                
                if self.verbose:
                    print(f"\n💭 Assistant: {message.content}")
            
            else:
                # No content and no tool calls - something went wrong
                if self.verbose:
                    print("⚠️  Empty response from LLM")
                break
        
        # Check if max steps reached without completion
        if not trajectory["success"] and step_num >= self.max_steps:
            trajectory["final_summary"] = (
                f"Task incomplete: Reached maximum steps ({self.max_steps}). "
                f"Completed {len(trajectory['steps'])} tool calls."
            )
            if self.verbose:
                print(f"\n⚠️  {trajectory['final_summary']}")
        
        # Convert set to list for JSON serialization
        trajectory["tools_used"] = list(trajectory["tools_used"])
        
        return trajectory


# ==========================================
# Convenience Function
# ==========================================

def create_assistant_agent(
    google_credentials,
    verbose: bool = False
) -> PersonalAssistantAgent:
    """
    Create a PersonalAssistantAgent with default settings.
    
    Args:
        google_credentials: Google OAuth credentials
        verbose: Whether to print detailed progress
    
    Returns:
        Configured PersonalAssistantAgent instance
    
    Example:
        >>> from src.assistaant_tools.auth_helper import get_google_credentials
        >>> creds = get_google_credentials()
        >>> agent = create_assistant_agent(creds, verbose=True)
        >>> result = agent.execute_task("Check weather in Hong Kong")
    """
    return PersonalAssistantAgent(
        google_credentials=google_credentials,
        verbose=verbose
    )


# ==========================================
# Testing
# ==========================================

if __name__ == "__main__":
    """Quick test of the agent."""
    from src.assistaant_tools.auth_helper import get_google_credentials
    
    print("=" * 70)
    print("Personal Assistant Agent - Quick Test")
    print("=" * 70)
    
    # Get credentials
    print("\n🔐 Authenticating...")
    creds = get_google_credentials()
    print("✅ Authenticated")
    
    # Create agent
    print("\n🤖 Creating agent...")
    agent = PersonalAssistantAgent(
        google_credentials=creds,
        verbose=True
    )
    print("✅ Agent ready")
    
    # Test simple task
    print("\n📋 Testing simple task...")
    task = "What's the current weather in Hong Kong?"
    
    result = agent.execute_task(task)
    
    print("\n" + "=" * 70)
    print("📊 Test Results")
    print("=" * 70)
    print(f"Task: {result['task']}")
    print(f"Success: {result['success']}")
    print(f"Steps: {len(result['steps'])}")
    print(f"Tools used: {', '.join(result['tools_used'])}")
    print(f"Summary: {result['final_summary']}")
