"""
Tool definitions for Personal Assistant Agent (Option C).

These definitions follow OpenAI Function Calling format and are compatible
with DeepSeek API. Each tool specifies:
- name: Function identifier
- description: What the tool does (helps LLM decide when to use it)
- parameters: JSON Schema for function arguments

Tools included:
- Google Calendar (create/list events)
- Weather API (current weather, forecasts)
- Google Maps (directions, nearby search)
"""

# Tool definitions for DeepSeek Function Calling
ASSISTANT_TOOLS = [
    # ==========================================
    # Google Calendar Tools
    # ==========================================
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": (
                "Create a new event in Google Calendar. Use this to schedule "
                "meetings, appointments, reminders, or any time-based activities. "
                "Supports location and description fields for additional context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Event title/name (e.g., 'Team Meeting', 'Doctor Appointment')"
                    },
                    "start_time": {
                        "type": "string",
                        "description": (
                            "Start time in ISO 8601 format: YYYY-MM-DDTHH:MM:SS "
                            "(e.g., '2024-12-01T14:00:00' for Dec 1, 2024 at 2:00 PM)"
                        )
                    },
                    "end_time": {
                        "type": "string",
                        "description": (
                            "End time in ISO 8601 format: YYYY-MM-DDTHH:MM:SS "
                            "(e.g., '2024-12-01T15:00:00' for Dec 1, 2024 at 3:00 PM)"
                        )
                    },
                    "location": {
                        "type": "string",
                        "description": "Event location - can be address, place name, or online meeting link"
                    },
                    "description": {
                        "type": "string",
                        "description": "Additional event details, notes, or agenda"
                    }
                },
                "required": ["summary", "start_time", "end_time"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_calendar_events",
            "description": (
                "List upcoming events from Google Calendar. Use this to check "
                "schedule availability, find existing appointments, or review "
                "planned activities before scheduling new events."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of events to return (default: 10)",
                        "default": 10
                    },
                    "days_ahead": {
                        "type": "integer",
                        "description": "Number of days to look ahead from today (default: 7)",
                        "default": 7
                    }
                },
                "required": []
            }
        }
    },
    
    # ==========================================
    # Weather Tools (OpenWeatherMap)
    # ==========================================
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": (
                "Get current weather conditions for a city. Returns temperature, "
                "humidity, weather description, and wind speed. Use this to make "
                "decisions about outdoor activities or what to wear."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": (
                            "City name in English (e.g., 'Hong Kong', 'London', 'New York', 'Tokyo'). "
                            "Can include country for disambiguation (e.g., 'Paris, France')"
                        )
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_forecast",
            "description": (
                "Get weather forecast for upcoming days (up to 5 days). Returns "
                "daily min/max temperatures and weather conditions. Use this for "
                "planning future outdoor activities or travel."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name in English (e.g., 'Hong Kong', 'Tokyo')"
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days to forecast (1-5, default: 3)",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 5
                    }
                },
                "required": ["city"]
            }
        }
    },
    
    # ==========================================
    # Google Maps Tools
    # ==========================================
    {
        "type": "function",
        "function": {
            "name": "get_directions",
            "description": (
                "Get directions and travel time between two locations. Returns "
                "distance, duration, and step-by-step directions. Use this to plan "
                "routes, estimate travel time, or provide navigation instructions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": (
                            "Starting location - can be address, place name, or landmark "
                            "(e.g., 'HKUST', '123 Main St', 'Hong Kong Airport')"
                        )
                    },
                    "destination": {
                        "type": "string",
                        "description": (
                            "Destination location - can be address, place name, or landmark "
                            "(e.g., 'Tsim Sha Tsui', 'Central Station')"
                        )
                    },
                    "mode": {
                        "type": "string",
                        "description": "Travel mode for directions",
                        "enum": ["driving", "walking", "transit", "bicycling"],
                        "default": "driving"
                    }
                },
                "required": ["origin", "destination"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_nearby_places",
            "description": (
                "Search for places (restaurants, shops, services, etc.) near a location. "
                "Returns top 5 results with names, addresses, and ratings. Use this to "
                "find dining options, services, or points of interest near a given location."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": (
                            "Center location for search - can be address, place name, or landmark "
                            "(e.g., 'HKUST', 'Causeway Bay', 'Central MTR Station')"
                        )
                    },
                    "keyword": {
                        "type": "string",
                        "description": (
                            "Search keyword for type of place "
                            "(e.g., 'restaurant', 'coffee', 'pharmacy', 'Italian restaurant', 'bike rental')"
                        )
                    },
                    "radius": {
                        "type": "integer",
                        "description": "Search radius in meters (default: 1000m = 1km, max: 50000m = 50km)",
                        "default": 1000,
                        "minimum": 100,
                        "maximum": 50000
                    }
                },
                "required": ["location", "keyword"]
            }
        }
    },
    
    # ==========================================
    # Google Search Tool (from Part 1)
    # ==========================================
    {
        "type": "function",
        "function": {
            "name": "google_search",
            "description": (
                "Search Google for general information, facts, or web content. "
                "Use this when you need to look up information that's not available "
                "through other specialized tools (weather, maps, calendar). "
                "Returns search results with titles, links, and snippets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search query - be specific and clear "
                            "(e.g., 'Hong Kong weather next week', 'best restaurants in Tsim Sha Tsui')"
                        )
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of search results to return (default: 3, max: 10)",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": ["query"]
            }
        }
    }
]


# Helper function to get tool names (useful for debugging)
def get_tool_names():
    """Return list of all available tool names."""
    return [tool["function"]["name"] for tool in ASSISTANT_TOOLS]


# Tool categories for documentation
TOOL_CATEGORIES = {
    "calendar": ["create_calendar_event", "list_calendar_events"],
    "weather": ["get_current_weather", "get_weather_forecast"],
    "maps": ["get_directions", "search_nearby_places"],
    "search": ["google_search"]  # Additional tool from Part 1
}


if __name__ == "__main__":
    """Print tool definitions for verification."""
    import json
    
    print("=" * 70)
    print("Personal Assistant Tool Definitions")
    print("=" * 70)
    
    print(f"\n📊 Total tools: {len(ASSISTANT_TOOLS)}")
    print(f"📋 Tool names: {', '.join(get_tool_names())}")
    
    print("\n" + "=" * 70)
    print("Tool Categories:")
    print("=" * 70)
    for category, tools in TOOL_CATEGORIES.items():
        print(f"\n{category.upper()}:")
        for tool in tools:
            print(f"  - {tool}")
    
    print("\n" + "=" * 70)
    print("Full Tool Definitions (JSON):")
    print("=" * 70)
    print(json.dumps(ASSISTANT_TOOLS, indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 70)
    print("✅ Tool definitions ready for DeepSeek Function Calling!")
    print("=" * 70)
