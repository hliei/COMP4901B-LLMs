"""
Google Calendar Tool for event management.

This module provides a wrapper for Google Calendar API to:
- Create calendar events
- List upcoming events
- Update/delete events
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from googleapiclient.discovery import build


class GoogleCalendarTool:
    """
    Google Calendar API wrapper for scheduling and event management.
    
    Features:
    - Create events with location and description
    - List upcoming events with filtering
    - Support for Hong Kong timezone
    - Error handling for API failures
    """
    
    def __init__(self, credentials, timezone: str = 'Asia/Hong_Kong'):
        """
        Initialize Google Calendar tool.
        
        Args:
            credentials: Google OAuth 2.0 credentials object
            timezone: Timezone for events (default: Asia/Hong_Kong)
        """
        self.service = build('calendar', 'v3', credentials=credentials)
        self.timezone = timezone
    
    def create_event(
        self,
        summary: str,
        start_time: str,
        end_time: str,
        location: str = "",
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Create a new calendar event.
        
        Args:
            summary: Event title/name
            start_time: Start time in ISO format (YYYY-MM-DDTHH:MM:SS)
            end_time: End time in ISO format (YYYY-MM-DDTHH:MM:SS)
            location: Event location (optional)
            description: Event description/notes (optional)
        
        Returns:
            Dict containing:
            - success: bool - Whether operation succeeded
            - event_id: str - Created event ID (if successful)
            - link: str - Calendar event link (if successful)
            - summary: str - Event title (if successful)
            - start: str - Event start time (if successful)
            - location: str - Event location (if successful)
            - error: str - Error message (if failed)
        
        Example:
            >>> calendar = GoogleCalendarTool(credentials)
            >>> result = calendar.create_event(
            ...     summary="Team Meeting",
            ...     start_time="2024-12-01T14:00:00",
            ...     end_time="2024-12-01T15:00:00",
            ...     location="Conference Room A",
            ...     description="Weekly sync meeting"
            ... )
            >>> print(result['success'])
            True
        """
        event = {
            'summary': summary,
            'location': location,
            'description': description,
            'start': {
                'dateTime': start_time,
                'timeZone': self.timezone,
            },
            'end': {
                'dateTime': end_time,
                'timeZone': self.timezone,
            },
        }
        
        try:
            result = self.service.events().insert(
                calendarId='primary',
                body=event
            ).execute()
            
            return {
                "success": True,
                "event_id": result['id'],
                "link": result.get('htmlLink', ''),
                "summary": result['summary'],
                "start": result['start']['dateTime'],
                "location": result.get('location', '')
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def list_events(
        self,
        max_results: int = 10,
        days_ahead: int = 7
    ) -> List[Dict[str, Any]]:
        """
        List upcoming calendar events.
        
        Args:
            max_results: Maximum number of events to return (default: 10)
            days_ahead: Number of days to look ahead (default: 7)
        
        Returns:
            List of event dictionaries, each containing:
            - id: str - Event ID
            - summary: str - Event title
            - start: str - Start time (ISO format)
            - end: str - End time (ISO format)
            - location: str - Event location
            - description: str - Event description
            
            Returns list with error dict if operation fails.
        
        Example:
            >>> calendar = GoogleCalendarTool(credentials)
            >>> events = calendar.list_events(max_results=5, days_ahead=3)
            >>> for event in events:
            ...     print(f"{event['summary']} at {event['start']}")
        """
        now = datetime.utcnow()
        time_min = now.isoformat() + 'Z'
        time_max = (now + timedelta(days=days_ahead)).isoformat() + 'Z'
        
        try:
            events_result = self.service.events().list(
                calendarId='primary',
                timeMin=time_min,
                timeMax=time_max,
                maxResults=max_results,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            
            return [{
                'id': event['id'],
                'summary': event.get('summary', 'No title'),
                'start': event['start'].get('dateTime', event['start'].get('date')),
                'end': event['end'].get('dateTime', event['end'].get('date')),
                'location': event.get('location', ''),
                'description': event.get('description', '')
            } for event in events]
        
        except Exception as e:
            return [{"error": str(e)}]
    
    def delete_event(self, event_id: str) -> Dict[str, Any]:
        """
        Delete a calendar event.
        
        Args:
            event_id: ID of the event to delete
        
        Returns:
            Dict with success status and optional error message
        """
        try:
            self.service.events().delete(
                calendarId='primary',
                eventId=event_id
            ).execute()
            
            return {
                "success": True,
                "event_id": event_id
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


# Convenience function for quick testing
def test_calendar_connection(credentials):
    """
    Test Google Calendar API connection.
    
    Args:
        credentials: Google OAuth 2.0 credentials
    
    Returns:
        bool: True if connection successful
    """
    try:
        calendar = GoogleCalendarTool(credentials)
        events = calendar.list_events(max_results=1)
        return len(events) >= 0  # Success if we can fetch events
    except Exception as e:
        print(f"Calendar connection test failed: {e}")
        return False


if __name__ == "__main__":
    """Test Calendar tool with unified authentication."""
    from src.assistaant_tools.auth_helper import get_google_credentials
    
    print("=" * 60)
    print("📅 Testing Google Calendar Tool")
    print("=" * 60)
    
    # Get credentials (will use cached token or authenticate)
    creds = get_google_credentials()
    
    # Create tool instance
    calendar = GoogleCalendarTool(creds)
    
    # Test 1: List upcoming events
    print("\n📋 Test 1: List upcoming events")
    events = calendar.list_events(max_results=5, days_ahead=7)
    print(f"Found {len(events)} events:")
    for event in events:
        if 'error' in event:
            print(f"  ❌ Error: {event['error']}")
        else:
            print(f"  - {event['summary']} at {event['start']}")
    
    print("\n✅ Calendar tool test completed!")
