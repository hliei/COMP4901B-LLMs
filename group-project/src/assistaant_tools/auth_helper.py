"""
Google OAuth 2.0 authentication helper for Personal Assistant Agent.

This module provides unified authentication for:
- Google Calendar API (read/write)

All scopes are combined into one token.json to avoid re-authentication.
"""

import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow


# Combined scopes for all Google APIs we use
SCOPES = [
    # Calendar - need full access to create/edit events
    'https://www.googleapis.com/auth/calendar'
]


def get_google_credentials(
    credentials_file: str = "credentials.json",
    token_file: str = "token.json"
) -> Credentials:
    """
    Get or create Google OAuth 2.0 credentials with all required scopes.
    
    This function handles the OAuth flow:
    1. Check if token.json exists with valid credentials
    2. If expired, try to refresh
    3. If no valid token, run OAuth flow (opens browser)
    4. Save token for future use
    
    Args:
        credentials_file: Path to credentials.json from Google Cloud Console
        token_file: Path to store/load token.json (cached credentials)
    
    Returns:
        Credentials object that can be used with Google API clients
    
    Raises:
        FileNotFoundError: If credentials.json doesn't exist
    
    Note:
        If you change SCOPES, delete token.json to force re-authentication.
    
    Example:
        >>> creds = get_google_credentials()
        >>> calendar_service = build('calendar', 'v3', credentials=creds)
    """
    creds = None
    
    # Check if we have a saved token
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    
    # If no valid credentials, get new ones
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # Try to refresh expired token
            print("🔄 Refreshing expired token...")
            creds.refresh(Request())
        else:
            # Run OAuth flow (will open browser)
            if not os.path.exists(credentials_file):
                raise FileNotFoundError(
                    f"❌ {credentials_file} not found!\n"
                    f"Please download it from Google Cloud Console:\n"
                    f"1. Go to https://console.cloud.google.com/\n"
                    f"2. Navigate to APIs & Services > Credentials\n"
                    f"3. Download OAuth 2.0 Client ID credentials\n"
                    f"4. Save as {credentials_file}"
                )
            
            print("🔐 Starting OAuth authentication...")
            print(f"📋 Requesting scopes: {', '.join(SCOPES)}")
            print("🌐 Browser will open for authorization...")
            
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_file, 
                SCOPES
            )
            creds = flow.run_local_server(port=0)
            
            print("✅ Authentication successful!")
        
        # Save credentials for next time
        with open(token_file, 'w') as token:
            token.write(creds.to_json())
        print(f"💾 Token saved to {token_file}")
    
    return creds


def test_authentication():
    """
    Test authentication and verify all API access.
    
    This will:
    1. Authenticate (or use cached token)
    2. Test Calendar API access
    """
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    
    print("=" * 60)
    print("🧪 Testing Google API Authentication")
    print("=" * 60)
    
    try:
        # Get credentials
        creds = get_google_credentials()
        print("\n✅ Credentials obtained")
        
        # Test Calendar API
        print("\n📅 Testing Calendar API access...")
        calendar_service = build('calendar', 'v3', credentials=creds)
        
        # Try to list calendars (lightweight operation)
        calendar_list = calendar_service.calendarList().list().execute()
        print(f"   ✅ Calendar API working - found {len(calendar_list.get('items', []))} calendars")
        
        print("\n" + "=" * 60)
        print("✅ All API tests passed!")
        print("=" * 60)
        print("\n💡 You can now use these tools:")
        print("   - GoogleCalendarTool")
        print("   - WeatherTool (API key-based, no OAuth)")
        print("   - GoogleMapsTool (API key-based, no OAuth)")
        
        return True
        
    except HttpError as e:
        print(f"\n❌ API Error: {e}")
        print("\n💡 Possible issues:")
        print("   1. APIs not enabled in Google Cloud Console")
        print("   2. Invalid credentials")
        print("   3. Token expired (delete token.json and retry)")
        return False
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


if __name__ == "__main__":
    # Run authentication test
    test_authentication()
