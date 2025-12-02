"""
Google Maps Tool for location services and directions.

This module provides a wrapper for Google Maps API to:
- Get directions between locations
- Search for nearby places
- Geocode addresses to coordinates
"""

import os
import googlemaps
from typing import Dict, Any, List, Optional
from datetime import datetime

# Load environment variables
from src.utils import load_env
load_env()


class GoogleMapsTool:
    """
    Google Maps API wrapper for location and navigation services.
    
    Features:
    - Get directions with multiple travel modes (driving, walking, transit, cycling)
    - Search for nearby places (restaurants, shops, etc.)
    - Geocode addresses to coordinates
    - Calculate travel time and distance
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Google Maps tool.
        
        Args:
            api_key: Google Maps API key
                     (if None, uses GOOGLE_MAPS_API_KEY from environment)
        """
        self.api_key = api_key or os.getenv('GOOGLE_MAPS_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Google Maps API key not found. "
                "Set GOOGLE_MAPS_API_KEY environment variable."
            )
        
        self.client = googlemaps.Client(key=self.api_key)
    
    def get_directions(
        self,
        origin: str,
        destination: str,
        mode: str = "driving"
    ) -> Dict[str, Any]:
        """
        Get directions between two locations.
        
        Args:
            origin: Starting location (address or place name)
            destination: Destination location (address or place name)
            mode: Travel mode - one of:
                  - "driving" (default)
                  - "walking"
                  - "transit" (public transportation)
                  - "bicycling"
        
        Returns:
            Dict containing:
            - success: bool - Whether operation succeeded
            - origin: str - Starting address (if successful)
            - destination: str - Destination address (if successful)
            - distance: str - Total distance (e.g., "5.2 km")
            - duration: str - Estimated duration (e.g., "15 mins")
            - mode: str - Travel mode used
            - steps: List[Dict] - First 5 navigation steps (if successful), each with:
                - instruction: str - Turn-by-turn instruction (HTML)
                - distance: str - Step distance
                - duration: str - Step duration
            - error: str - Error message (if failed)
        
        Example:
            >>> maps = GoogleMapsTool()
            >>> result = maps.get_directions(
            ...     origin="HKUST",
            ...     destination="Hong Kong Convention Centre",
            ...     mode="transit"
            ... )
            >>> if result['success']:
            ...     print(f"Distance: {result['distance']}")
            ...     print(f"Duration: {result['duration']}")
        """
        if mode not in ["driving", "walking", "transit", "bicycling"]:
            return {
                "success": False,
                "error": f"Invalid mode: {mode}. Must be one of: driving, walking, transit, bicycling"
            }
        
        try:
            result = self.client.directions(
                origin,
                destination,
                mode=mode,
                departure_time=datetime.now()
            )
            
            if not result:
                return {
                    "success": False,
                    "error": f"No route found from {origin} to {destination}"
                }
            
            route = result[0]
            leg = route['legs'][0]
            
            # Extract first 5 steps for concise directions
            steps = []
            for step in leg['steps'][:5]:
                steps.append({
                    "instruction": step['html_instructions'],
                    "distance": step['distance']['text'],
                    "duration": step['duration']['text']
                })
            
            return {
                "success": True,
                "origin": leg['start_address'],
                "destination": leg['end_address'],
                "distance": leg['distance']['text'],
                "duration": leg['duration']['text'],
                "mode": mode,
                "steps": steps
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def search_nearby(
        self,
        location: str,
        keyword: str,
        radius: int = 1000
    ) -> Dict[str, Any]:
        """
        Search for places near a location.
        
        Args:
            location: Center location for search (address or place name)
            keyword: Search term (e.g., "restaurant", "coffee", "pharmacy")
            radius: Search radius in meters (default: 1000m = 1km)
        
        Returns:
            Dict containing:
            - success: bool - Whether operation succeeded
            - location: str - Search center location
            - keyword: str - Search keyword used
            - places: List[Dict] - Top 5 places found, each with:
                - name: str - Place name
                - address: str - Place address
                - rating: float or str - User rating (or "N/A")
                - types: List[str] - Place types (up to 3)
            - error: str - Error message (if failed)
        
        Example:
            >>> maps = GoogleMapsTool()
            >>> result = maps.search_nearby(
            ...     location="HKUST",
            ...     keyword="restaurant",
            ...     radius=500
            ... )
            >>> if result['success']:
            ...     for place in result['places']:
            ...         print(f"{place['name']} - Rating: {place['rating']}")
        """
        try:
            # First, geocode the location to get coordinates
            geocode_result = self.client.geocode(location)
            
            if not geocode_result:
                return {
                    "success": False,
                    "error": f"Could not find location: {location}"
                }
            
            lat_lng = geocode_result[0]['geometry']['location']
            
            # Search for nearby places
            places_result = self.client.places_nearby(
                location=(lat_lng['lat'], lat_lng['lng']),
                keyword=keyword,
                radius=radius
            )
            
            # Extract top 5 places
            places = []
            for place in places_result.get('results', [])[:5]:
                places.append({
                    "name": place.get('name', 'Unknown'),
                    "address": place.get('vicinity', 'No address'),
                    "rating": place.get('rating', 'N/A'),
                    "types": place.get('types', [])[:3]  # First 3 types
                })
            
            return {
                "success": True,
                "location": location,
                "keyword": keyword,
                "places": places
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def geocode(self, address: str) -> Dict[str, Any]:
        """
        Convert address to geographic coordinates.
        
        Args:
            address: Address or place name to geocode
        
        Returns:
            Dict containing:
            - success: bool - Whether operation succeeded
            - address: str - Formatted address (if successful)
            - latitude: float - Latitude coordinate (if successful)
            - longitude: float - Longitude coordinate (if successful)
            - error: str - Error message (if failed)
        
        Example:
            >>> maps = GoogleMapsTool()
            >>> result = maps.geocode("HKUST")
            >>> if result['success']:
            ...     print(f"Coordinates: {result['latitude']}, {result['longitude']}")
        """
        try:
            result = self.client.geocode(address)
            
            if not result:
                return {
                    "success": False,
                    "error": f"Could not geocode address: {address}"
                }
            
            location = result[0]['geometry']['location']
            
            return {
                "success": True,
                "address": result[0]['formatted_address'],
                "latitude": location['lat'],
                "longitude": location['lng']
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_travel_time(
        self,
        origin: str,
        destination: str,
        mode: str = "driving"
    ) -> Dict[str, Any]:
        """
        Get estimated travel time between two locations.
        
        This is a convenience method that extracts just the duration
        from get_directions().
        
        Args:
            origin: Starting location
            destination: Destination location
            mode: Travel mode (default: "driving")
        
        Returns:
            Dict with success, duration, and optional error
        """
        directions = self.get_directions(origin, destination, mode)
        
        if directions['success']:
            return {
                "success": True,
                "duration": directions['duration'],
                "distance": directions['distance'],
                "mode": mode
            }
        else:
            return directions


# Convenience function for quick testing
def test_maps_api(api_key: Optional[str] = None) -> bool:
    """
    Test Google Maps API connection.
    
    Args:
        api_key: Optional API key (uses env variable if not provided)
    
    Returns:
        bool: True if API key works
    """
    try:
        maps = GoogleMapsTool(api_key)
        result = maps.geocode("London")
        return result.get('success', False)
    except Exception as e:
        print(f"Maps API test failed: {e}")
        return False


if __name__ == "__main__":
    # Quick test
    print("Testing Google Maps API...")
    if test_maps_api():
        print("✅ Maps API connection successful!")
        
        # Example usage
        maps = GoogleMapsTool()
        
        print("\n--- Geocoding ---")
        result = maps.geocode("HKUST")
        if result['success']:
            print(f"Address: {result['address']}")
            print(f"Coordinates: {result['latitude']}, {result['longitude']}")
        
        print("\n--- Directions ---")
        directions = maps.get_directions(
            origin="HKUST",
            destination="Hong Kong International Airport",
            mode="driving"
        )
        if directions['success']:
            print(f"Distance: {directions['distance']}")
            print(f"Duration: {directions['duration']}")
        
        print("\n--- Nearby Search ---")
        nearby = maps.search_nearby(
            location="HKUST",
            keyword="restaurant",
            radius=1000
        )
        if nearby['success']:
            print(f"Found {len(nearby['places'])} places:")
            for place in nearby['places'][:3]:
                print(f"  - {place['name']} (Rating: {place['rating']})")
    else:
        print("❌ Maps API test failed. Check your API key.")
