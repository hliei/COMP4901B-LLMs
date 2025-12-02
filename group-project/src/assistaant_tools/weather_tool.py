"""
OpenWeatherMap Tool for weather information.

This module provides a wrapper for OpenWeatherMap API to:
- Get current weather conditions
- Get multi-day weather forecasts
- Support for any city worldwide
"""

import os
import requests
from typing import Dict, Any, Optional
from datetime import datetime

# Load environment variables
from src.utils import load_env
load_env()


class WeatherTool:
    """
    OpenWeatherMap API wrapper for weather information.
    
    Features:
    - Current weather conditions (temperature, humidity, wind)
    - Multi-day weather forecasts (up to 5 days on free tier)
    - Global city coverage
    - Metric units (Celsius)
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Weather tool.
        
        Args:
            api_key: OpenWeatherMap API key 
                     (if None, uses OPENWEATHER_API_KEY from environment)
        """
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OpenWeatherMap API key not found. "
                "Set OPENWEATHER_API_KEY environment variable."
            )
        
        self.base_url = "https://api.openweathermap.org/data/2.5"
    
    def get_current_weather(self, city: str) -> Dict[str, Any]:
        """
        Get current weather conditions for a city.
        
        Args:
            city: City name (e.g., "Hong Kong", "London", "New York")
        
        Returns:
            Dict containing:
            - success: bool - Whether operation succeeded
            - city: str - City name (if successful)
            - temperature: float - Temperature in Celsius (if successful)
            - feels_like: float - Feels like temperature (if successful)
            - humidity: int - Humidity percentage (if successful)
            - weather: str - Weather condition (e.g., "Clear", "Clouds")
            - description: str - Detailed description (e.g., "scattered clouds")
            - wind_speed: float - Wind speed in m/s (if successful)
            - timestamp: str - Data timestamp ISO format (if successful)
            - error: str - Error message (if failed)
        
        Example:
            >>> weather = WeatherTool()
            >>> result = weather.get_current_weather("Hong Kong")
            >>> if result['success']:
            ...     print(f"Temperature: {result['temperature']}°C")
            ...     print(f"Conditions: {result['description']}")
        """
        try:
            url = f"{self.base_url}/weather"
            params = {
                'q': city,
                'appid': self.api_key,
                'units': 'metric'  # Celsius
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                "success": True,
                "city": data['name'],
                "temperature": data['main']['temp'],
                "feels_like": data['main']['feels_like'],
                "humidity": data['main']['humidity'],
                "weather": data['weather'][0]['main'],
                "description": data['weather'][0]['description'],
                "wind_speed": data['wind']['speed'],
                "timestamp": datetime.fromtimestamp(data['dt']).isoformat()
            }
        
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"API request failed: {str(e)}"
            }
        except (KeyError, IndexError) as e:
            return {
                "success": False,
                "error": f"Unexpected response format: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_forecast(self, city: str, days: int = 3) -> Dict[str, Any]:
        """
        Get weather forecast for upcoming days.
        
        Args:
            city: City name
            days: Number of days to forecast (1-5, default: 3)
                  Free tier supports up to 5 days
        
        Returns:
            Dict containing:
            - success: bool - Whether operation succeeded
            - city: str - City name (if successful)
            - forecasts: List[Dict] - Daily forecasts (if successful), each with:
                - date: str - Date in ISO format
                - temp_min: float - Minimum temperature
                - temp_max: float - Maximum temperature
                - temp_avg: float - Average temperature
                - weather: str - Predominant weather condition
                - description: str - Weather description
            - error: str - Error message (if failed)
        
        Example:
            >>> weather = WeatherTool()
            >>> result = weather.get_forecast("Hong Kong", days=3)
            >>> if result['success']:
            ...     for forecast in result['forecasts']:
            ...         print(f"{forecast['date']}: {forecast['temp_min']}-{forecast['temp_max']}°C")
            ...         print(f"  Conditions: {forecast['description']}")
        """
        if days < 1 or days > 5:
            return {
                "success": False,
                "error": "Days must be between 1 and 5 (free tier limitation)"
            }
        
        try:
            url = f"{self.base_url}/forecast"
            params = {
                'q': city,
                'appid': self.api_key,
                'units': 'metric',
                'cnt': days * 8  # 8 forecasts per day (3-hour intervals)
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Group forecasts by day
            daily_forecasts = []
            current_date = None
            day_data = []
            
            for item in data['list']:
                date = datetime.fromtimestamp(item['dt']).date()
                
                if current_date != date:
                    # Process previous day's data
                    if day_data:
                        daily_forecasts.append(
                            self._summarize_day(current_date, day_data)
                        )
                    
                    current_date = date
                    day_data = [item]
                else:
                    day_data.append(item)
            
            # Add last day
            if day_data:
                daily_forecasts.append(
                    self._summarize_day(current_date, day_data)
                )
            
            return {
                "success": True,
                "city": data['city']['name'],
                "forecasts": daily_forecasts[:days]
            }
        
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"API request failed: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _summarize_day(self, date, day_data: list) -> Dict[str, Any]:
        """
        Summarize a day's worth of 3-hour interval forecasts.
        
        Args:
            date: Date object
            day_data: List of forecast items for the day
        
        Returns:
            Dictionary with aggregated daily weather data
        """
        temps = [d['main']['temp'] for d in day_data]
        weathers = [d['weather'][0]['main'] for d in day_data]
        
        # Find most common weather condition
        weather_counts = {}
        for w in weathers:
            weather_counts[w] = weather_counts.get(w, 0) + 1
        predominant_weather = max(weather_counts, key=weather_counts.get)
        
        return {
            "date": date.isoformat(),
            "temp_min": round(min(temps), 1),
            "temp_max": round(max(temps), 1),
            "temp_avg": round(sum(temps) / len(temps), 1),
            "weather": predominant_weather,
            "description": day_data[0]['weather'][0]['description']
        }


# Convenience function for quick testing
def test_weather_api(api_key: Optional[str] = None) -> bool:
    """
    Test OpenWeatherMap API connection.
    
    Args:
        api_key: Optional API key (uses env variable if not provided)
    
    Returns:
        bool: True if API key works
    """
    try:
        weather = WeatherTool(api_key)
        result = weather.get_current_weather("London")
        return result.get('success', False)
    except Exception as e:
        print(f"Weather API test failed: {e}")
        return False


if __name__ == "__main__":
    # Quick test
    print("Testing OpenWeatherMap API...")
    if test_weather_api():
        print("✅ Weather API connection successful!")
        
        # Example usage
        weather = WeatherTool()
        
        print("\n--- Current Weather ---")
        result = weather.get_current_weather("Hong Kong")
        if result['success']:
            print(f"City: {result['city']}")
            print(f"Temperature: {result['temperature']}°C")
            print(f"Conditions: {result['description']}")
        
        print("\n--- 3-Day Forecast ---")
        forecast = weather.get_forecast("Hong Kong", days=3)
        if forecast['success']:
            for day in forecast['forecasts']:
                print(f"{day['date']}: {day['temp_min']}-{day['temp_max']}°C, {day['weather']}")
    else:
        print("❌ Weather API test failed. Check your API key.")
