"""
AsthmAI - Real-Time Air Quality Integration
Goal: Fetch live environmental data to provide real-time asthma risk assessments.
Uses Open-Meteo API (Free, no key required).
"""

import requests
import json
from datetime import datetime
from typing import Dict, Any, Optional

class RealTimeAQI:
    """Fetches real-time air quality and weather data."""
    
    def __init__(self):
        self.aqiu_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        self.weather_url = "https://api.open-meteo.com/v1/forecast"
        
    def get_live_data(self, latitude: float, longitude: float) -> Optional[Dict[str, Any]]:
        """
        Fetch current air quality and weather for a location.
        """
        try:
            # 1. Fetch Air Quality
            aq_params = {
                "latitude": latitude,
                "longitude": longitude,
                "current": ["us_aqi", "pm2_5", "nitrogen_dioxide", "sulphur_dioxide"],
                "timezone": "auto"
            }
            aq_response = requests.get(self.aqiu_url, params=aq_params)
            aq_data = aq_response.json()
            
            # 2. Fetch Weather
            weather_params = {
                "latitude": latitude,
                "longitude": longitude,
                "current": ["temperature_2m", "relative_humidity_2m"],
                "timezone": "auto"
            }
            weather_response = requests.get(self.weather_url, params=weather_params)
            w_data = weather_response.json()
            
            # Combine data
            if "current" in aq_data and "current" in w_data:
                combined_data = {
                    "timestamp": datetime.now().isoformat(),
                    "location": {"lat": latitude, "lon": longitude},
                    "AQI": aq_data["current"]["us_aqi"],
                    "PM2.5": aq_data["current"]["pm2_5"],
                    "NO2 level": aq_data["current"]["nitrogen_dioxide"],
                    "SO2 level": aq_data["current"]["sulphur_dioxide"],
                    "CO2 level": 420.0,  # OpenMeteo doesn't provide CO2, using global baseline
                    "Temperature": w_data["current"]["temperature_2m"],
                    "Humidity": w_data["current"]["relative_humidity_2m"]
                }
                return combined_data
            else:
                print("Error: Incomplete data from API")
                return None
                
        except Exception as e:
            print(f"Error fetching live data: {str(e)}")
            return None

def main():
    # Test with New York coordinates
    aqi_service = RealTimeAQI()
    print("Fetching live data for New York (40.71, -74.00)...")
    data = aqi_service.get_live_data(40.7128, -74.0060)
    
    if data:
        print("\nLive Environmental Data:")
        print(json.dumps(data, indent=2))
        print("\nReady for inference!")
    else:
        print("Failed to fetch data.")

if __name__ == "__main__":
    main()
