import re
from typing import Dict, Optional, Tuple

class ResolutionSourceScraper:
    """
    Parses Polymarket contract descriptions and rules to identify specific
    weather stations, NOAA IDs, or Weather Underground links.
    """

    # Common NOAA Station ID pattern (K + 3 letters for US stations)
    NOAA_STATION_RE = re.compile(r'\b(K[A-Z0-9]{3})\b', re.IGNORECASE)
    
    # Weather Underground station pattern (e.g., https://www.wunderground.com/dashboard/pws/KNYNEWYO123)
    WUNDERGROUND_RE = re.compile(r'wunderground\.com/(?:dashboard/pws/|history/daily/)\b([A-Z0-9]{5,})\b', re.IGNORECASE)
    
    # Specific station names that might be mentioned (not just IDs)
    STATION_KEYWORDS = {
        "CENTRAL PARK": "KNYC",
        "JFK AIRPORT": "KJFK",
        "LAGUARDIA": "KLGA",
        "NEWARK": "KEWR",
        "O'HARE": "KORD",
        "MIDWAY": "KMDW",
        "HEATHROW": "EGLL",
    }

    # Mapping common station IDs to coordinates to avoid basis risk
    STATION_COORDS = {
        "KNYC": (40.7833, -73.9667),  # Central Park
        "KJFK": (40.6413, -73.7781),  # JFK Airport
        "KLGA": (40.7769, -73.8740),  # LaGuardia
        "KEWR": (40.6895, -74.1745),  # Newark
        "KORD": (41.9742, -87.9073),  # O'Hare
        "KMDW": (41.7868, -87.7522),  # Midway
        "EGLL": (51.4700, -0.4543),   # Heathrow
        "EGLC": (51.5048, 0.0503),    # London City
        "RJTT": (35.5494, 139.7798), # Haneda (Tokyo)
        "RJAA": (35.7720, 140.3929), # Narita (Tokyo)
    }

    @classmethod
    def scrape(cls, description: str, rules: str) -> Dict[str, Optional[any]]:
        """
        Main entry point for scraping a market's resolution source.
        """
        combined_text = f"{description} {rules}"
        
        # 1. Try to find NOAA ID
        noaa_match = cls.NOAA_STATION_RE.search(combined_text)
        station_id = noaa_match.group(1).upper() if noaa_match else None
        
        # 2. Try to find Weather Underground ID if no NOAA ID found or verify it
        wunder_match = cls.WUNDERGROUND_RE.search(combined_text)
        wunder_id = wunder_match.group(1).upper() if wunder_match else None
        
        # 3. Check for keywords if no ID found
        if not station_id:
            for keyword, mapped_id in cls.STATION_KEYWORDS.items():
                if keyword.upper() in combined_text.upper():
                    station_id = mapped_id
                    break
        
        final_id = station_id or wunder_id
        coords = cls.STATION_COORDS.get(final_id) if final_id else None
        
        return {
            "station_id": final_id,
            "source_type": "NOAA" if station_id else ("Wunderground" if wunder_id else "Generic"),
            "coordinates": coords,
            "is_specific_station": final_id is not None
        }

    @classmethod
    def get_refined_coordinates(cls, city: str, lat: float, lon: float, description: str, rules: str) -> Tuple[float, float, bool]:
        """
        Given a city and default coordinates, returns refined coordinates if a station matches.
        """
        source_data = cls.scrape(description, rules)
        if source_data["coordinates"]:
            return source_data["coordinates"][0], source_data["coordinates"][1], True
        return lat, lon, False
