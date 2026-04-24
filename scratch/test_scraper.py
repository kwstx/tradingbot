from source_scraper import ResolutionSourceScraper

def test_scraper():
    test_cases = [
        {
            "desc": "This market will resolve to 'Yes' if the temperature at JFK Airport is above 70F.",
            "rules": "Resolution source: https://www.wunderground.com/history/daily/KJFK",
            "expected_id": "KJFK"
        },
        {
            "desc": "Will it rain at Central Park?",
            "rules": "Data from NOAA station KNYC.",
            "expected_id": "KNYC"
        },
        {
            "desc": "Heatwave in Chicago.",
            "rules": "Resolution source: https://www.wunderground.com/dashboard/pws/KILCHICA123",
            "expected_id": "KILCHICA123"
        }
    ]

    for i, tc in enumerate(test_cases):
        res = ResolutionSourceScraper.scrape(tc["desc"], tc["rules"])
        print(f"Test {i+1}: {res['station_id']} (Expected: {tc['expected_id']})")
        if res['coordinates']:
            print(f" -> Coords: {res['coordinates']}")

if __name__ == "__main__":
    test_scraper()
