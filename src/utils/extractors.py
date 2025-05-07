from bs4 import BeautifulSoup
import re

def extract_interest_rate(html):
    soup = BeautifulSoup(html, 'html.parser')
    
    # Look for interest rate patterns in Bank of America format
    interest_rate_patterns = [
        r'(\d+\.\d+)%\s*APR',   # Matches "X.XXX% APR"
        r'(\d+\.\d+)%\s*Interest',  # Matches "X.XXX% Interest"
        r'(\d+\.\d+)%\s*Fixed',  # Matches "X.XXX% Fixed"
        r'Rate\s*(\d+\.\d+)%',  # Matches "Rate X.XXX%"
        r'(\d+\.\d+)%'  # Generic rate match
    ]
    
    # First try to find the rates in the rate calculator results
    rate_results = soup.select('.rate-results, .mortgage-rates, .rate-calculator-results')
    if rate_results:
        for result in rate_results:
            text = result.get_text()
            for pattern in interest_rate_patterns:
                match = re.search(pattern, text)
                if match:
                    return {
                        'interest_rate': float(match.group(1)),
                        'source': text.strip(),
                        'context': 'Rate calculator results',
                        'loan_type': 'Calculated rate'
                    }
    
    # Try to find rates in tables
    rate_tables = soup.select('table.rates, .rate-table, [class*="rate"]')
    if rate_tables:
        for table in rate_tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                row_text = ' '.join([cell.get_text(strip=True) for cell in cells])
                
                # Look for rate patterns in the row
                for pattern in interest_rate_patterns:
                    match = re.search(pattern, row_text)
                    if match:
                        return {
                            'interest_rate': float(match.group(1)),
                            'source': row_text.strip(),
                            'context': 'Rate table',
                            'loan_type': cells[0].get_text(strip=True) if cells else 'Unknown'
                        }
    
    # If no rate found in tables, search in other elements
    potential_elements = soup.find_all(['p', 'div', 'span', 'td', 'h1', 'h2', 'h3', 'li'])
    for element in potential_elements:
        text = element.get_text()
        for pattern in interest_rate_patterns:
            match = re.search(pattern, text)
            if match:
                return {
                    'interest_rate': float(match.group(1)),
                    'source': text.strip(),
                    'element': element.name,
                    'context': text[:100] + '...' if len(text) > 100 else text
                }
    
    # If no rate found, return debug info
    return {
        'interest_rate': None,
        'source': 'Not found',
        'debug_info': {
            'page_title': soup.title.string if soup.title else 'No title',
            'rate_tables_found': len(rate_tables),
            'rate_results_found': len(rate_results),
            'sample_content': soup.find('div', class_='rates-content').get_text() if soup.find('div', class_='rates-content') else 'No rates content found'
        }
    }

def extract_estate_listings(html):
    soup = BeautifulSoup(html, 'html.parser')
    listings = []
    
    # Common selectors for property listings
    listing_elements = soup.select('.property-listing, .listing, .offer, .item')
    
    for element in listing_elements:
        listing = {
            'price': element.select_one('.price, .cost, [class*="price"]'),
            'location': element.select_one('.location, .address, [class*="location"]'),
            'size': element.select_one('.size, .area, [class*="size"]'),
            'rooms': element.select_one('.rooms, .bedrooms, [class*="room"]')
        }
        
        # Clean up the data
        cleaned_listing = {}
        for key, value in listing.items():
            if value:
                cleaned_listing[key] = value.get_text(strip=True)
            else:
                cleaned_listing[key] = None
        
        listings.append(cleaned_listing)
    
    return listings 