from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

def setup_driver():
    chrome_options = Options()
    # chrome_options.add_argument('--headless')  # Comment out headless mode for debugging
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    # Modify navigator.webdriver flag to prevent detection
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    return driver

def search_google(query):
    driver = setup_driver()
    try:
        url = f"https://www.google.com/search?q={query}"
        driver.get(url)
        time.sleep(2)
        return driver.page_source
    finally:
        driver.quit()

def open_link(url, timeout=30):
    driver = setup_driver()
    try:
        driver.get(url)
        # Wait for the page to load completely, use the timeout parameter
        # Add a maximum wait time for the page to load
        driver.set_page_load_timeout(timeout)
        
        # Wait for page to be responsive
        WebDriverWait(driver, timeout).until(
            lambda d: d.execute_script('return document.readyState') == 'complete'
        )
        
        # Allow some extra time for JavaScript to run
        time.sleep(min(5, timeout/2))
        
        # Try to find and interact with the rate calculator if present
        try:
            # Look for the rate calculator form
            purchase_price = driver.find_element(By.CSS_SELECTOR, "input[name='purchasePrice']")
            down_payment = driver.find_element(By.CSS_SELECTOR, "input[name='downPayment']")
            zip_code = driver.find_element(By.CSS_SELECTOR, "input[name='zipCode']")
            
            # Fill in some default values
            purchase_price.send_keys("200000")
            down_payment.send_keys("40000")
            zip_code.send_keys("10001")
            
            # Find and click the update button
            update_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
            update_button.click()
            
            # Wait for rates to update
            time.sleep(5)
        except Exception as e:
            print(f"Could not interact with rate calculator: {e}")
        
        # Get the final page source
        return driver.page_source
    except Exception as e:
        print(f"Error opening link: {e}")
        return None
    finally:
        driver.quit()

def scrape_text(html, selector):
    if not html:
        return []
    soup = BeautifulSoup(html, 'html.parser')
    elements = soup.select(selector)
    return [elem.get_text(strip=True) for elem in elements] 