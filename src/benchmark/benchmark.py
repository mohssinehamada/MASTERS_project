import time
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

# Import agent components using new structure
from src.agent.agent import extract_urls, extract_rate_from_text
from src.tools.opendeepsearch_tools import search_with_opendeepsearch
from src.tools.browser_tools import open_link
from src.utils.helpers import format_duration

class AgentBenchmark:
    def __init__(self):
        self.results = defaultdict(dict)
        self.total_queries = 0
        self.successful_queries = 0
        self.total_websites = 0
        self.successful_websites = 0
        self.total_rates_found = 0
        self.total_execution_time = 0
        
    def run_benchmark(self, test_queries, max_websites_per_query=3, timeout=60):
        """Run benchmark tests on the agent with various queries"""
        print(f"Starting benchmark with {len(test_queries)} test queries")
        print("-" * 50)
        
        for i, query in enumerate(test_queries):
            print(f"\nTest {i+1}/{len(test_queries)}: '{query}'")
            self.total_queries += 1
            
            try:
                # Measure query execution time
                start_time = time.time()
                
                # Step 1: Search for information
                print(f"Searching for: {query}")
                search_results = search_with_opendeepsearch(query)
                
                if not search_results:
                    print("❌ No search results found")
                    self.results[query]['search_success'] = False
                    continue
                
                self.results[query]['search_success'] = True
                
                # Step 2: Extract URLs from search results
                website_query = f"List 5 direct URLs to check current mortgage interest rates in California. Only provide the URLs, one per line."
                website_results = search_with_opendeepsearch(website_query)
                
                if not website_results:
                    print("❌ No website suggestions found")
                    self.results[query]['website_extraction'] = False
                    continue
                
                websites_text = website_results[0]['content']
                print(f"Suggested websites:\n{websites_text[:200]}...")
                
                urls = extract_urls(websites_text)
                
                # Use fallback if no URLs found
                if not urls:
                    print("⚠️ Using fallback websites")
                    urls = [
                        "https://www.bankrate.com/mortgages/mortgage-rates/california/",
                        "https://www.nerdwallet.com/mortgages/mortgage-rates/california",
                        "https://www.calhfa.ca.gov/homeownership/rates/"
                    ]
                
                self.results[query]['urls_found'] = len(urls)
                self.total_websites += len(urls[:max_websites_per_query])
                
                # Step 3: Visit websites and extract rates
                rates_found = []
                successful_sites = 0
                
                for url in urls[:max_websites_per_query]:
                    site_start_time = time.time()
                    print(f"\nVisiting: {url}")
                    
                    try:
                        # Set a timeout for each website
                        html_content = open_link(url, timeout=timeout)
                        
                        if not html_content:
                            print(f"❌ Failed to load {url}")
                            continue
                        
                        successful_sites += 1
                        self.successful_websites += 1
                        
                        # Extract simple test data from the content
                        sample_text = html_content[:5000]
                        
                        # Look for rate patterns
                        import re
                        rate_patterns = [
                            r'(\d+\.\d+)%',  # Simple percentage
                            r'(\d+\.\d+)\s*percent',  # Percentage written out
                            r'(rate|interest|apr).{0,20}(\d+\.\d+)%'  # Rate keywords
                        ]
                        
                        for pattern in rate_patterns:
                            matches = re.findall(pattern, sample_text, re.IGNORECASE)
                            if matches:
                                for match in matches:
                                    if isinstance(match, tuple):
                                        # If the pattern captured multiple groups
                                        rate_value = match[-1]  # Last group is usually the number
                                    else:
                                        rate_value = match
                                    
                                    # Try to extract structured data
                                    rate_data = extract_rate_from_text(
                                        sample_text[max(0, sample_text.find(rate_value)-50):
                                                  min(len(sample_text), sample_text.find(rate_value)+50)],
                                        url
                                    )
                                    
                                    if rate_data:
                                        rates_found.append(rate_data)
                                        print(f"✅ Found rate: {rate_data.get('interest_rate')} ({rate_data.get('type', 'Unknown')})")
                        
                        site_time = time.time() - site_start_time
                        print(f"Site processing time: {site_time:.2f}s")
                        
                    except Exception as e:
                        print(f"❌ Error processing {url}: {str(e)}")
                
                # Record statistics
                self.results[query]['websites_processed'] = successful_sites
                self.results[query]['rates_found'] = len(rates_found)
                self.total_rates_found += len(rates_found)
                
                query_time = time.time() - start_time
                self.total_execution_time += query_time
                self.results[query]['execution_time'] = query_time
                
                print(f"\nQuery execution time: {query_time:.2f}s")
                print(f"Rates found: {len(rates_found)}")
                
                if successful_sites > 0:
                    self.successful_queries += 1
                
                # Output summary for this query
                print("\nRate Summary:")
                for rate in rates_found[:5]:  # Show top 5 rates
                    print(f"- {rate.get('type', 'Unknown')}: {rate.get('interest_rate')} (source: {rate['source'].split('//')[1].split('/')[0] if '://' in rate['source'] else rate['source']})")
                
            except Exception as e:
                print(f"❌ Benchmark error for query '{query}': {str(e)}")
                self.results[query]['error'] = str(e)
        
        # Calculate overall benchmark results
        self.generate_report()
    
    def generate_report(self):
        """Generate a comprehensive benchmark report"""
        print("\n" + "="*50)
        print("BENCHMARK SUMMARY")
        print("="*50)
        
        # Calculate success rates
        query_success_rate = (self.successful_queries / self.total_queries) * 100 if self.total_queries > 0 else 0
        website_success_rate = (self.successful_websites / self.total_websites) * 100 if self.total_websites > 0 else 0
        avg_rates_per_query = self.total_rates_found / self.total_queries if self.total_queries > 0 else 0
        avg_execution_time = self.total_execution_time / self.total_queries if self.total_queries > 0 else 0
        
        print(f"Queries processed: {self.total_queries}")
        print(f"Query success rate: {query_success_rate:.1f}%")
        print(f"Website success rate: {website_success_rate:.1f}%")
        print(f"Average rates found per query: {avg_rates_per_query:.1f}")
        print(f"Average execution time: {format_duration(avg_execution_time)}")
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"data/benchmark_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'summary': {
                    'total_queries': self.total_queries,
                    'successful_queries': self.successful_queries,
                    'query_success_rate': query_success_rate,
                    'total_websites': self.total_websites,
                    'successful_websites': self.successful_websites,
                    'website_success_rate': website_success_rate,
                    'total_rates_found': self.total_rates_found,
                    'avg_rates_per_query': avg_rates_per_query,
                    'total_execution_time': self.total_execution_time,
                    'avg_execution_time': avg_execution_time
                },
                'results': self.results
            }, f, indent=2)
            
        print(f"\nDetailed report saved to: {report_file}")

def main():
    benchmark = AgentBenchmark()
    
    # Define test queries
    test_queries = [
        "What are the current mortgage interest rates in California?",
        "Find me the best interest rates for home loans in California",
        "What is the average 30-year fixed mortgage rate in California?",
        "Compare interest rates from different banks in California",
        "What are today's refinance rates in California?"
    ]
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run agent benchmark tests')
    parser.add_argument('--queries', type=int, default=2, help='Number of test queries to run (max 5)')
    parser.add_argument('--websites', type=int, default=2, help='Maximum websites to check per query')
    parser.add_argument('--timeout', type=int, default=30, help='Timeout in seconds for loading each website')
    args = parser.parse_args()
    
    # Run benchmark with specified number of queries
    num_queries = min(args.queries, len(test_queries))
    benchmark.run_benchmark(
        test_queries[:num_queries],
        max_websites_per_query=args.websites,
        timeout=args.timeout
    )

if __name__ == "__main__":
    main() 