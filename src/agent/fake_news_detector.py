from google.generativeai import GenerativeModel
import json
import time
from dotenv import load_dotenv
import os
import re
from bs4 import BeautifulSoup
import random
from datetime import datetime
import signal
import html2text
import sys

# Add the project root to the Python path if running as a script
if __name__ == "__main__":
    import os.path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.insert(0, project_root)

# Import from our project structure
from src.tools.browser_tools import open_link, scrape_text
from src.tools.opendeepsearch_tools import search_with_opendeepsearch
from src.utils.helpers import save_results
import google.generativeai as genai
from src.agent.model_integration import get_model_integration

# Load environment variables
load_dotenv()

# Initialize Gemini client for analysis
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)
    client = GenerativeModel(model_name="gemini-1.5-pro")
else:
    print("Warning: No Google API key found. Text analysis capabilities will be limited.")
    client = None

# HTML to text converter
text_maker = html2text.HTML2Text()
text_maker.ignore_links = False
text_maker.ignore_images = True

class FakeNewsDetector:
    def __init__(self, use_mock=False):
        self.use_mock = use_mock
        self.timeout_handler = TimeoutHandler(max_runtime_seconds=300)
        self.max_sources = 5
        self.search_results = []
        self.article_texts = []
        self.article_facts = []
        self.claim = ""
        self.analysis_result = {}
    
    def detect(self, claim):
        """Main method to detect if a news claim is fake or real"""
        self.claim = claim
        print(f"Analyzing claim: {claim}")
        
        try:
            # Step 1: Search for information about the claim
            if not self._search_for_information():
                return self._generate_fallback_result("Could not find enough information about this claim.")
            
            # Step 2: Retrieve article content from sources
            if not self._retrieve_article_content():
                return self._generate_fallback_result("Could not retrieve enough content from sources.")
            
            # Step 3: Extract facts from articles
            if not self._extract_facts():
                return self._generate_fallback_result("Could not extract enough facts from the articles.")
            
            # Step 4: Analyze facts and determine veracity
            result = self._analyze_facts()
            
            # Save and return results
            results_path = save_results(result, "fact_check")
            print(f"\nAnalysis completed. Results saved to: {results_path}")
            
            return result
            
        except Exception as e:
            print(f"Error in fake news detection: {e}")
            return self._generate_fallback_result(f"Error during analysis: {str(e)}")
    
    def _search_for_information(self):
        """Search for information about the claim"""
        print("\nSearching for information about this claim...")
        
        # Use mock mode if needed
        if self.use_mock:
            self.search_results = self._generate_mock_search_results()
            print(f"Found {len(self.search_results)} mock sources to analyze")
            return len(self.search_results) > 0
        
        # Create search queries
        search_queries = [
            f"fact check {self.claim}",
            f"{self.claim} true or false",
            f"{self.claim} debunked OR confirmed"
        ]
        
        all_results = []
        
        # Execute searches
        for query in search_queries:
            print(f"Searching: {query}")
            results = search_with_opendeepsearch(query)
            
            if results:
                # Extract URLs
                urls = extract_urls(results[0]['content'])
                for url in urls[:3]:  # Take top 3 URLs from each search
                    # Skip social media and avoid duplicates
                    if not any(r['url'] == url for r in all_results) and not self._is_social_media(url):
                        domain = self._extract_domain(url)
                        all_results.append({
                            'url': url,
                            'domain': domain,
                            'query': query
                        })
            
            # Check timeout
            if self.timeout_handler.check_timeout():
                break
        
        # Take the top results, prioritizing fact-checking sites
        self.search_results = self._prioritize_results(all_results)
        print(f"Found {len(self.search_results)} sources to analyze")
        
        return len(self.search_results) > 0
    
    def _prioritize_results(self, results):
        """Prioritize fact-checking and news sites"""
        # List of known fact-checking domains
        fact_check_domains = [
            'snopes.com', 'factcheck.org', 'politifact.com', 'reuters.com/fact-check',
            'apnews.com', 'bbc.com', 'usatoday.com/fact-check', 'fullfact.org',
            'washingtonpost.com/fact-checker', 'leadstories.com', 'afp.com/fact-checking'
        ]
        
        # Sort results by priority
        prioritized = sorted(results, key=lambda x: 
            (0 if any(fc in x['domain'] for fc in fact_check_domains) else 1,  # Fact-check sites first
             0 if 'fact check' in x['query'].lower() else 1))  # Results from fact check queries next
        
        return prioritized[:self.max_sources]
    
    def _is_social_media(self, url):
        """Check if URL is from social media"""
        social_media_domains = [
            'facebook.com', 'twitter.com', 'instagram.com', 'tiktok.com',
            'reddit.com', 'youtube.com', 'linkedin.com', 'pinterest.com'
        ]
        return any(domain in url.lower() for domain in social_media_domains)
    
    def _extract_domain(self, url):
        """Extract domain from URL"""
        match = re.search(r'https?://(?:www\.)?([^/]+)', url)
        return match.group(1) if match else url
    
    def _retrieve_article_content(self):
        """Retrieve article content from the identified sources"""
        print("\nRetrieving content from sources...")
        
        if self.use_mock:
            self.article_texts = self._generate_mock_article_texts()
            return len(self.article_texts) > 0
        
        for idx, source in enumerate(self.search_results):
            try:
                print(f"\nRetrieving content from {source['domain']} ({idx+1}/{len(self.search_results)})")
                
                # Try to load the webpage with timeout
                html_content = open_link(source['url'], timeout=30)
                
                if not html_content:
                    print(f"Failed to retrieve content from {source['url']}")
                    continue
                
                # Convert HTML to plaintext
                text_content = text_maker.handle(html_content)
                
                # Save relevant information
                self.article_texts.append({
                    'url': source['url'],
                    'domain': source['domain'],
                    'content': text_content[:10000],  # Limit to 10,000 chars
                    'timestamp': datetime.now().isoformat()
                })
                
                print(f"✅ Successfully retrieved content ({len(text_content)} chars)")
                
            except Exception as e:
                print(f"Error retrieving content from {source['url']}: {e}")
            
            # Check timeout
            if self.timeout_handler.check_timeout():
                print("Timeout reached. Stopping content retrieval.")
                break
        
        return len(self.article_texts) > 0
    
    def _extract_facts(self):
        """Extract relevant facts from the article contents"""
        print("\nExtracting relevant facts from sources...")
        
        if self.use_mock:
            self.article_facts = self._generate_mock_article_facts()
            return len(self.article_facts) > 0
        
        for idx, article in enumerate(self.article_texts):
            try:
                print(f"\nExtracting facts from {article['domain']} ({idx+1}/{len(self.article_texts)})")
                
                # Use LLM to extract facts if available
                if client:
                    prompt = f"""
                    Analyze the following article content regarding this claim: "{self.claim}"
                    
                    Extract the most relevant facts and statements that confirm or refute the claim.
                    Focus on direct evidence, expert opinions, and verifiable information.
                    
                    Article content:
                    {article['content'][:5000]}
                    
                    Please output only the key facts and statements related to the claim, organized as bullet points.
                    """
                    
                    response = client.generate_content(prompt)
                    extracted_facts = response.text if response else "No facts extracted"
                else:
                    # Fallback if no LLM available
                    # Extract paragraphs containing keywords from the claim
                    claim_keywords = self._extract_keywords(self.claim)
                    paragraphs = article['content'].split('\n\n')
                    relevant_paragraphs = []
                    
                    for paragraph in paragraphs:
                        if len(paragraph) > 100 and any(keyword in paragraph.lower() for keyword in claim_keywords):
                            relevant_paragraphs.append(paragraph)
                    
                    extracted_facts = "\n".join(relevant_paragraphs[:5])  # Take top 5 relevant paragraphs
                
                # Save extracted facts
                self.article_facts.append({
                    'url': article['url'],
                    'domain': article['domain'],
                    'facts': extracted_facts,
                    'timestamp': datetime.now().isoformat()
                })
                
                print(f"✅ Successfully extracted facts")
                
            except Exception as e:
                print(f"Error extracting facts from {article['domain']}: {e}")
            
            # Check timeout
            if self.timeout_handler.check_timeout():
                print("Timeout reached. Stopping fact extraction.")
                break
        
        return len(self.article_facts) > 0
    
    def _extract_keywords(self, text):
        """Extract keywords from text"""
        # Remove common words
        common_words = ['the', 'a', 'an', 'and', 'or', 'but', 'of', 'to', 'in', 'on', 'at', 'for', 'with', 'by']
        words = text.lower().split()
        keywords = [word for word in words if word not in common_words and len(word) > 3]
        return keywords
    
    def _analyze_facts(self):
        """Analyze the extracted facts to determine if the claim is true or false"""
        print("\nAnalyzing facts to determine veracity of the claim...")
        
        if self.use_mock:
            return self._generate_mock_analysis()
        
        # Compile all facts into one document
        all_facts = []
        for article in self.article_facts:
            all_facts.append(f"Source: {article['domain']}\n{article['facts']}\n")
        
        all_facts_text = "\n".join(all_facts)
        
        # Try to use the ML model integration first
        model_integration = get_model_integration()
        if model_integration.enabled:
            print("Using ML model for analysis...")
            result = model_integration.analyze_claim(self.claim, all_facts_text)
        else:
            # Fall back to using Gemini if available
            if client:
                print("Using Gemini for analysis...")
                prompt = f"""
                Analyze the following claim and the facts collected from various sources:
                
                CLAIM: "{self.claim}"
                
                FACTS FROM MULTIPLE SOURCES:
                {all_facts_text}
                
                Based on these facts, determine whether the claim is TRUE, FALSE, PARTIALLY TRUE, or UNVERIFIABLE.
                
                Provide:
                1. Your verdict (TRUE/FALSE/PARTIALLY TRUE/UNVERIFIABLE)
                2. Confidence level (High/Medium/Low)
                3. Key evidence supporting your verdict
                4. Any competing evidence or alternative perspectives
                5. Reasoning for your conclusion
                
                Format your response as JSON, using the following structure:
                {{
                    "verdict": "TRUE/FALSE/PARTIALLY TRUE/UNVERIFIABLE",
                    "confidence": "High/Medium/Low",
                    "key_evidence": ["evidence point 1", "evidence point 2", ...],
                    "competing_evidence": ["competing point 1", "competing point 2", ...],
                    "reasoning": "Your reasoning for the verdict"
                }}
                """
                
                response = client.generate_content(prompt)
                
                try:
                    # Try to extract JSON from the response
                    result = self._extract_json_from_text(response.text)
                    if not result:
                        result = {
                            "verdict": "UNVERIFIABLE",
                            "confidence": "Low",
                            "key_evidence": ["Could not analyze the facts properly"],
                            "competing_evidence": [],
                            "reasoning": "Technical error in analyzing the facts"
                        }
                except Exception as e:
                    print(f"Error parsing analysis result: {e}")
                    result = {
                        "verdict": "UNVERIFIABLE",
                        "confidence": "Low",
                        "key_evidence": ["Error in analysis"],
                        "competing_evidence": [],
                        "reasoning": f"Error: {str(e)}"
                    }
            else:
                # Fallback if no LLM available
                result = {
                    "verdict": "UNVERIFIABLE",
                    "confidence": "Low",
                    "key_evidence": ["No language model available for analysis"],
                    "competing_evidence": [],
                    "reasoning": "Without language model capabilities, detailed analysis is not possible"
                }
        
        # Add metadata to the result
        result["claim"] = self.claim
        result["timestamp"] = datetime.now().isoformat()
        result["sources"] = [{'domain': article['domain'], 'url': article['url']} for article in self.article_facts]
        
        self.analysis_result = result
        
        # Print summary
        print(f"\nVerdict: {result['verdict']} (Confidence: {result['confidence']})")
        print(f"Reasoning: {result['reasoning']}")
        
        return result
    
    def _extract_json_from_text(self, text):
        """Extract JSON data from text"""
        # Find JSON pattern
        json_match = re.search(r'({[\s\S]*})', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # If no JSON found or parsing failed, try to create JSON from structured text
        result = {}
        
        verdict_match = re.search(r'verdict[:\s]+"?([^",\n]+)"?', text, re.IGNORECASE)
        if verdict_match:
            result['verdict'] = verdict_match.group(1).strip()
        
        confidence_match = re.search(r'confidence[:\s]+"?([^",\n]+)"?', text, re.IGNORECASE)
        if confidence_match:
            result['confidence'] = confidence_match.group(1).strip()
        
        # Extract key evidence
        evidence_list = []
        evidence_section = re.search(r'key_evidence[\s\S]+?(?:competing_evidence|reasoning)', text, re.IGNORECASE)
        if evidence_section:
            evidence_items = re.findall(r'"([^"]+)"', evidence_section.group(0))
            evidence_list = evidence_items if evidence_items else []
        
        result['key_evidence'] = evidence_list
        
        # Extract competing evidence
        competing_list = []
        competing_section = re.search(r'competing_evidence[\s\S]+?(?:reasoning)', text, re.IGNORECASE)
        if competing_section:
            competing_items = re.findall(r'"([^"]+)"', competing_section.group(0))
            competing_list = competing_items if competing_items else []
        
        result['competing_evidence'] = competing_list
        
        # Extract reasoning
        reasoning_match = re.search(r'reasoning[:\s]+"?([^"]+)', text, re.IGNORECASE)
        if reasoning_match:
            result['reasoning'] = reasoning_match.group(1).strip()
        
        return result
    
    def _generate_fallback_result(self, reason):
        """Generate a fallback result when analysis can't be completed"""
        result = {
            "verdict": "UNVERIFIABLE",
            "confidence": "Low",
            "key_evidence": [],
            "competing_evidence": [],
            "reasoning": reason,
            "claim": self.claim,
            "timestamp": datetime.now().isoformat(),
            "sources": []
        }
        
        results_path = save_results(result, "fact_check")
        print(f"\nCould not complete analysis. Fallback results saved to: {results_path}")
        
        return result
    
    def _generate_mock_search_results(self):
        """Generate mock search results for testing"""
        mock_domains = [
            'factcheck.org', 'snopes.com', 'reuters.com/fact-check', 'apnews.com',
            'politifact.com', 'usatoday.com/fact-check', 'washingtonpost.com/fact-checker', 'bbc.com/news/reality_check'
        ]
        
        # Generate different URLs for different domains
        mock_results = []
        for domain in mock_domains[:self.max_sources]:
            # Create a URL-friendly version of the claim
            url_claim = '-'.join(self.claim.split()[:3]).lower()
            mock_results.append({
                'url': f'https://www.{domain}/fact-check/{url_claim}-{random.randint(1000, 9999)}',
                'domain': domain,
                'query': 'fact check ' + self.claim
            })
        
        return mock_results
    
    def _generate_mock_article_texts(self):
        """Generate mock article texts for testing"""
        mock_texts = []
        
        claim_words = self.claim.split()
        claim_subject = ' '.join(claim_words[:3]) if len(claim_words) >= 3 else self.claim
        
        for source in self.search_results:
            truth_value = random.choice(['true', 'false', 'partially true', 'unverifiable'])
            
            mock_content = f"""
            # Fact Check: {self.claim}
            
            We investigated the claim that {self.claim} and found it to be {truth_value}.
            
            ## Background
            
            The claim appeared on social media and various news outlets in recent weeks.
            Our researchers examined multiple primary sources to verify the accuracy.
            
            ## Analysis
            
            According to experts in the field, the evidence {
                'supports' if truth_value == 'true' else 
                'contradicts' if truth_value == 'false' else 
                'partially supports' if truth_value == 'partially true' else
                'is insufficient to determine'
            } the claim.
            
            ## Key Facts
            
            - Research from {random.choice(['Harvard', 'Stanford', 'Oxford', 'MIT'])} found that {
                'the claim is accurate' if truth_value == 'true' else
                'the claim is inaccurate' if truth_value == 'false' else
                'parts of the claim are accurate while others are not' if truth_value == 'partially true' else
                'there is insufficient evidence to verify the claim'
            }.
            
            - Multiple {random.choice(['studies', 'reports', 'investigations', 'experts'])} have {
                'confirmed' if truth_value == 'true' else
                'debunked' if truth_value == 'false' else
                'found mixed results regarding' if truth_value == 'partially true' else
                'been unable to conclusively verify'
            } the assertion that {self.claim}.
            
            ## Conclusion
            
            Based on our investigation, we rate this claim {truth_value.upper()}.
            """
            
            mock_texts.append({
                'url': source['url'],
                'domain': source['domain'],
                'content': mock_content,
                'timestamp': datetime.now().isoformat()
            })
        
        return mock_texts
    
    def _generate_mock_article_facts(self):
        """Generate mock article facts for testing"""
        mock_facts = []
        verdicts = ['TRUE', 'FALSE', 'PARTIALLY TRUE', 'UNVERIFIABLE']
        verdict_weights = {
            'TRUE': 0.3,
            'FALSE': 0.4,  # Slightly higher chance of false
            'PARTIALLY TRUE': 0.2,
            'UNVERIFIABLE': 0.1
        }
        
        # Choose an overall verdict with weighted probability
        overall_verdict = random.choices(
            population=list(verdict_weights.keys()),
            weights=list(verdict_weights.values()),
            k=1
        )[0]
        
        # Generate facts aligned with the chosen overall verdict
        for idx, article in enumerate(self.article_texts):
            # For variation, some sources may have different opinions
            if idx == 0 or random.random() < 0.7:  # 70% chance to align with overall verdict
                article_verdict = overall_verdict
            else:
                # Choose a different verdict
                other_verdicts = [v for v in verdicts if v != overall_verdict]
                article_verdict = random.choice(other_verdicts)
            
            facts = []
            
            if article_verdict == 'TRUE':
                facts = [
                    f"Our investigation found strong evidence supporting the claim that {self.claim}.",
                    f"Multiple credible sources confirmed the accuracy of this assertion.",
                    f"Experts in the field have verified the key aspects of this claim.",
                    f"Statistical data from {random.choice(['recent studies', 'government reports', 'academic research'])} confirms this information."
                ]
            elif article_verdict == 'FALSE':
                facts = [
                    f"Our fact checkers found no evidence supporting the claim that {self.claim}.",
                    f"This assertion contradicts well-established facts from reliable sources.",
                    f"Experts in the field have refuted key aspects of this claim.",
                    f"Statistical data from {random.choice(['recent studies', 'government reports', 'academic research'])} contradicts this information."
                ]
            elif article_verdict == 'PARTIALLY TRUE':
                facts = [
                    f"Our investigation found that parts of the claim that {self.claim} are accurate, while others are not.",
                    f"The claim contains elements of truth but includes significant inaccuracies or exaggerations.",
                    f"While the core idea has some merit, important context is missing from the original claim.",
                    f"Some aspects are supported by evidence, but others are contradicted by reliable sources."
                ]
            else:  # UNVERIFIABLE
                facts = [
                    f"Our fact checkers could not find sufficient evidence to verify or refute the claim that {self.claim}.",
                    f"Available information is too limited to make a determination about this claim.",
                    f"Experts disagree significantly about the accuracy of this assertion.",
                    f"The claim contains elements that cannot be independently verified at this time."
                ]
            
            # Add random specific fact
            facts.append(f"According to {random.choice(['Dr. Smith at Harvard', 'a recent CDC report', 'court documents', 'internal records'])}, the specific assertion about {self.claim.split()[-3:]} is {article_verdict.lower().replace('_', ' ')}.")
            
            mock_facts.append({
                'url': article['url'],
                'domain': article['domain'],
                'facts': "\n".join([f"• {fact}" for fact in facts]),
                'timestamp': datetime.now().isoformat()
            })
        
        return mock_facts
    
    def _generate_mock_analysis(self):
        """Generate mock analysis result for testing"""
        verdicts = ['TRUE', 'FALSE', 'PARTIALLY TRUE', 'UNVERIFIABLE']
        confidence_levels = ['High', 'Medium', 'Low']
        
        # Count verdicts from article facts
        verdict_counts = {}
        for article in self.article_facts:
            facts_text = article['facts'].lower()
            
            if 'true' in facts_text and 'not true' not in facts_text and 'false' not in facts_text:
                verdict = 'TRUE'
            elif 'false' in facts_text or 'not true' in facts_text:
                verdict = 'FALSE'
            elif 'partially true' in facts_text or 'partly true' in facts_text or 'mixed' in facts_text:
                verdict = 'PARTIALLY TRUE'
            else:
                verdict = 'UNVERIFIABLE'
            
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        
        # Determine the most common verdict
        if verdict_counts:
            max_count = max(verdict_counts.values())
            most_common_verdicts = [v for v, c in verdict_counts.items() if c == max_count]
            final_verdict = random.choice(most_common_verdicts)
            
            # Determine confidence based on consensus
            if max_count >= len(self.article_facts) * 0.7:
                confidence = 'High'
            elif max_count >= len(self.article_facts) * 0.5:
                confidence = 'Medium'
            else:
                confidence = 'Low'
        else:
            final_verdict = random.choice(verdicts)
            confidence = random.choice(confidence_levels)
        
        # Generate key evidence
        key_evidence = []
        for article in self.article_facts:
            facts_list = article['facts'].split('\n')
            for fact in facts_list:
                if fact.strip() and len(key_evidence) < 5:  # Limit to 5 key pieces of evidence
                    cleaned_fact = fact.strip('• ').strip()
                    if cleaned_fact and len(cleaned_fact) > 10:
                        key_evidence.append(f"{cleaned_fact} (Source: {article['domain']})")
        
        # Generate competing evidence (for non-unanimous verdicts)
        competing_evidence = []
        if confidence != 'High':
            for _ in range(random.randint(1, 3)):
                competing_evidence.append(f"Some sources suggest {random.choice(['different interpretations', 'contrasting data', 'alternative explanations', 'other perspectives'])}.")
        
        # Generate reasoning
        reasoning_templates = {
            'TRUE': [
                "Multiple reliable sources confirm this claim with consistent evidence.",
                "The preponderance of evidence from fact-checking organizations supports this assertion.",
                "Available data from credible sources aligns with the claim."
            ],
            'FALSE': [
                "Multiple reliable sources refute this claim with consistent evidence.",
                "The preponderance of evidence from fact-checking organizations contradicts this assertion.",
                "Available data from credible sources does not support the claim."
            ],
            'PARTIALLY TRUE': [
                "While some aspects of the claim are accurate, others contain significant inaccuracies.",
                "The claim contains elements of truth but omits important context or exaggerates key details.",
                "Different sources confirm some parts of the claim while refuting others."
            ],
            'UNVERIFIABLE': [
                "There is insufficient evidence available to verify or refute this claim conclusively.",
                "Reliable sources provide contradictory information that prevents a definitive judgment.",
                "The claim contains elements that cannot be independently verified at this time."
            ]
        }
        
        reasoning = random.choice(reasoning_templates[final_verdict])
        
        # Add specific details to reasoning
        if key_evidence:
            reasoning += f" Notably, {random.choice(key_evidence).split('(Source')[0].strip()}."
        
        # Add consensus information
        if confidence == 'High':
            reasoning += f" There is strong consensus across multiple sources regarding this conclusion."
        elif confidence == 'Medium':
            reasoning += f" There is moderate agreement among sources, though some disagreement exists."
        else:
            reasoning += f" There is significant disagreement among sources about this claim."
        
        # Create the final result
        result = {
            "verdict": final_verdict,
            "confidence": confidence,
            "key_evidence": key_evidence,
            "competing_evidence": competing_evidence,
            "reasoning": reasoning,
            "claim": self.claim,
            "timestamp": datetime.now().isoformat(),
            "sources": [{'domain': article['domain'], 'url': article['url']} for article in self.article_facts]
        }
        
        return result

class TimeoutHandler:
    def __init__(self, max_runtime_seconds=300):  # Default 5 minute timeout
        self.max_runtime = max_runtime_seconds
        self.start_time = time.time()
    
    def check_timeout(self):
        elapsed = time.time() - self.start_time
        if elapsed > self.max_runtime:
            print(f"\nTimeout reached after {int(elapsed)} seconds. Stopping execution.")
            return True
        return False

def extract_urls(text):
    """Extract URLs from text"""
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[-\w%!./?=&+#]*)*'
    urls = re.findall(url_pattern, text)
    
    # Remove duplicates while preserving order
    unique_urls = []
    for url in urls:
        if url not in unique_urls:
            unique_urls.append(url)
    
    # If no URLs found in the response, use these fallback fact-checking sites
    if not urls:
        print("No specific URLs found in the search results. Using fallback fact-checking websites.")
        urls = [
            "https://www.snopes.com/fact-check/",
            "https://www.factcheck.org/",
            "https://www.politifact.com/",
            "https://apnews.com/hub/ap-fact-check",
            "https://www.reuters.com/fact-check/"
        ]
    
    return unique_urls

# Main function to run the detector
def main():
    # Check for mock mode
    use_mock = os.getenv("MOCK_MODE", "False").lower() == "true"
    
    # Display disclaimer if using mock mode
    if use_mock:
        print("\n" + "!"*70)
        print("! MOCK MODE ENABLED - RESULTS ARE SIMULATED AND NOT REAL FACT-CHECKING !")
        print("! This is only for testing the application structure and functionality !")
        print("!"*70 + "\n")
    
    # Initialize the detector
    detector = FakeNewsDetector(use_mock=use_mock)
    
    # Get claim from arguments or use a default claim
    if len(sys.argv) > 1:
        claim = " ".join(sys.argv[1:])
    else:
        claim = "Drinking water with lemon every morning cures cancer"
    
    # Run the detection
    result = detector.detect(claim)
    
    # Print a summary of the result
    print("\n" + "="*50)
    print(f"CLAIM: {result['claim']}")
    print(f"VERDICT: {result['verdict']} (Confidence: {result['confidence']})")
    print("-"*50)
    print(f"REASONING: {result['reasoning']}")
    print("="*50)
    
    # Final disclaimer for mock mode
    if use_mock:
        print("\nNOTE: This was a SIMULATED fact-check using mock data.")
        print("For real fact-checking, disable MOCK_MODE and ensure all dependencies are installed.")
    
    return result

if __name__ == "__main__":
    main() 