import os
import requests
import openai
from bs4 import BeautifulSoup
import tiktoken
from datetime import datetime
from urllib.parse import urljoin, urlparse




# count tokens
def count_tokens(text, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

# input data
openai.api_key = ""
company_name = "GLS Door Systems"
company_url = "https://glsdoorsystems.co.uk/"
today = datetime.today().strftime("%d %b %Y")  # e.g. "17 Jun 2025"

#Scrape content from home page only
def fetch_website_text(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        #Extract visible text
        texts = soup.stripped_strings
        visible_text = " ".join(texts)
        return visible_text[:6000]  #limit tokens

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""


#Check for Multiple Pages
def fetch_multiple_pages(base_url, paths=["", "/services", "/about", "/contact", "/doors"]):
    full_text = ""
    for path in paths:
        try:
            url = base_url.rstrip("/") + path
            print(f" Fetching {url}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            page_text = " ".join(soup.stripped_strings)
            full_text += page_text + " "
        except Exception as e:
            print(f" Error loading {url}: {e}")
    return full_text[:6000]

#Thorough search for all pages
def fetch_all_internal_pages(base_url, max_pages=5):
    visited = set()
    to_visit = [base_url]
    full_text = ""

    while to_visit and len(visited) < max_pages:
        current_url = to_visit.pop(0)
        try:
            response = requests.get(current_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract and clean visible text
            page_text = " ".join(soup.stripped_strings)
            full_text += page_text + " "
            visited.add(current_url)

            # Extract valid internal links
            for link in soup.find_all("a", href=True):
                href = link['href']
                full_url = urljoin(base_url, href)
                parsed = urlparse(full_url)

                # Filter: internal HTML pages only, not visited, no anchors/mails/docs
                if (
                    parsed.scheme in ["http", "https"] and
                    parsed.netloc == urlparse(base_url).netloc and
                    not full_url.endswith(('.pdf', '.jpg', '.png', '.jpeg')) and
                    not full_url.startswith(("mailto:", "tel:", "#")) and
                    full_url not in visited and
                    full_url not in to_visit
                ):
                    to_visit.append(full_url)
                    print(f"🔗 Found page: {full_url}")

        except Exception as e:
            print(f"Failed to load {current_url}: {e}")

    return full_text  #can choose to limi with full_text[:MaxChar]

# Handle ChatGPT prompt
def generate_structured_summary(company_name, website_text):
    prompt = f"""
    You are a research assistant helping produce structured summaries of UK-based industrial and commercial door companies. The client specialises in roller garage doors, sectional garage doors, shutters, industrial shutters, steel shutters, LPCB-certified shutters, and fire shutters—so prioritise confirming whether these products are offered.

    Using only the content below from the company's official website and cautious inference, produce a one-line structured company summary for {company_name}. Include:  
    - Est. year (if available)  
    - Turnover (if available)  
    - Employee estimate  
    - Showroom description (if available)  
    - Product types supplied/installed (confirmed only)  
    - Brand partnerships  
    - Key people  

    ---
    {website_text}
    ---

    **Format it exactly like this (including speech marks and punctuation):**  
    "Est: 2019. Employees: 51–200 | Showroom: None; HQ in Chatham (Kent) with workshops and field demos | Supply/Install: industrial roller shutters (standard & insulated), fire-rated shutters, sectional overhead doors, folding shutter doors, high-speed doors, domestic/insulated roller garage doors, automatic gates & barriers, louvre panels, PVC strip curtains, loading-bay equipment | Brand partnerships: in-house manufactured; Nassau Door A/S distributor; SeceuroShutters? (implicit); accredited with NASSAU, SuperAGI mentions standards | Key people: Ben Williams (Business Development Manager), Justin Harris (Operations Manager), Peter Jamieson (Lead Engineer Scotland)"

    Do not add any analysis or extra text—just return the summary line in that exact format, without exceeding 100 words.
    """
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )


    final_output = f'{response.choices[0].message.content.strip()} – ChatGPT {today}'

    if response.choices and response.choices[0].message:
        final_output = f'{response.choices[0].message.content.strip()} – ChatGPT {today}'
    else:
        final_output = f"ERROR: No content returned – ChatGPT {today}"

    return final_output

#run
print(f"\n Scraping website for: {company_name} ({company_url})")
#website_text = fetch_website_text(company_url)
website_text = fetch_all_internal_pages(company_url)
token_count = count_tokens(website_text)
#print(f" Website token count: {token_count}")
#print(f" RAW - ", website_text)


if website_text: #chec for non empty website text
    print("\n Generating GPT summary...")
    summary = generate_structured_summary(company_name, website_text)
    print("\n GPT Company Overview:\n", summary)
else:
    print(" No usable content found on the website.")