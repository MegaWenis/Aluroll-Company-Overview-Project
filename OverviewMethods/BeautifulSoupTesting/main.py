import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import tiktoken

standardOutput = "Est: ~2001. Employees: not specified. Turnover: not specified | Overview: Based in Northwest England, offers mobile installation and repair services across the North of England | Showroom: None | Supply/Install: roller shutters, garage doors, fire curtains, automatic doors, manual doors | Brand partnerships: none specified | Key people: not specified – ChatGPT 18 Jun 2025"

# count tokens

def count_tokens(text):
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


from urllib.parse import urljoin, urlparse

def Full_Website_Overview(base_url, company_name, max_chars):
    visited = set()
    to_visit = [base_url] #use stack structure to track links that need searching
    full_text = ""

    while to_visit:
        current_url = to_visit.pop(0)
        if current_url in visited:
            continue

        # Skip weak or template-like URLs
        skip_keywords = ["areas", "locations", "contact", "terms", "privacy", "cookies", "sitemap"]
        if any(kw in current_url.lower() for kw in skip_keywords):
            continue

        try:
            response = requests.get(current_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract and clean text only from main tags (skip nav/menu/footer)
            main_text = " ".join([p.get_text(strip=True) for p in soup.find_all(["p", "h1", "h2", "h3"])])

            # Skip if page is just repeated filler (already seen text)
            if main_text in full_text:
                continue

            full_text += main_text + " "
            visited.add(current_url)

            # Stop if character limit reached
            if len(full_text) >= max_chars:
                print("Max character limit reached.")
                break

            # Extract valid internal links
            for link in soup.find_all("a", href=True):
                href = link['href']
                full_url = urljoin(base_url, href)
                parsed = urlparse(full_url)

                if (
                        parsed.scheme in ["http", "https"]
                        and parsed.netloc == urlparse(base_url).netloc
                        and not full_url.endswith(('.pdf', '.jpg', '.png', '.jpeg'))
                        and not full_url.startswith(("mailto:", "tel:", "#"))
                        and full_url not in visited
                        and full_url not in to_visit
                ):
                    to_visit.append(full_url)

        except Exception as e:
            print(f"Failed to load {current_url}: {e}")

    full_text = full_text[:max_chars]
    return full_text

# Example use
total_text = 0
url = "https://www.garageconversions.co.uk/"
text = Full_Website_Overview(url, "bob", 6000)
input_tokens = count_tokens(text)
output_tokens = count_tokens(standardOutput)


print("\nSite characters\n", len(text))
print("\nTokens -", input_tokens)# just preview
inputGBP = input_tokens * (2 / 1000000) / 1.34
outputGBP = output_tokens * (8/100000) / 1.34

print("\nEst Cost £" + str(inputGBP+outputGBP))
