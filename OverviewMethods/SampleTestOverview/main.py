from openai import OpenAI
import os
import requests
from bs4 import BeautifulSoup
import tiktoken
from datetime import datetime
from urllib.parse import urljoin, urlparse
import csv
import time


# === SETUP ===

start_time = time.time()
client = OpenAI(api_key="")
today = datetime.today().strftime("%d %b %Y")  # e.g. 17 Jun 2025
model_temperature = 0.4
searchContextSize = "high"

#=========== Request Params ==================

instructions_forSearchTool = f"""
You are an expert research assistant. Your task is to extract structured, verified company information from the business’s official website and wider online presence.

Use the official website as the primary source. To confirm details such as turnover, employee estimates, and key people, also consult official databases like Companies House, Endole, or trusted business directories (e.g. GOV.UK). Use explicit data wherever possible. If an estimate is necessary, clearly mark estimates using a tilde (~).

Be cautious with external sources. Many UK companies share similar names. Only include external information if the source clearly matches the correct business — e.g. by domain name, physical address, or confirmed directors. Disregard content from similarly named businesses if you cannot be certain of the match.

### Fields:
- **Est**: Just the year the business was founded.
- **Employees/Turnover**: Prioritise official data (e.g. Companies House, Endole); otherwise give reasonable estimates based on the website or other verified sources.
-**Overview**: Always include a brief sentence (under 15 words) about location, scale, and service area, even for companies without relevant products — e.g. “Based in Norwich, installs windows and offers free site surveys.”
- **Showroom**: If not mentioned, write “None”. If present, briefly describe location and what is on display.
- **Supply/Install**: The client specialises in: roller garage doors, sectional garage doors, domestic roller shutters, industrial shutters, steel shutters, security shutters, retail roller shutters, LPCB-certified shutters, and fire shutters. They may also deal in commercial shutters, transparent shutters, insulated roller shutters, high-security shutters, and vision shutters. Confirm whether these are supplied and/or installed. Mention other categories only briefly.
- **Brand partnerships**: Check whether the company supplies or installs any of the following: Hörmann, Gliderol, SWS UK (SeceuroGlide), Novoferm, Garador, Birkdale, Samson Doors, Tradedor, Garolla. Confirm via website text, brochures, or product listings.
- **Key People**: List names and roles of directors, team members, or owners from verified sources. If reviews mention names, include them tentatively (e.g. “Paul (Installer, per reviews)”).

### Rules:
- Output must be a **single-line summary**, with fields separated by vertical bars `|`.
- Use semicolons **within** a field to separate multiple items.
- Never use “unknown” or “N/A” — use “not specified” if data is unavailable.
- Do not include hyperlinks or Markdown formatting.
- Avoid phrases like “the company is” — summarise facts cleanly.
- Do not return multiple lines, paragraphs, or bullet points.
- If the company does not appear to supply or install any relevant garage door or shutter products, still return a full structured line using the standard format. Use “None” for Supply/Install, “none specified” for Brand partnerships, and “not specified” for any missing data. Write a neutral Overview sentence such as: “Based in [location], with no listed shutter or garage door products.”

### Output Format (single line):
Est: [year]. Employees: [number or estimate]. Turnover: [£X.XM or £XXXK] | Overview: [brief sentence] | Showroom: [text or "None"] | Supply/Install: [products] | Brand partnerships: [brands] | Key people: [name + role] 

### Examples:
Est: 2018. Employees: ~5–10. Turnover: under £750k | Overview: Operates from Frome, covering Somerset and Wiltshire with mobile installation and repair services | Showroom: None | Supply/Install: roller garage doors, sectional garage doors, up‑and‑over doors, side-hinged doors, automatic garage doors; also handles repairs and spare parts | Brand partnerships: Hörmann, Garador, SWS UK (SeceuroGlide) | Key people: Laura Bennett (Director), Mark Hughes (Senior Installer), Jake (Engineer, per reviews) 

Est: 2013 Employees: 17. Turnover: £1.2M | Overview: Operates from Wigan, offering mobile installation of doors and shutters with repair services across the North of England | Showroom: Small showroom in Wigan with Garage Door displays | Supply/Install: roller shutters, garage doors, fire curtains, automatic doors, manual doors | Brand partnerships: Garador, Hörmann, Cardale, SeceuroGlide | Key people: Jamie Clarke (Secretary); Lee Hogg (Director); Stephen Thomas Pendergrast (Director)

Est: 2005. Employees: ~11–50. Turnover: ~ £2.5-3M | Overview: Leeds-based specialists in fire-rated and security shutters, partnered with Hörmann | Showroom: Head office in Leeds with product displays | Supply/Install: insulated roller garage doors, steel security shutters, LPCB-certified fire shutters | Brand partnerships: Hörmann; Gliderol | Key people: Sarah Langford (Managing Director), Tom Hughes (Technical Director)

"""

#USE ENDOLE FOR TURNOVER MICRO BUSINESS ETC


instructions_forWebsiteText = f"""
You are an expert research assistant. Your task is to extract structured company information using only the official website content provided.

Use the website text as your only source. Do not attempt to include or verify information from any external databases (e.g. Companies House, Endole) or directories. Only use what is explicitly or clearly implied in the provided content.

If some details (e.g. employee count or turnover) are not stated, you may give reasonable estimates if they can be inferred from the site content. Clearly mark estimates using a tilde (~).

### Fields:
- **Est**: The year the business was founded, if available in the website text.
- **Employees/Turnover**: Estimate from the text if not explicitly stated.
- **Overview**: Always include a brief sentence (under 15 words) about location, scale, and service area — e.g. “Based in Norwich, installs windows and offers free site surveys.”
- **Showroom**: If not mentioned, write “None”. If present, briefly describe location and what is on display.
- **Supply/Install**: The client specialises in: roller garage doors, sectional garage doors, domestic roller shutters, industrial shutters, steel shutters, security shutters, retail roller shutters, LPCB-certified shutters, and fire shutters. They may also deal in commercial shutters, transparent shutters, insulated roller shutters, high-security shutters, and vision shutters. Confirm whether these are supplied and/or installed. Mention other categories only briefly.
- **Brand partnerships**: Check if the website mentions any of the following brands: Hörmann, Gliderol, SWS UK (SeceuroGlide), Novoferm, Garador, Birkdale, Samson Doors, Tradedor, Garolla. Include only those explicitly mentioned in the website text.
- **Key People**: List names and roles if included in the website text. If names appear in testimonials or team sections, include them tentatively (e.g. “Paul (Installer, per reviews)”).

### Rules:
- Output must be a **single-line summary**, with fields separated by vertical bars `|`.
- Use semicolons **within** a field to separate multiple items.
- Never use “unknown” or “N/A” — use “not specified” if data is unavailable.
- Do not include hyperlinks or Markdown formatting.
- Avoid phrases like “the company is” — summarise facts cleanly.
- Do not return multiple lines, paragraphs, or bullet points.
- If the company does not appear to supply or install any relevant garage door or shutter products, still return a full structured line using the standard format. Use “None” for Supply/Install, “none specified” for Brand partnerships, and “not specified” for any missing data. Use a neutral Overview sentence such as: “Based in [location], with no listed shutter or garage door products.”

### Output Format (single line):
Est: [year]. Employees: [number or estimate]. Turnover: [£X.XM or £XXXK] | Overview: [brief sentence] | Showroom: [text or "None"] | Supply/Install: [products] | Brand partnerships: [brands] | Key people: [name + role]

### Examples:
Est: 2018. Employees: ~5–10. Turnover: under £750k | Overview: Operates from Frome, covering Somerset and Wiltshire with mobile installation and repair services | Showroom: None | Supply/Install: roller garage doors, sectional garage doors, up‑and‑over doors, side-hinged doors, automatic garage doors; also handles repairs and spare parts | Brand partnerships: Hörmann, Garador, SWS UK (SeceuroGlide) | Key people: Laura Bennett (Director), Mark Hughes (Senior Installer), Jake (Engineer, per reviews)

Est: 2013 Employees: 17. Turnover: £1.2M | Overview: Operates from Wigan, offering mobile installation of doors and shutters with repair services across the North of England | Showroom: Small showroom in Wigan with Garage Door displays | Supply/Install: roller shutters, garage doors, fire curtains, automatic doors, manual doors | Brand partnerships: Garador, Hörmann, Cardale, SeceuroGlide | Key people: Jamie Clarke (Secretary); Lee Hogg (Director); Stephen Thomas Pendergrast (Director)

Est: 2005. Employees: ~11–50. Turnover: ~ £2.5-3M | Overview: Leeds-based specialists in fire-rated and security shutters, partnered with Hörmann | Showroom: Head office in Leeds with product displays | Supply/Install: insulated roller garage doors, steel security shutters, LPCB-certified fire shutters | Brand partnerships: Hörmann; Gliderol | Key people: Sarah Langford (Managing Director), Tom Hughes (Technical Director)
"""






#=========================Cost Calcluating =====================

PRICES = {
    "gpt-4.1": {
        "input": 0.01 /1000,
        "output": 0.03 /1000,
        "tool_use": 50/1000  # Web Search Tool Call 0.035 for medium context
    },
    "gpt-4": {
        "input": 0.03 / 1000,
        "output": 0.06 / 1000
    }
}
def estimate_cost(usage, model, tool_used=False):
    if model == "gpt-4.1":
        input_tokens = getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0))
        output_tokens = getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0))
        cost = (
            input_tokens * PRICES["gpt-4.1"]["input"]
            + output_tokens * PRICES["gpt-4.1"]["output"]
        )
        if tool_used:
            cost += PRICES["gpt-4.1"]["tool_use"]
    elif model == "gpt-4":
        input_tokens = getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0))
        output_tokens = getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0))
        cost = (
            input_tokens * PRICES["gpt-4"]["input"]
            + output_tokens * PRICES["gpt-4"]["output"]
        )
    else:
        cost = 0
    return round(cost, 5)


# ***********===================== Methods of overview generation  ===================********

# ====================== Using the Web Preview search tool with differnet context sizes =====================================
def Search_Tool_Overview(company_website, company_name, ContextSize):
    response = client.responses.create(
        model="gpt-4.1",
        tools=[{
            "type": "web_search_preview",
            "search_context_size": ContextSize,
            "user_location": {
                "type": "approximate",
                "country": "GB"
            }
        }],
        instructions=instructions_forSearchTool,
        temperature=model_temperature,
        input=f"Company name: {company_name}\nCompany website: {company_website}"
    )

    cost = estimate_cost(response.usage, model="gpt-4.1", tool_used=True) #using the search tool
    message = safe_extract_openai_message(response)
    clean_message = message.replace("\n", " ").replace("\r", " ") #ensure single line for CSV
    final_output = f'"{clean_message} – ChatGPT {today}"'


    return final_output, cost

#=================== BS4 searching all internal pages until some character limit reached =========================

def Full_Website_Overview(base_url, company_name, max_chars):
    visited = set()
    to_visit = [base_url] #use stack structure to track links that need searching
    full_text = ""

    while to_visit:

        current_url = to_visit.pop(0)
        if current_url in visited: #avoid reeating visits
            continue

        try:
            #user agent mozilla to avoid being blocked, in headers param
            response = requests.get(current_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            #extract and clean visible text
            page_text = " ".join(soup.stripped_strings)
            full_text += page_text + " "
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
                    parsed.scheme in ["http", "https"] and
                    parsed.netloc == urlparse(base_url).netloc and
                    not full_url.endswith(('.pdf', '.jpg', '.png', '.jpeg')) and
                    not full_url.startswith(("mailto:", "tel:", "#")) and
                    full_url not in visited and
                    full_url not in to_visit
                ):
                    to_visit.append(full_url)

        except Exception as e:
            print(f"Failed to load {current_url}: {e}")

    full_text = full_text[:max_chars]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": instructions_forWebsiteText},
            {"role": "user", "content": full_text}
        ],
        temperature=model_temperature
    )

    cost = estimate_cost(response.usage, model="gpt-4", tool_used=False)
    message = safe_extract_openai_message(response)
    clean_message = message.replace("\n", " ").replace("\r", " ")
    final_output = f'"{clean_message} – ChatGPT {today}"'

    return final_output, cost

#================= Overview with just home page (redundant) ========================

def Home_Page_Overview(url, company_name, max_chars):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        #Extract visible text
        texts = soup.stripped_strings
        visible_text = " ".join(texts)
        visible_text = visible_text[:max_chars]

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return "ERROR: Failed to load homepage", 0

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": instructions_forWebsiteText},
            {"role": "user", "content": visible_text}
        ],
        temperature=model_temperature
    )

    cost = estimate_cost(response.usage, model="gpt-4", tool_used=False)
    message = safe_extract_openai_message(response)
    clean_message = message.replace("\n", " ").replace("\r", " ")

    final_output = f'"{clean_message} – ChatGPT {today}"'

    return final_output, cost


# ============= safe message extraction ===============

def safe_extract_openai_message(response, fallback_text="ERROR: No content returned"):
    """
    Safely extracts message content from an OpenAI response.
    Supports both newer `.output[].content[].text` (gpt-4.1 tools) and legacy `.choices[0].message.content`.
    """
    # new tool-enhanced response format
    if hasattr(response, "output") and isinstance(response.output, list):
        for item in response.output:
            if hasattr(item, "content") and isinstance(item.content, list):
                for content_piece in item.content:
                    if hasattr(content_piece, "text") and content_piece.text:
                        return content_piece.text.strip()

    #(GPT-4 chat without tools)
    if (
        hasattr(response, "choices")
        and isinstance(response.choices, list) #check if it is in a  list format
        and len(response.choices) > 0
        and hasattr(response.choices[0], "message")
        and hasattr(response.choices[0].message, "content")
        and response.choices[0].message.content
    ):
        return response.choices[0].message.content.strip()

    print("Response structure invalid or empty:\n", response)
    return fallback_text


#====================== INPUT DATA ========================

company_names = [
     "Security Direct Products",
     "Garage Conversion",
     "warrior doors ltd",
     "klm Doors & Shutters",
     "Ideal Garage Door Company Ltd",
     "JBH Design",
     "VOCA Solutions",
     "Severn Valley Window & Door",
     "Phoenix Systems",
     "Kent Door Store"]
company_websites = [
     "https://securitydirectuk.com/",
     "https://www.garageconversions.co.uk/",
     "https://warriordoors.co.uk/",
     "https://www.klmdoors.co.uk/",
     "https://www.idealgaragedoorcompany.co.uk/",
     "https://jbhdesign.co.uk/",
     "https://vocadoortechnologies.co.uk/",
     "https://www.severnvalleywindows.co.uk/",
     "https://www.phoenix-sys.co.uk/",
    "https://www.kentdoorstore.co.uk/"]

companyNames2 = ["Aluroll", "Roche Security"]
companyWebsites2 = ["https://www.aluroll.co.uk/", "https://www.rochesecurity.com/" ]




#====================== EXECUTING FUNCTIONS ====================

def create_result_csv(filename=None):
    if not filename:
        filename = f"overview_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    path = os.path.join(os.getcwd(), filename)
    with open(path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["SearchType", "CompanyName", "Message", "Cost"])
    return path


csv_file_path = create_result_csv()

for i in range(len(companyNames2)):  # Loop through all companies TEST FOR ALUROLL
    TestWebsite = companyWebsites2[i]
    TestName = companyNames2[i]


    SearchMessage, SearchCost = Search_Tool_Overview(TestWebsite, TestName, searchContextSize)
    #FullWebsiteMessage, FullWebsiteCost = Full_Website_Overview(TestWebsite, TestName, max_chars=10000)
    #HomeMessage, HomeCost = Home_Page_Overview(TestWebsite, TestName, max_chars=10000)
    print(f"{TestName} processed.")

    # Save to CSV
    with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["SearchTool", TestName, SearchMessage, SearchCost])
        #writer.writerow(["FullWebsiteBS4", TestName, FullWebsiteMessage, FullWebsiteCost])
        #writer.writerow(["HomePageOnly", TestName, HomeMessage, HomeCost])
        print(f"{TestName} Messages saved to CSV.")


end_time = time.time()
print(f"\n Total runtime: {end_time - start_time:.2f} seconds")








