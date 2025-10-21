from openai import OpenAI
from datetime import datetime

# === SETUP ===


client = OpenAI(api_key="")
today = datetime.today().strftime("%d %b %Y")  # e.g. 17 Jun 2025

# === INPUT ===
company_name = "GLS Door Services Limited"
company_website = "https://glsdoorsystems.co.uk/"

# === BUILD REQUEST ===
instructions = f"""
You are a structured research assistant. Your task is to extract confirmed company information from the company’s official website and wider online presence.

The client specialises in: roller garage doors, sectional garage doors, shutters, industrial shutters, steel shutters, LPCB-certified shutters, and fire shutters. Prioritise confirming whether these are supplied or installed. If they make other related products just briefly mention them, but go in detail wherever you find information on garage doors and shutters.

Use the company’s website as the primary source, but also search official databases such as Companies House, Endole, or trusted business directories such as GOV UK websites to confirm turnover, employee estimates, and details on key people. Use explicit data where possible, only estimate if you cannot find the explicit data, make it clear where you are estimating.

Be cautious when using search results. Many UK companies have similar or identical names. Only use information from other websites (outside the official company website) if you are certain that it refers to the correct company, based on domain match, physical address, or confirmed key people. Disregard any content from similarly named businesses that cannot be clearly matched to the provided company name and website. If there is uncertainty about whether a source matches the correct business, do not use it.

Pay special attention to whether the company partners with or supplies any of the following preferred brands. Include brand partnerships if confirmed in text, product listings, downloadable documents, or credible online directories.:Hörmann, Gliderol, SWS UK (SeceuroGlide), Novoferm, Garador, Birkdale, Roché, Samson Doors


Output format (single line, human-readable):
Est: [year]. Employees: [estimate] | Turnover: [£X.XM or £XXXK] | Showroom: [brief description] | Supply/Install: [confirmed product types] | Brand partnerships: [especially from preferred list] | Key people: [name + role] | Phone: [number] | Website: [URL]

Rules:
- Use vertical bars (|) to separate major fields.
- Use semicolons within a field if multiple entries are needed.
- If turnover or employee numbers are not confirmed from a reliable source, you can add an estimate.
- Never include filler terms like “unknown” or “not available”.
- Do not end the task until all confirmed and relevant details have been gathered.
- Ensure the final output is accurate, concise, and suitable for briefing client-facing staff.

Example outputs:
Est: 2014. Employees: ~2–10 | Turnover: under £1 million | Showroom: ; based in Calne, Wiltshire with mobile demos and free quotes | Supply/Install: roller garage doors, sectional garage doors, up‑and‑over doors, automation & electric openers, garage door repairs & servicing | Brand partnerships: installs proprietary makes (Hörmann, Garador, SWS, Cardale, Novoferm, Henderson, Wessex, Wayne Dalton, Apex) — independent supplier | Key people: Ben Alexander Lee (Director, appointed Dec 2023) | Phone: 123435 324 | Website: https://www.decordors.co.uk/
Est: 2024. Employees: 2 | Turnover: under £500k | Showroom: None; operates from Cottingham with mobile demos across Radstock, Bath, Frome, Trowbridge, Shepton Mallet & Wells | Supply/Install: roller garage doors, sectional garage doors, up‑and‑over doors, side‑hinged doors, automatic garage doors; focused on repair, servicing & spare parts | Brand partnerships: Cardale, SWS, Hörmann spares compatibility | Key people: Richard Arthur Cheeseman (Managing Director), Ryan Daniel Orman (Director) | Phone: 07977927590 | Website: https://www.garagedoortech.co.uk/
Est: 2005. Employees: 11–50 | Turnover: £3.2M | Showroom: Head office in Leeds with product displays | Supply/Install: insulated roller garage doors, steel security shutters, LPCB-certified fire shutters | Brand partnerships: Hörmann; Gliderol | Key people: Sarah Langford (Managing Director), Tom Hughes (Technical Director) | Phone: 0113 876 5432 | Website: www.shuttersecure.co.uk
"""


# === MAKE REQUEST ===
response = client.responses.create(
    model="gpt-4.1",
    tools=[{
        "type": "web_search_preview",
        "search_context_size": "medium",
        "user_location": {
            "type": "approximate",
            "country": "GB",
            "city": "London",
            "region": "London"
        }
    }],
    instructions=instructions,
    input=f"Company name: {company_name}\nCompany website: {company_website}"
)

# === OUTPUT ===
print("\nGPT Company Overview:")
print(response.output_text)