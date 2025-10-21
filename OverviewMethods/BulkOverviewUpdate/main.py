#Batch of GPT web-search updates.
#Keeps a living master CSV and writes each batch to its own file just in case.

import os, re, time, pandas as pd
from datetime import datetime
from openai import OpenAI

# ======== Key Variables ==================

start_time = time.time()

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL          = "gpt-4.1"
TEMP           = 0.4
CONTEXT_SIZE   = "high"
ORIGINAL_CSV   = ""
MASTER_CSV     = ""
BATCH_LOG      = f"Batch_{datetime.today().strftime('%Y%m%d_%H%M%S')}.csv"
MAX_ROWS       = 0
SLEEP_SECONDS  = 1
MaxCharCount   = 400

client = OpenAI(api_key=API_KEY)

# ========= prompt webssite ===================

instructions_forSearchTool = f"""
You are a research assistant creating structured summaries of UK-based garage door and shutter companies. The client specialises in roller garage doors, sectional garage doors, domestic roller shutters, industrial shutters, steel shutters, LPCB-certified shutters, and fire shutters. Confirm whether these product types and brand partnerships are offered.

# Role and Objective
Your task is to summarise a company's key commercial and technical data for market comparison.

# Instructions
Using only confirmed public data (from the company's website, Endole, Companies House, LinkedIn, or verified directories), produce a one-line structured company summary

## Sub-categories for more detailed instructions
1. **Est. Year**: State the earliest available founding or incorporation year.
2. **Turnover**:
   - Use Endole or Companies House if available.
   - If not, cautiously estimate based on size and type of operation.  
   - Use micro business threshold: turnover under £632k.
3. **Employees**:
   - Use Endole, LinkedIn, team pages, job listings or implicit data (e.g. fleet size).
   - If unclear, provide estimated ranges with "~".
4. **Overview**:
   - Describe HQ base and operating region (e.g. "based in Frome, covers Somerset").
5. **Showroom**:
   - Only mention if explicitly stated (e.g. “Small showroom in Wigan”).
6. **Supply/Install**:
   - Confirmed products only. Include repairs/spare parts if mentioned.
7. **Brand partnerships**:
   - Use confirmed names (e.g. Hörmann, Gliderol, SWS UK).
8. **Key people**:
   - Use named staff only.

# Reasoning Steps
- Use web search to find relevant company data.
- Prioritise Endole, Companies House, team pages, and brand pages.
- If no exact figures exist, infer responsibly and indicate with '~'.

# Output Format
Respond in this exact format:
"Est: [year]. Employees: [range or number]. Turnover: [confirmed figure or estimated range] | Overview: [headquarters, coverage area, operations] | Showroom: [if any] | Supply/Install: [confirmed products] | Brand partnerships: [list] | Key people: [name(s), role(s)]"



# Examples
Est: 2018. Employees: ~5–10. Turnover: under £750k | Overview: Operates from Frome, covering Somerset and Wiltshire with mobile installation and repair services | Showroom: None | Supply/Install: roller garage doors, sectional garage doors, up‑and‑over doors, side-hinged doors, automatic garage doors; also handles repairs and spare parts | Brand partnerships: Hörmann, Garador, SWS UK (SeceuroGlide) | Key people: Laura Bennett (Director), Mark Hughes (Senior Installer), Jake (Engineer, per reviews)

Est: 2013 Employees: 17. Turnover: £1.2M | Overview: Operates from Wigan, offering mobile installation of doors and shutters with repair services across the North of England | Showroom: Small showroom in Wigan with Garage Door displays | Supply/Install: roller shutters, garage doors, fire curtains, automatic doors, manual doors | Brand partnerships: Garador, Hörmann, Cardale, SeceuroGlide | Key people: Jamie Clarke (Secretary); Lee Hogg (Director); Stephen Thomas Pendergrast (Director)

Est: 2005. Employees: ~11–50. Turnover: ~ £2.5-3M | Overview: Leeds-based specialists in fire-rated and security shutters, partnered with Hörmann | Showroom: Head office in Leeds with product displays | Supply/Install: insulated roller garage doors, steel security shutters, LPCB-certified fire shutters | Brand partnerships: Hörmann; Gliderol | Key people: Sarah Langford (Managing Director), Tom Hughes (Technical Director)

# Context
Competitor and product categories include:
- Brands: Hörmann, Gliderol, SWS UK (SeceuroGlide), Novoferm, Garador, Birkdale, Samson Doors, Tradedor, Garolla, Thermosecure, Garage Door Systems, CGT Security, RCS Doors, Alluguard Ltd, TWF Roller Garage Doors, Hurricane Shutters & Doors, Crocodile Products LTD, Charter Global, Westwood Security Shutters, SSS Industrial Doors, Mechdoors Ltd, Kemen Roller Shutters Ltd, Assa Abloy Ltd, Alliance Doors, A1 Shutters
- Products: roller garage doors, sectional garage doors, domestic roller shutters, industrial shutters, steel shutters, security shutters, retail roller shutters, LPCB-certified shutters, fire shutters

# Final instructions
Use structured reasoning. Confirm data wherever possible. If not available, estimate responsibly and clearly indicate it with a tilde (~). Do not add speculative details.

Avoid including URLs, hyperlinks, or web addresses in the output. Keep the summary concise—aim for no more than ~3 lines of text total.

Do not list long blocks of services, project histories, or job titles. Instead, summarise focus areas clearly (e.g. “specialises in fire-rated shutters and garage doors”). Only include named individuals under “Key people” if they hold relevant commercial or technical roles.

The output must be in the specified one-line format.

If no specific information about a company could be found, just return the message, "No information found".

"""

# ======================== Check Contact listing for valid website =================================

URL_RE = re.compile(r'^(https?://)?([a-z0-9-]+\.)+[a-z]{2,}', re.I)
def is_valid_url(url: str) -> bool:
    return bool(url and URL_RE.match(url.strip()))

# ========= extract a safe message from the AI response ===========

def safe_extract_openai_message(rsp, fallback="ERROR: empty"):
    if getattr(rsp, "output", None):
        for item in rsp.output:
            for piece in getattr(item, "content", []):
                if getattr(piece, "text", None):
                    return piece.text.strip()
    if getattr(rsp, "choices", None):

        txt = rsp.choices[0].message.content
        if txt:
            return txt.strip()
    return fallback

# ============= openai api function def =============

def search_tool_overview(company, website, ctx="high"):
    rsp = client.responses.create(
        model=MODEL,
        tools=[{
            "type": "web_search_preview",
            "search_context_size": ctx,
            "user_location": {"type": "approximate", "country": "GB"}
        }],
        instructions=instructions_forSearchTool,
        temperature=TEMP,
        input=f"Company name: {company}\nCompany website: {website or 'N/A'}"
    )
    return safe_extract_openai_message(rsp)

# ========== loading / creating master ============

if not os.path.exists(MASTER_CSV):
    print(f"Creating master file from original: {ORIGINAL_CSV}")
    df = pd.read_csv(ORIGINAL_CSV)
    df["CompanyOverviewUpdate"] = ""
    df.to_csv(MASTER_CSV, index=False)
else:
    print(f"Using existing master file: {MASTER_CSV}")
    df = pd.read_csv(MASTER_CSV)
    if "CompanyOverviewUpdate" not in df.columns:
        df["CompanyOverviewUpdate"] = ""

df["Status"] = df["Status"].astype(str).str.strip()

status_keep = {
    "House Account (Keep)", "House Account (Grow)",
    "House Account", "Target Account", "Competitor", "Lead"
}
print("Total rows:", len(df))

print("Status match:", df["Status"].isin(status_keep).sum())

print("Update blank:", (df["CompanyOverviewUpdate"].isna() | (df["CompanyOverviewUpdate"].astype(str).str.strip() == "")).sum())

print('No "chatgpt" in overview:',
      (~df["CompanyOverview"].astype(str).str.contains("chatgpt", case=False, na=False)).sum())

print("Short overview (<400):",
      (df["CompanyOverview"].astype(str).str.len() < MaxCharCount).sum())

#filters to apply to potential contacts

mask = (
    df["Status"].isin(status_keep) &
    (df["CompanyOverviewUpdate"].isna() | (df["CompanyOverviewUpdate"].astype(str).str.strip() == "")) &
    (~df["CompanyOverview"].astype(str).str.contains("chatgpt", case=False, na=False)) &
    (df["CompanyOverview"].astype(str).str.len() < 400)
)


to_process = df[mask]
print(f"{len(to_process)} rows still need summaries")

# =========== loop to handle current batch ============

processed_idx = []
for idx, row in to_process.iterrows():
    if len(processed_idx) >= MAX_ROWS:
        print(f"\nStopped after {MAX_ROWS} rows for review.")
        break

    name = row["CompanyName"]
    site_raw = str(row.get("Website") or "")
    site     = site_raw if is_valid_url(site_raw) else ""
    try:
        overview_text = search_tool_overview(name, site, CONTEXT_SIZE)
    except Exception as e:
        overview_text = f"ERROR: {e}"
        print(f"{name} ({row.get('Id')}): {e}")

    date_tag = datetime.today().strftime("%d/%m/%Y")
    final_text = f'"{overview_text}"' + f'– ChatGPT {date_tag}'

    df.at[idx, "CompanyOverviewUpdate"] = final_text
    processed_idx.append(idx)

    print(f"✓ {name} ({row.get('Id')}): {overview_text[:60]}...")
    time.sleep(SLEEP_SECONDS)

  # ============ saving results =============
if processed_idx:

       #save to master and the batch, for safety
       df.to_csv(MASTER_CSV, index=False)
       batch_df = df.loc[processed_idx, ["Id", "CompanyName", "CompanyOverviewUpdate"]]
       batch_df.to_csv(BATCH_LOG, index=False)
       print(f"\nBatch of {len(processed_idx)} rows written to {BATCH_LOG}")
else:
       print("\nNo rows processed in this run.")
