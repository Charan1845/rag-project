"""Map the many spellings of companies, cities and roles onto one canonical name."""

COMPANY_ALIASES = {
    "tcs": "TCS",
    "tata consultancy services": "TCS",
    "tata consultancy": "TCS",
    "tata consultancy services ltd": "TCS",
    "infosys ltd": "Infosys",
    "infosys limited": "Infosys",
    "wipro ltd": "Wipro",
    "infosys": "Infosys",
    "infy": "Infosys",
    "wipro": "Wipro",
    "hcl": "HCLTech",
    "hcl technologies": "HCLTech",
    "hcltech": "HCLTech",
    "cognizant": "Cognizant",
    "cts": "Cognizant",
    "accenture": "Accenture",
    "tech mahindra": "Tech Mahindra",
    "flipkart": "Flipkart",
    "amazon": "Amazon",
    "aws": "Amazon",
    "amazon web services": "Amazon",
    "google": "Google",
    "alphabet": "Google",
    "microsoft": "Microsoft",
    "msft": "Microsoft",
    "meta": "Meta",
    "facebook": "Meta",
    "apple": "Apple",
    "netflix": "Netflix",
    "uber": "Uber",
    "swiggy": "Swiggy",
    "zomato": "Zomato",
    "paytm": "Paytm",
    "byju's": "Byju's",
    "byjus": "Byju's",
    "ola": "Ola",
}

CITY_ALIASES = {
    "bangalore": "Bangalore",
    "bengaluru": "Bangalore",
    "blr": "Bangalore",
    "hyderabad": "Hyderabad",
    "hyd": "Hyderabad",
    "secunderabad": "Hyderabad",
    "pune": "Pune",
    "chennai": "Chennai",
    "madras": "Chennai",
    "mumbai": "Mumbai",
    "bombay": "Mumbai",
    "delhi": "Delhi NCR",
    "new delhi": "Delhi NCR",
    "gurgaon": "Delhi NCR",
    "gurugram": "Delhi NCR",
    "noida": "Delhi NCR",
    "ncr": "Delhi NCR",
    "kolkata": "Kolkata",
    "calcutta": "Kolkata",
    "sf": "San Francisco",
    "bay area": "San Francisco",
    "sf bay area": "San Francisco",
    "san francisco": "San Francisco",
    "silicon valley": "San Francisco",
    "sv": "San Francisco",
    "sunnyvale": "San Francisco",
    "mountain view": "San Francisco",
    "palo alto": "San Francisco",
    "nyc": "New York",
    "new york": "New York",
    "seattle": "Seattle",
    "london": "London",
}

ROLE_ALIASES = {
    "sde": "software engineer",
    "swe": "software engineer",
    "software developer": "software engineer",
    "developer": "software engineer",
    "programmer": "software engineer",
    "software engineer": "software engineer",
    "engineer": "software engineer",
    "data scientist": "data scientist",
    "ml engineer": "ml engineer",
    "machine learning engineer": "ml engineer",
    "engineering manager": "engineering manager",
    "em": "engineering manager",
    "manager": "engineering manager",
    "product manager": "product manager",
    "pm": "product manager",
    "qa": "qa engineer",
    "tester": "qa engineer",
    "devops": "devops engineer",
    "sre": "devops engineer",
}


def _norm(value: str | None, aliases: dict[str, str], title: bool = True) -> str | None:
    if not value:
        return None
    key = value.strip().lower().rstrip(".")
    if key in {"", "none", "null", "unknown", "n/a"}:
        return None
    if key in aliases:
        return aliases[key]
    return value.strip().title() if title else key


# Words that mean the model returned a category ("Indian startups") instead of an employer.
GENERIC_COMPANY_WORDS = (
    "compan", "startup", "consultanc", "faang", "maang", "big tech", "employer", "firm",
    "corporat", "mnc", "gcc", "service", "government", "agency", "industry",
)
COUNTRIES = {"india", "usa", "us", "united states", "uk", "united kingdom", "canada", "germany", "europe"}


def company(value: str | None) -> str | None:
    if value and value.strip().lower().rstrip(".") in COMPANY_ALIASES:  # "Tata Consultancy Services"
        return COMPANY_ALIASES[value.strip().lower().rstrip(".")]
    if value and any(w in value.lower() for w in GENERIC_COMPANY_WORDS):
        return None
    return _norm(value, COMPANY_ALIASES)


VAGUE_PLACES = {"various", "multiple", "remote", "anywhere", "west coast", "east coast"}


def city(value: str | None) -> str | None:
    if not value:
        return None
    value = value.split(",")[0].split("/")[0]  # "Hyderabad, Bangalore" -> first city mentioned
    if value.strip().lower() in COUNTRIES | VAGUE_PLACES:
        return None
    return _norm(value, CITY_ALIASES)


def find_in_text(text: str, aliases: dict[str, str]) -> str | None:
    """First known alias that appears as a whole word in text (longest aliases checked first)."""
    import re
    low = text.lower()
    for alias in sorted(aliases, key=len, reverse=True):
        if re.search(rf"(?<![\w']){re.escape(alias)}(?![\w'])", low):
            return aliases[alias]
    return None


COUNTRY_ALIASES = {
    "india": "India", "bharat": "India",
    "us": "United States", "usa": "United States", "united states": "United States",
    "america": "United States", "united states of america": "United States",
    "uk": "United Kingdom", "united kingdom": "United Kingdom", "england": "United Kingdom",
}


def country(value: str | None) -> str | None:
    return _norm(value, COUNTRY_ALIASES)


def mentioned(name: str, text: str) -> bool:
    """Is this company actually named in the text? Acronyms (TCS, HCL, AWS) must match case-sensitively,
    so "my 7 figure TCs" (total compensation) is not read as TCS. Other names match as a word prefix,
    so "Googler" counts for Google."""
    import re
    names = {name} | {a for a, c in COMPANY_ALIASES.items() if c == name}
    for n in names:
        if len(n) <= 4:
            # Short names only count as "TCS"/"TCS's" or "Meta"/"Infy", never "TCs" or "meta".
            if re.search(rf"\b(?:{re.escape(n.upper())}|{re.escape(n.capitalize())})\b", text):
                return True
        elif re.search(rf"\b{re.escape(n)}", text, re.I):
            return True
    return False


def record(rec: dict) -> dict:
    """Re-normalize an extracted record (safe to run repeatedly)."""
    text = f"{rec.get('title') or ''}\n{rec.get('text') or ''}"
    # Drop employers the model inferred but the post never names (e.g. "TCs" read as TCS).
    companies = sorted({c for c in map(company, rec.get("companies", [])) if c and mentioned(c, text)})
    opinions = []
    for op in rec.get("opinions", []):
        op = {**op, "company": company(op.get("company"))}
        if op["company"] and not mentioned(op["company"], text):
            op["company"] = None
        # A post about one employer: unattributed opinions are about that employer.
        if op["company"] is None and len(companies) == 1:
            op["company"] = companies[0]
        if op["company"]:
            companies = sorted(set(companies) | {op["company"]})
        opinions.append(op)
    # "Hyderabad" as the city implies India even when the model left country empty.
    the_country = country(rec.get("country"))
    if the_country is None and rec.get("city") and rec["city"].strip().lower() in ("india",):
        the_country = "India"
    the_city = city(rec.get("city"))
    if the_country is None and the_city in INDIAN_CITIES:
        the_country = "India"
    return {**rec, "companies": companies, "city": the_city, "country": the_country,
            "role": role(rec.get("role")), "opinions": opinions,
            # Some source titles already contain U+FFFD where a quote used to be; drop it.
            "title": (rec.get("title") or "").replace("�", ""),
            "text": (rec.get("text") or "").replace("�", "")}


INDIAN_CITIES = {"Bangalore", "Hyderabad", "Pune", "Chennai", "Mumbai", "Delhi NCR", "Kolkata"}


def role(value: str | None) -> str | None:
    if not value:
        return None
    key = value.strip().lower()
    for alias, canon in ROLE_ALIASES.items():
        if key == alias or key.startswith(("senior " + alias, "junior " + alias, "staff " + alias, "lead " + alias)):
            return canon
    return _norm(value, ROLE_ALIASES, title=False)
