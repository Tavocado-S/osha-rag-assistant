"""
Step 1 of the pipeline: pull OSHA 29 CFR 1910 (General Industry Standards)
from the eCFR API and save it as structured JSON, preserving the
part -> subpart -> section hierarchy. This structure is what
src/chunking.py uses to build metadata-rich chunks instead of
blindly splitting raw text.

eCFR (Electronic Code of Federal Regulations) is the official, current,
public-domain source — no scraping fragile HTML, no copyright question.
Docs: https://www.ecfr.gov/developers/documentation/api/v1

Run:
    python -m src.ingest
"""
import json
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

RAW_DIR = Path("data/raw")
STRUCTURED_PATH = Path("data/osha_1910_structured.json")

ECFR_XML_URL = (
    "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-29.xml"
    "?part=1910"
)


def fetch_current_date() -> str:
    """eCFR versioner needs an as-of date; today's date returns the latest version."""
    resp = requests.get(
        "https://www.ecfr.gov/api/versioner/v1/titles.json", timeout=30
    )
    resp.raise_for_status()
    titles = resp.json()["titles"]
    title_29 = next(t for t in titles if t["number"] == 29) #Title 29 is specifically the CFR title that covers Labor
    return title_29["latest_issue_date"]


def fetch_part_1910_xml(as_of_date: str) -> str:
    url = ECFR_XML_URL.format(date=as_of_date)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return resp.text


def parse_sections(xml_text: str) -> list[dict]:
    """
    Walk the eCFR XML and pull out each <DIV8> (section-level) node,
    keeping subpart context for metadata. It gives error messages if
    the eCFR schema tag names differ slightly from what's expected here —
    check data/raw/1910_raw.xml and adjust the tag names if this returns
    an empty list.
    """
    soup = BeautifulSoup(xml_text, "xml")
    sections = []
    current_subpart = "Unknown"

    for node in soup.find_all(["DIV6", "DIV8"]):  # DIV6=subpart, DIV8=section
        if node.name == "DIV6":
            head = node.find("HEAD")
            current_subpart = head.get_text(strip=True) if head else current_subpart
            continue

        head = node.find("HEAD")
        section_id = node.get("N", "unknown")
        title = head.get_text(strip=True) if head else section_id
        body_text = node.get_text(separator=" ", strip=True)

        sections.append(
            {
                "section_id": section_id,
                "title": title,
                "subpart": current_subpart,
                "text": body_text,
                "source_url": f"https://www.osha.gov/laws-regs/regulations/standardnumber/1910/{section_id}",
            }
        )
    return sections


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("Looking up latest eCFR issue date for Title 29...")
    as_of_date = fetch_current_date()

    print(f"Fetching 29 CFR 1910 as of {as_of_date} ...")
    xml_text = fetch_part_1910_xml(as_of_date)
    (RAW_DIR / "1910_raw.xml").write_text(xml_text, encoding="utf-8")

    print("Parsing sections...")
    sections = parse_sections(xml_text)
    print(f"Parsed {len(sections)} sections.")

    if not sections:
        raise RuntimeError(
            "No sections parsed — the eCFR XML tag names may not match "
            "what this script expects. Open data/raw/1910_raw.xml and "
            "check the actual DIV-level tag names, then update parse_sections()."
        )

    STRUCTURED_PATH.write_text(
        json.dumps(sections, indent=2), encoding="utf-8"
    )
    print(f"Saved structured data to {STRUCTURED_PATH}")


if __name__ == "__main__":
    main()
