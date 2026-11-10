import logging
from pathlib import Path

import requests

from ingestion.fetch_filings import USER_AGENT

logger = logging.getLogger(__name__)

RAW_PDF_DIR = Path("data/raw_pdf")

# Apple's own investor-relations-hosted PDF rendition of its latest 10-K.
# SEC EDGAR does not natively serve 10-Ks as PDF (HTML/iXBRL only), so we use
# the company-published copy instead of converting SEC HTML ourselves.
PDF_URLS = {
    "AAPL": "https://d18rn0p25nwr6d.cloudfront.net/CIK-0000320193/c87043b9-5d89-4717-9f49-c4f9663d0061.pdf",
}


def fetch_filing_pdf(ticker: str, force: bool = False) -> Path:
    RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_PDF_DIR / f"{ticker}.pdf"
    if out_path.exists() and not force:
        logger.info("Using cached PDF for %s at %s", ticker, out_path)
        return out_path

    logger.info("Fetching PDF for %s from %s", ticker, PDF_URLS[ticker])
    resp = requests.get(PDF_URLS[ticker], headers={"User-Agent": USER_AGENT}, timeout=60)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    logger.info("Saved %s (%d bytes) to %s", ticker, len(resp.content), out_path)
    return out_path


def fetch_all_filing_pdfs(force: bool = False) -> None:
    for ticker in PDF_URLS:
        fetch_filing_pdf(ticker, force=force)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    fetch_all_filing_pdfs()
