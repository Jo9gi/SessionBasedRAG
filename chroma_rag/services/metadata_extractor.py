import re
# from .document_loader import load_document

def extract_headings(documents):
    headings = []

    pattern = re.compile(r"^(?:CHAPTER\s+\d+|[0-9]+\.[0-9]*\s+.+|[A-Z][A-Z\s]{5,})$", re.MULTILINE)

    for doc in documents:
        matches = pattern.findall(doc.page_content)
        for match in matches:

            heading = match.strip()

            if heading and heading not in headings:

                headings.append(
                    {
                        "text": heading
                    }
                )

    return headings

