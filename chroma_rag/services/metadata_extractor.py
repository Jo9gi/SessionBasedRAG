import re

def extract_headings(documents):

    headings = []

    chapter_pattern = re.compile(r"^CHAPTER\s+\d+", re.IGNORECASE)

    toc_pattern = re.compile(r"^\d+\.\s+[A-Z].+")

    for doc in documents:

        lines = doc.page_content.splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if chapter_pattern.match(line):
                headings.append({"text": line})

            elif toc_pattern.match(line):
                if len(line) < 120:
                    headings.append({"text": line})

    return headings