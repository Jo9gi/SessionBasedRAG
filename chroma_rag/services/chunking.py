import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =====================================================
# CHAPTER DETECTION
# =====================================================

CHAPTER_PATTERN = re.compile(
    r"(CHAPTER\s+\d+)",
    re.IGNORECASE
)


# =====================================================
# SPLIT DOCUMENT INTO CHAPTERS
# =====================================================

def split_by_chapters(text):

    matches = list(
        CHAPTER_PATTERN.finditer(text)
    )

    if not matches:
        return [text]

    chapters = []

    for i, match in enumerate(matches):

        start = match.start()

        end = (matches[i + 1].start() if i + 1 < len(matches) else len(text))

        chapters.append(text[start:end])

    return chapters


# =====================================================
# EXTRACT CHAPTER TITLE
# =====================================================

def get_chapter_title(chapter_text):

    lines = [
        line.strip()
        for line in chapter_text.splitlines()
        if line.strip()
    ]

    if len(lines) > 1:
        return lines[1]

    return "General Content"


# =====================================================
# MAIN CHUNKING FUNCTION
# =====================================================

def chapter_recursive_chunking(documents,chunk_size=1200,chunk_overlap=150):

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    final_chunks = []

    for doc in documents:

        file_name = doc.metadata.get("file_name","unknown")
        source = doc.metadata.get("source","unknown")
        page = doc.metadata.get("page",0)

        chapter_blocks = split_by_chapters(doc.page_content)

        for chapter_text in chapter_blocks:

            chapter_title = get_chapter_title(
                chapter_text
            )

            chapter_doc = Document(
                page_content=chapter_text,
                metadata={
                    **doc.metadata,
                    "file_name": file_name,
                    "source": source,
                    "page": page,
                    "chapter": chapter_title,
                }
            )

            chunks = splitter.split_documents(
                [chapter_doc]
            )

            final_chunks.extend(chunks)

    return final_chunks