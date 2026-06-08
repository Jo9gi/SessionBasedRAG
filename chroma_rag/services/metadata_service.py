import json
from chroma_rag.models import (SessionDocument,DocumentMetadata)

def get_document_count(session):
    return SessionDocument.objects.filter(session=session).count()


def get_document_names(session):
    return list(SessionDocument.objects.filter(session=session).values_list("document_name",flat=True))


def get_chapter_names(session):
    result = []
    metadata_rows = DocumentMetadata.objects.filter(session=session)

    for row in metadata_rows:
        try:
            headings = json.loads(row.headings)
            result.extend(headings)
        
        except:
            pass

    return result