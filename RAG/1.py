from langchain_community.document_loaders import JSONLoader
import json
from pathlib import Path

file_path = "RAG/files/activity.json"
data = json.loads(Path(file_path).read_text())


def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["title"] = record.get("title")
    return metadata


def process_content(content):
    return json.loads(content)


loader = JSONLoader(
    file_path=file_path,
    jq_schema=".[] | {title: .title, content: .content}",
    content_key="content",
    text_content=False,
    metadata_func=metadata_func,
)
docs = loader.load()

for doc in docs:
    doc.page_content = process_content(doc.page_content)
    print(doc)

# https://python.langchain.com/v0.2/docs/integrations/document_loaders/json/
