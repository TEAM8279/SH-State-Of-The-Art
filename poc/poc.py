from ResearchRAGSystem import ResearchRAGSystem
import os


DATA_PATH = "../data/"
ARCHIVE_PATH = DATA_PATH + "arxiv-metadata-oai-snapshot.json"
PDF_PATH = DATA_PATH + "pdfs/"

rag_system = ResearchRAGSystem()

max = 100
count = 0

for paper in os.listdir(PDF_PATH)[:max]:
    if not paper.startswith("."):
        rag_system.add_paper(PDF_PATH + paper)
        print(f"Added paper ({count}): {paper}")
        count += 1


query = "What are the latest advances in SVM?"
prompt = rag_system.generate_llm_prompt(query)

print("-----------------------------------------------------")
print("Generated prompt:")
print("-----------------------------------------------------")
print(prompt)
