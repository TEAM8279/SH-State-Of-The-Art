import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel
import torch
import PyPDF2
import pickle


@dataclass
class ResearchPaper:
    ref: str
    content: str


class ResearchRAGSystem:
    def __init__(
        self,
        embedding_model_name: str = "allenai/specter",
        embeddings_path: str = "embeddings.pkl",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.model = AutoModel.from_pretrained(embedding_model_name)
        self.papers: List[ResearchPaper] = []
        self.embeddings: np.ndarray = None
        self.embeddings_path = embeddings_path
        self._load_embeddings()

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
            return text
        except PyPDF2.errors.PdfReadError:
            print(f"Error reading PDF: {pdf_path}. Skipping...")
            return ""

    def parse_paper(self, pdf_path: str, text: str) -> ResearchPaper:
        return ResearchPaper(
            ref=pdf_path,
            content=text,
        )

    def add_paper(self, pdf_path: str):
        text = self.extract_text_from_pdf(pdf_path)
        if text:
            paper = self.parse_paper(pdf_path, text)
            self.papers.append(paper)
            self._update_embeddings()

    def _create_embedding(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            text, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def _update_embeddings(self):
        self.embeddings = np.vstack(
            [
                self._create_embedding(f"{paper.ref} {paper.content}")
                for paper in self.papers
            ]
        )
        self._save_embeddings()

    def _save_embeddings(self):
        with open(self.embeddings_path, "wb") as file:
            pickle.dump(self.embeddings, file)

    def _load_embeddings(self):
        try:
            with open(self.embeddings_path, "rb") as file:
                self.embeddings = pickle.load(file)
        except FileNotFoundError:
            self.embeddings = None

    def find_similar_papers(
        self, query: str, n: int = 10
    ) -> List[Tuple[ResearchPaper, float]]:
        query_embedding = self._create_embedding(query)
        similarities = np.dot(self.embeddings, query_embedding.T)
        top_indices = np.argsort(similarities.flatten())[-n:][::-1]

        return [(self.papers[i], float(similarities[i])) for i in top_indices]

    def generate_llm_prompt(self, query: str, n_papers: int = 10) -> str:
        similar_papers = self.find_similar_papers(query, n_papers)

        prompt = f"""Based on the following research papers, provide a comprehensive state-of-the-art summary 
regarding: {query}

Analyze these papers and their findings:

"""
        # Add each paper's details to the prompt
        for i, (paper, similarity) in enumerate(similar_papers, 1):
            prompt += f"""
Paper {i}:
Ref: {paper.ref}
Relevance Score: {similarity:.2f}

Content:
{paper.content}

---
"""

        prompt += """
Please provide:
1. A comprehensive summary of the current state of the art based on these papers
2. Key findings and methodologies
3. Common themes and potential research gaps
4. Citations for each major point

Format the response with:
- Clear sections for each major topic
- Bullet points for key findings
- A "References" section at the end listing all papers used, numbered according to their appearance in the summary.
"""

        return prompt
