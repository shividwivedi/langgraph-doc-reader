import os
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from typing import List

class DocumentProcessor:
    """
    This class handles loading PDFs, splitting text into chunks,
    and creating a searchable vector database
    """

    def __init__(self, openai_api_key: str):  # FIXED: Add openai_api_key parameter
        """Initialize with OpenAI API key for embeddings"""
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]  # FIXED: Uncommented separators
        )
        self.vectorstore = None

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract all text from a PDF file
        Args:
            pdf_path: Path to the PDF file
        Returns:
            Extracted text as a string
        """
        text = ""
        try:
            doc = fitz.open(pdf_path)

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
                text += f"\n--- Page {page_num + 1} ---\n"

            doc.close()

        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return ""

        return text

    def load_documents(self, documents_folder: str) -> List[Document]:
        """
        Load all PDF files from a folder and convert them to Document objects
        Args:
            documents_folder: Path to folder containing PDF files
        Returns:
            List of Document objects
        """
        documents = []

        pdf_files = [f for f in os.listdir(documents_folder) if f.endswith('.pdf')]

        for pdf_file in pdf_files:
            pdf_path = os.path.join(documents_folder, pdf_file)

            text = self.extract_text_from_pdf(pdf_path)
            if text.strip():
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": pdf_file,
                        "file_path": pdf_path
                    }
                )
                documents.append(doc)

        return documents

    def create_vector_database(self, documents: List[Document]):
        """
        Split documents into chunks and create a searchable vector database
        Args:
            documents: List of Document objects to process
        """
        all_chunks = []
        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])
            # FIXED: Use extend instead of append to flatten the list
            all_chunks.extend(chunks)

        self.vectorstore = Chroma.from_documents(
            documents=all_chunks,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )

    def get_vectorstore(self):
        """Return the vector database for querying"""
        return self.vectorstore

    def process_all_documents(self, documents_folder: str):
        """
        Complete pipeline: load PDFs -> create vector database
        Args:
            documents_folder: Path to folder containing PDF files
        """
        documents = self.load_documents(documents_folder)

        if not documents:
            print("No documents found or processed!")
            return

        self.create_vector_database(documents)