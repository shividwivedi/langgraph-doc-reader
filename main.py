import os
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from langraph_workflow import DocumentIntelligenceWorkflow


class DocumentIntelligenceApp:
    """
    Main application class that ties everything together
    """

    def __init__(self):
        load_dotenv()

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("Please set OPENAI_API_KEY in your .env file")

        # FIXED: Pass openai_api_key to DocumentProcessor
        self.processor = DocumentProcessor(self.openai_api_key)
        self.workflow = None
        self.is_initialized = False

    def setup_documents(self, documents_folder: str = "documents"):
        """
        Process all PDF documents and create the searchable database
        Args:
            documents_folder: Folder containing your PDF files
        """
        if not os.path.exists(documents_folder):
            print(f"Error: '{documents_folder}' folder not found!")
            return False

        pdf_files = [f for f in os.listdir(documents_folder) if f.endswith(".pdf")]
        if not pdf_files:
            print(f"Error: No PDF files found in '{documents_folder}' folder!")
            return False

        self.processor.process_all_documents(documents_folder)

        # FIXED: Typo - vectorstore not vectostore
        vectorstore = self.processor.get_vectorstore()

        if vectorstore:
            self.workflow = DocumentIntelligenceWorkflow(vectorstore, self.openai_api_key)
            self.is_initialized = True
            print("System Initialized Successfully")
            return True
        else:
            print("Failed to initialize vector database")
            return False

    def ask_question(self, question: str) -> dict:
        """
        Ask a question about your documents
        Args:
            question: Your question about the documents
        Returns:
            Dictionary with answer and metadata
        """
        if not self.is_initialized:
            return {
                "error": "System not initialized. Please run setup_documents() first."
            }

        return self.workflow.process_question(question)

    def interactive_mode(self):
        """
        Start interactive question-answering session
        """
        if not self.is_initialized:
            print("System not initialized. Please run setup first.")
            return


        while True:
            try:
                question = input("\nYour question: ").strip()

                if question.lower() in ['quit', 'exit', 'q']:
                    break


                if not question:
                    print("Please enter a question.")
                    continue

                # Process the question
                result = self.ask_question(question)

                # Display results
                self.display_result(result)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {str(e)}")


    def display_result(self, result: dict):
        """Display the answer in a formatted way"""
        print("\n" + "-" * 60)
        print("ANSWER:")
        print("-" * 60)
        print(result["answer"])

        print(f"\nConfidence: {result['confidence']}")
        print(f"Sources Used: {result['num_sources']} document chunks")
        print(f"Files Referenced: {', '.join(result['source_files'])}")
        print("-" * 60)

    def show_sample_questions(self):
        """Show sample questions users can ask"""
        print("\nSAMPLE QUESTIONS:")
        print("• What are the main topics covered in these documents?")
        print("• Summarize the key findings from the research papers")
        print("• What recommendations are mentioned across the documents?")
        print("• Are there any conflicting viewpoints in the documents?")
        print("• What are the most important dates or deadlines mentioned?")
        print("• Who are the key people or organizations referenced?")
        print("• What are the main conclusions or takeaways?")


def main():
    """Main function to run the application"""

    # Create the app
    app = DocumentIntelligenceApp()

    # Setup documents (this will process all PDFs in the 'documents' folder)
    success = app.setup_documents("documents")

    if success:
        # Start interactive mode
        app.interactive_mode()
    else:
        print("\nSetup failed. Please check your documents folder and try again.")
        print("\nMake sure you:")
        print("1. Created a 'documents' folder in your project directory")
        print("2. Added your PDF files to the 'documents' folder")
        print("3. Set your OPENAI_API_KEY in the .env file")


if __name__ == "__main__":
    main()