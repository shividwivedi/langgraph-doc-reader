import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document


class WorkflowState(TypedDict):
    """
    Defines the state that flows through our workflow
    Each step can read and modify this state
    """
    question: str
    retrieved_docs: List[Document]  # FIXED: Consistent naming
    answer: str
    confidence: str
    source_files: List[str]


class DocumentIntelligenceWorkflow:
    """
    LangGraph workflow for processing document queries
    This creates a multi-step process: retrieve -> analyze -> answer
    """

    def __init__(self, vectorstore, openai_api_key: str):
        self.vectorstore = vectorstore
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            openai_api_key=openai_api_key  # FIXED: Pass the API key
        )
        # FIXED: Add parentheses to call the method
        self.workflow = self._create_workflow()

    def _create_workflow(self):
        """Create the LangGraph workflow with multiple steps"""

        workflow = StateGraph(WorkflowState)

        workflow.add_node("retrieve_documents", self.retrieve_documents)
        workflow.add_node("analyze_relevance", self.analyze_relevance)
        workflow.add_node("generate_answer", self.generate_answer)

        workflow.set_entry_point("retrieve_documents")
        workflow.add_edge("retrieve_documents", "analyze_relevance")
        workflow.add_edge("analyze_relevance", "generate_answer")
        workflow.add_edge("generate_answer", END)

        return workflow.compile()

    def retrieve_documents(self, state: WorkflowState) -> WorkflowState:
        """
        Step 1: Find relevant documents based on the question
        """
        # FIXED: Typo - similarity_search not similarity_serach
        relevant_docs = self.vectorstore.similarity_search(
            state["question"],
            k=4
        )
        # FIXED: Use consistent key name
        state["retrieved_docs"] = relevant_docs
        return state

    def analyze_relevance(self, state: WorkflowState) -> WorkflowState:
        """
        Step 2: Analyze how relevant the found documents are
        and extract source information
        """
        # FIXED: Use consistent key name
        docs = state["retrieved_docs"]

        source_files = list(set([
            doc.metadata.get("source", "Unknown")  # FIXED: Capitalization
            for doc in docs
        ]))

        # Determine confidence based on number of relevant docs
        if len(docs) >= 3:
            confidence = "High - Found multiple relevant sources"
        elif len(docs) >= 1:
            confidence = "Medium - Found some relevant information"
        else:
            confidence = "Low - Limited relevant information found"

        state["confidence"] = confidence
        state["source_files"] = source_files

        return state

    def generate_answer(self, state: WorkflowState) -> WorkflowState:
        """
        Step 3: Generate a comprehensive answer using the retrieved documents
        """
        context = "\n\n".join([
            f"From {doc.metadata.get('source', 'Unknown')}:\n{doc.page_content}"
            for doc in state["retrieved_docs"]  # FIXED: Use consistent key name
        ])

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert document analyst. Use the provided document context to answer the user's question comprehensively.

Guidelines:
- Answer based only on the information in the provided documents
- If the documents don't contain enough information, say so clearly
- Cite which documents you're referencing
- Provide specific details and examples when available
- If there are conflicting information, mention it"""),

            ("human", """Context from documents:
{context}

Question: {question}

Please provide a detailed answer based on the document context.""")
        ])

        prompt = prompt_template.format(
            context=context,
            question=state["question"]
        )

        response = self.llm.invoke(prompt)
        answer = response.content

        state["answer"] = answer
        return state

    def process_question(self, question: str) -> dict:
        """
        Main method to process a question through the entire workflow
        Args:
            question: User's question about the documents
        Returns:
            Dictionary with answer, confidence, and sources
        """

        initial_state = {
            "question": question,
            "retrieved_docs": [],  # FIXED: Use consistent key name
            "answer": "",
            "confidence": "",
            "source_files": []
        }

        final_state = self.workflow.invoke(initial_state)

        return {
            "question": question,
            "answer": final_state["answer"],
            "confidence": final_state["confidence"],
            "source_files": final_state["source_files"],
            "num_sources": len(final_state["retrieved_docs"])  # FIXED: Use consistent key name
        }