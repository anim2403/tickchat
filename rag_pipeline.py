import json
import os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from pinecone import Pinecone, ServerlessSpec

class AtlanRAGPipeline:
    def debug_index_status(self):
        """Debug method to check index status and sample data"""
        try:
            stats = self.index.describe_index_stats()
            print(f"📊 Index Stats:")
            print(f"  - Total vectors: {stats.get('total_vector_count', 0)}")
            print(f"  - Index fullness: {stats.get('index_fullness', 0)}")
            print(f"  - Dimension: {stats.get('dimension', 0)}")
            
            index_dimension = stats.get('dimension', 384)
            model_dimension = self.embedder.get_sentence_embedding_dimension()
            
            print(f"🔍 Embedding Check:")
            print(f"  - Model dimension: {model_dimension}")
            print(f"  - Index dimension: {index_dimension}")
            
            if model_dimension != index_dimension:
                print(f"⚠️ WARNING: Dimension mismatch! Model: {model_dimension}, Index: {index_dimension}")
                print(f"   This could cause poor search results. Consider using a model with {index_dimension} dimensions.")
            
            # Try a simple query to check if data exists
            test_vector = [0.1] * index_dimension  # Use actual index dimension
            
            test_results = self.index.query(
                vector=test_vector,
                top_k=3,
                include_metadata=True,
                namespace="atlan",
            )
            
            print(f"🧪 Test query returned {len(test_results.matches)} results")
            
            for i, match in enumerate(test_results.matches[:2]):
                print(f"  Sample {i+1}:")
                print(f"    Score: {match.score}")
                print(f"    Metadata keys: {list(match.metadata.keys()) if match.metadata else 'None'}")
                if match.metadata and match.metadata.get('text'):
                    text_preview = match.metadata['text'][:100].replace('\n', ' ')
                    print(f"    Text preview: {text_preview}...")
                if match.metadata and match.metadata.get('url'):
                    print(f"    URL: {match.metadata['url']}")
                    
        except Exception as e:
            print(f"❌ Error checking index status: {e}")
            import traceback
            print(f"📋 Full error trace: {traceback.format_exc()}")

    def __init__(self, pinecone_api_key: str = None, groq_api_key: str = None, 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the RAG pipeline with Pinecone vector database.
        
        Args:
            pinecone_api_key: Pinecone API key (will try to get from env if not provided)
            groq_api_key: Groq API key (will try to get from env if not provided)
            embedding_model: The sentence transformer model to use for embeddings
        """
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key or os.getenv("PINECONE_API_KEY"))
        self.index_name = "atlan-docs"
        
        # Initialize Groq client
        self.groq_client = Groq(api_key=groq_api_key or os.getenv("GROQ_API_KEY"))
        
        # Initialize embedder
        self.embedding_model = embedding_model
        self.embedder = SentenceTransformer(embedding_model)
        
        # Connect to existing index
        try:
            self.index = self.pc.Index(self.index_name)
            print(f"✅ Connected to Pinecone index: {self.index_name}")
            
            # Get index stats
            stats = self.index.describe_index_stats()
            print(f"✅ RAG Pipeline initialized with {stats['total_vector_count']} documents")
            
            # Debug index status
            self.debug_index_status()
            
        except Exception as e:
            print(f"❌ Error connecting to Pinecone index '{self.index_name}': {e}")
            print("Make sure the index exists and your API key is correct.")
            raise
    
    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents based on the query using Pinecone.
        
        Args:
            query: User query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents with scores and metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
            query_embedding = query_embedding[0].tolist()  # Convert to list for Pinecone
            
            # Search in Pinecone index
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace="atlan",
                include_metadata=True,
                include_values=False
            )
            
            # Process results
            results = []
            for match in search_results.matches:
                results.append({
                    "score": float(match.score),
                    "text": match.metadata.get("text", ""),
                    "url": match.metadata.get("url", ""),
                    "chunk": match.metadata.get("chunk", "")
                })
            
            return results
            
        except Exception as e:
            print(f"❌ Error retrieving documents: {e}")
            return []
    
    def should_use_rag(self, topic_tags: List[str]) -> bool:
        """
        Determine if RAG should be used based on topic tags.
        
        Args:
            topic_tags: List of topic tags from ticket classification
            
        Returns:
            Boolean indicating whether to use RAG
        """
        rag_topics = {"how-to", "product", "best practices", "api/sdk", "sso", "api", "sdk"}
        return any(tag.lower() in rag_topics for tag in topic_tags)
    
    def generate_answer(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Generate an answer using Groq LLM with retrieved context.
        
        Args:
            query: Original user query/ticket
            context_docs: Retrieved relevant documents
            
        Returns:
            Generated answer
        """
        # Prepare context from retrieved documents
        context_texts = []
        sources = []
        
        for i, doc in enumerate(context_docs):
            if doc.get("text"):  # Only add documents with text content
                context_texts.append(f"Document {i+1}:\n{doc['text']}")
                if doc.get("url"):
                    sources.append(f"- [{doc['url']}]({doc['url']})")
        
        if not context_texts:
            return "No relevant documentation found to answer your question."
        
        context = "\n\n".join(context_texts)
        sources_text = "\n".join(sources) if sources else "No sources available"
        
        # Create prompt
        system_prompt = """You are an expert Atlan support agent helping users with their questions about Atlan's data catalog platform. 

Use the provided documentation context to answer the user's question accurately and helpfully. Follow these guidelines:

1. Provide clear, step-by-step instructions when applicable
2. Include relevant code examples if available in the context
3. Reference specific Atlan features and capabilities
4. If the context doesn't fully answer the question, acknowledge this limitation
5. Keep your response focused and practical
6. Format your response with proper headings and bullet points for readability

Always base your answer primarily on the provided context from Atlan's official documentation."""

        user_prompt = f"""Question/Ticket: {query}

Context from Atlan Documentation:
{context}

Please provide a comprehensive answer based on the documentation context above."""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            # Add sources to the answer
            answer_with_sources = f"{answer}\n\n**Sources:**\n{sources_text}"
            
            return answer_with_sources
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def get_ai_help(self, ticket_subject: str, ticket_body: str, topic_tags: List[str]) -> Dict[str, Any]:
        """
        Main method to get AI help for a ticket.
        
        Args:
            ticket_subject: Subject of the ticket
            ticket_body: Body/description of the ticket
            topic_tags: Topic tags from classification
            
        Returns:
            Dictionary containing answer, sources, and metadata
        """
        # Check if we should use RAG
        if not self.should_use_rag(topic_tags):
            return {
                "use_rag": False,
                "message": f"This ticket has been classified with topics: {', '.join(topic_tags)}. It has been routed to the appropriate team for assistance.",
                "topic_tags": topic_tags
            }
        
        # Combine subject and body for better context
        full_query = f"Subject: {ticket_subject}\n\nDescription: {ticket_body}"
        
        # Retrieve relevant documents
        relevant_docs = self.retrieve_documents(full_query, top_k=5)
        
        if not relevant_docs:
            return {
                "use_rag": True,
                "error": "No relevant documentation found for this query.",
                "topic_tags": topic_tags
            }
        
        # Generate answer
        answer = self.generate_answer(full_query, relevant_docs)
        
        return {
            "use_rag": True,
            "answer": answer,
            "relevant_docs": relevant_docs,
            "topic_tags": topic_tags,
            "num_sources": len(relevant_docs)
        }

# Example usage and testing function
def test_rag_pipeline():
    """Test the RAG pipeline with some sample queries"""
    try:
        rag = AtlanRAGPipeline()
        
        test_cases = [
            {
                "subject": "How to create a new data asset",
                "body": "I need help creating a new data asset in Atlan. What are the steps?",
                "topic_tags": ["how-to", "product"]
            },
            {
                "subject": "API authentication issue",
                "body": "I'm having trouble authenticating with the Atlan API. Can you help?",
                "topic_tags": ["api", "authentication"]
            },
            {
                "subject": "Connector not working",
                "body": "My Snowflake connector is not syncing properly",
                "topic_tags": ["connector", "technical"]
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            print(f"\n--- Test Case {i+1} ---")
            print(f"Subject: {test_case['subject']}")
            print(f"Topic Tags: {test_case['topic_tags']}")
            
            result = rag.get_ai_help(
                test_case["subject"],
                test_case["body"],
                test_case["topic_tags"]
            )
            
            if result["use_rag"]:
                if "answer" in result:
                    print(f"AI Answer: {result['answer'][:200]}...")
                    print(f"Number of sources used: {result['num_sources']}")
                else:
                    print(f"Error: {result.get('error', 'Unknown error')}")
            else:
                print(f"Message: {result['message']}")
                
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    # Test the pipeline
    test_rag_pipeline()
