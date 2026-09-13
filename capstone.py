import sys
from config import DOCUMENTS_DIR
from loader import get_document_chunks
from vectorstore import get_vectorstore, get_retriever
from chain import create_rag_chain


def main():
    print("=" * 60)
    print("  Love2D Documentation RAG Assistant")
    print("=" * 60)

    try:
        # 1. Load or build vector store
        # If a cached FAISS index exists on disk, load it directly.
        # Otherwise, load & split PDFs and build a new vector store.
        try:
            vs = get_vectorstore()
        except ValueError:
            chunks = get_document_chunks(DOCUMENTS_DIR)
            vs = get_vectorstore(chunks)

        # 2. Setup retriever
        retriever = get_retriever(vs)

        # 3. Create RAG chain
        print("Initializing RAG chain...")
        rag_chain = create_rag_chain(retriever)
        print("System is ready!\n")

        # 4. Query interaction loop
        while True:
            try:
                query = input("Please enter your query (or type 'exit' to quit): \n\n").strip()
                if not query:
                    continue
                if query.lower() in {"exit", "quit", "q"}:
                    print("Goodbye!")
                    break

                print("\nSearching and generating response...")
                response = rag_chain.invoke(query)

                print("\n" + "-" * 50)
                if response and response.answers:
                    for i, ans in enumerate(response.answers):
                        ans_text = ans.get("text", "")
                        print(ans_text)
                        ctx = ans.get("context", "").strip()
                        negative_cues = (
                            "does not contain",
                            "not contain information",
                            "does not provide",
                            "not mentioned",
                            "not enough information",
                            "cannot be answered",
                            "not found in the documentation",
                            "no information",
                            "out of scope",
                        )
                        is_out_of_context = any(cue in ans_text.lower() for cue in negative_cues)
                        if ctx and not is_out_of_context and ctx not in ans_text:
                            print("\n[Supporting Code / Reference]:\n" + ctx)
                else:
                    print("No answer returned.")
                print("-" * 50 + "\n")

            except (KeyboardInterrupt, EOFError):
                print("\nExiting. Goodbye!")
                break
            except Exception as e:
                print(f"\nError processing query: {e}\n")

    except Exception as e:
        print(f"\nInitialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()