import os
import time
import json
import concurrent.futures
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate

UPLOAD_DIR = "uploads"
CHAT_HISTORY_FILE = "chat_history.json"


def save_uploaded_file(uploaded_file):
    """Save an uploaded file to the specified directory."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def list_files():
    """List all PDF files in the upload directory."""
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
    return [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".pdf")]


def delete_file(file_name):
    """Delete a file from the upload directory."""
    file_path = os.path.join(UPLOAD_DIR, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False


@st.cache_resource
def process_all_pdfs():
    """Combine text from all PDFs in the upload directory."""
    documents = []
    files = list_files()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_single_pdf, os.path.join(UPLOAD_DIR, file_name), file_name) for file_name in files]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                documents.extend(result)
    return documents


def process_single_pdf(file_path, file_name):
    """Process a single PDF and extract text with page numbers."""
    try:
        pdf_reader = PdfReader(file_path)
        return [{"text": page.extract_text() or "", "page_num": page_num + 1, "file_name": file_name}
                for page_num, page in enumerate(pdf_reader.pages)]
    except Exception as e:
        st.error(f"Error processing {file_name}: {e}")
        return []


@st.cache_resource
def create_knowledge_base(documents):
    """Create a FAISS knowledge base from the combined text."""
    if documents:
        texts = [
            f"[Page {doc['page_num']} in {doc['file_name']}]: {doc['text']}" for doc in documents
        ]
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text("\n".join(texts))

        embeddings = OpenAIEmbeddings()
        return FAISS.from_texts(chunks, embeddings)
    return None


def typing_animation(text, speed=0.01):
    """Simulates typing animation for text display with a faster speed."""
    output = ""
    placeholder = st.empty()
    for char in text:
        output += char
        placeholder.markdown(output)
        time.sleep(speed)


def load_chat_history():
    """Load chat history from JSON file."""
    if not os.path.exists(CHAT_HISTORY_FILE):
        # If the file does not exist, create it with an empty list
        with open(CHAT_HISTORY_FILE, "w") as f:
            json.dump([], f)

    try:
        with open(CHAT_HISTORY_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        # If the file is empty or corrupted, reset it to an empty list
        with open(CHAT_HISTORY_FILE, "w") as f:
            json.dump([], f)
        return []


def save_chat_history(chat_history):
    """Save chat history to JSON file."""
    with open(CHAT_HISTORY_FILE, "w") as f:
        json.dump(chat_history, f, indent=4)


def delete_chat(index):
    """Delete a specific chat entry from the history."""
    chat_history = load_chat_history()
    if 0 <= index < len(chat_history):
        chat_history.pop(index)
        save_chat_history(chat_history)


def main():
    """Main function to run the Streamlit app."""
    load_dotenv()

    st.set_page_config(
        page_title="The Anchor Builders",
        page_icon="🌟",
        layout="wide",
    )

    st.sidebar.markdown(
        """
        <div style="text-align: center; margin-bottom: 20px;">
            <img src="https://via.placeholder.com/80" alt="The Anchor Builders Logo" style="border-radius: 50%;">
            <h2 style="color: #4a90e2; font-family: Arial, sans-serif;">The Anchor Builders</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.title("Navigation")
    st.sidebar.subheader("Manage your PDFs")
    files = list_files()
    refresh_needed = False

    if files:
        with st.sidebar.expander("Uploaded Files", expanded=True):
            for file in files:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(file)
                with col2:
                    if st.button("🗑️", key=f"delete-{file}"):
                        if delete_file(file):
                            st.sidebar.success(f"Deleted {file}")
                            refresh_needed = True
                        else:
                            st.sidebar.error(f"Failed to delete {file}")
    else:
        st.sidebar.info("No files uploaded yet.")

    pdf = st.sidebar.file_uploader("Upload a PDF", type="pdf", accept_multiple_files=True)
    if pdf:
        for uploaded_file in pdf:
            save_uploaded_file(uploaded_file)
            st.sidebar.success(f"Uploaded: {uploaded_file.name}")
        refresh_needed = True

    if refresh_needed:
        st.experimental_set_query_params(**st.experimental_get_query_params())

    # Load chat history
    chat_history = load_chat_history()

    # Track deletion request
    delete_requested = False

    st.sidebar.subheader("Chat History")
    if chat_history:
        for index, chat in enumerate(chat_history):
            with st.sidebar.expander(f"Q: {chat['question']}"):
                st.markdown(f"**A:** {chat['answer']}")
                if st.button("🗑️ Delete", key=f"delete-chat-{index}"):
                    delete_chat(index)
                    delete_requested = True
                    break  # Exit the loop to handle rerun safely
    else:
        st.sidebar.info("No chat history yet.")

    if delete_requested:
        # Trigger a rerun only after a successful delete request
        st.experimental_set_query_params(**st.experimental_get_query_params())

    st.title("📖 PDF Knowledge Assistant")
    st.markdown("Upload your PDFs, ask questions, and get precise answers with citations.")

    documents = process_all_pdfs()
    knowledge_base = create_knowledge_base(documents)

    if knowledge_base:
        st.subheader("💬 Ask Your Questions")

        user_question = st.text_input(
            "Type your question here:",
            placeholder="e.g., What are the height restrictions for ADUs?",
            key="user_input",
        )

        if user_question:
            # Retrieve the top 3 most relevant documents
            docs = knowledge_base.similarity_search(user_question, k=3)
            llm = OpenAI()

            # Define a prompt template for summarizing and breaking down details
            prompt_template = """Based on the following information, summarize the most relevant answer concisely and provide a detailed breakdown of key points. Ensure the answer is clear and directly related to the query.
            Context: {context}
            Question: {question}
            Answer:"""
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

            # Load the question answering chain using the updated prompt
            chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)

            if response.strip():
                # Append new chat to chat history
                chat_history.append({"question": user_question, "answer": response.strip()})
                
                # Save the updated chat history
                save_chat_history(chat_history)

                # Display the response
                st.markdown(f"**Your Question:** {user_question}")
                typing_animation(f"**Answer:** {response}")
                st.info(
                    """
                    **Additional Notes:**
                    - Extracted from your uploaded PDFs.
                    - Cross-reference for accuracy when needed.
                    """
                )
            else:
                st.warning("No relevant information found in the uploaded PDFs.")
    else:
        st.warning("No data available. Please upload PDF files to begin.")


if __name__ == "__main__":
    main()
