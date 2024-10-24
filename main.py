# Import Langchain modules
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Other modules and packages
import tempfile
import streamlit as st
from functions import *

# Create prompt template
PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks.
You will be provided some documents, specifically bank statements.
You are expected to summarize and organize the transactions information based on the question that will be asked. Your primary goal will be to provide insights into the transactions in the bank statement. As such, it is absolutely crucial that you do not miss out any rows of transactions in the bank statement, and be absolutely accurate. This means that you should not create additional transactions that do not exist.
When asked to display or organize transactions in a table format, always organize it in ascending order by the date. There can be more than 1 of the same transaction on the same day, do not miss them from the output.
The transactions may also be split up into multiple pages or chunks, be sure to include all of them if relevant.
Use the following pieces of retrieved context to answer the question.


{context}

---

Answer the question based on the above context: {question}
"""
#Can you organize all the transactions in this bank statement from {question}, in a table format. Can you also sum up the total amount and show your working.

# Create prompt
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)





##### ========== Streamlit app setup ========== #####

st.set_page_config(page_title='🦜🔗 Bank Statement Tracker App')
st.title('🦜🔗 Bank Statement Tracker App')


# Initialize session state for template selection and query text
if "selected_template" not in st.session_state:
    st.session_state.selected_template = ""
if "query_text" not in st.session_state:
    st.session_state.query_text = ""

# Define available template questions
template_options = [
    "----- Default: Leave blank -----",  # Blank option
    "Can you organize all the transactions in this bank statement from <insert transaction name here>, in a table format.",
    "Can you extract and compile a list of all the transactions in this bank statement in a table format. The transactions that I want you to extract should contain either the EXACT word '<insert word1 here>' or '<insert word2 here>'. For example, this means that if I want '<insert word1 here>', only transactions that contain this word should be extracted, DO NOT ALTER THE NAMES OF THE TRANSACTIONS. Please also indicate the full transaction description, as well as the type of each transaction whether it is incoming or outgoing."
]

# Update session state when the user selects a template question
def update_template():
    st.session_state.selected_template = st.session_state.template_selectbox


# Template question dropdown (outside the form to update session state)
st.header('Ask a Question')
selected_template = st.selectbox("Template questions", options=template_options)


with st.form('myform'):
    # Get OpenAI API Key
    st.header("Input your OpenAI API Key")
    OPENAI_API_KEY = st.text_input("OpenAI API Key", type='password', disabled=False)

    # File upload (allow PDFs)
    st.header("Upload file")
    uploaded_file = st.file_uploader('Upload a PDF', type='pdf')

    # Handle clearing the text area when "Blank" is selected
    if selected_template == template_options[0]:
        st.session_state.query_text = ""
    elif selected_template and not st.session_state.query_text:
        # Set query text based on template if the user hasn't manually entered text
        st.session_state.query_text = selected_template

    # Query text area (inside the form, updated based on session state)
    query_text = st.text_area('Enter your question:', 
                              value=st.session_state.query_text, 
                              placeholder='Please provide a short summary.', 
                              height=200)

    # Form submit button
    submitted = st.form_submit_button("Submit", disabled=False)


if submitted and uploaded_file is not None:
    with st.spinner('Processing...'):
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name  # Get the path of the temp file

        # Process PDF
        loader = PyPDFLoader(temp_file_path)
        pages = loader.load()

        # Split document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, 
                                                       chunk_overlap=200,
                                                       length_function=len,
                                                       separators=["\n\n", "\n", " "])
        chunks = text_splitter.split_documents(pages)

        # Create embeddings
        embedding_function = get_embedding_function(OPENAI_API_KEY)

        # Create a temporary directory for the vectorstore
        temp_vectorstore_dir = tempfile.mkdtemp()
        try:
            # Create vectorstore in the temporary directory
            vectorstore = create_vectorstore(chunks=chunks,
                                             embedding_function=embedding_function,
                                             vectorstore_path=temp_vectorstore_dir)

            all_chunks = vectorstore._collection.get(include=["documents"])
            documents = all_chunks['documents']  # This will give you all the stored documents (chunks)

            # Concatenate context text
            context_text = "\n\n---\n\n".join([doc for doc in documents])

            # Create prompt
            prompt = prompt_template.format(context=context_text, question=query_text)
            response = llm.invoke(prompt)
            content = response.content

            # Show result
            if content:
                st.write(content)

        finally:
            # # Ensure the vectorstore is cleaned up properly
            # vectorstore._client._system.stop()
            # #SharedSystemClient.pop(vectorstore._client._identifier, None)
            # vectorstore = None

            shutil.rmtree(temp_vectorstore_dir)  # Clean up the temporary directory
            