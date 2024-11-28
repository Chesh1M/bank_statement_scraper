# Import Langchain modules
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Other modules and packages
import tempfile
import streamlit as st
from functions import *
import pandas as pd
import json


## REMOVE WHEN RUNNING LOCALLY ##
# Import latest sqlite for streamlit compatibility with chromadb
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
## --------------------------- ##


# List of models available
models_avail = ["gpt-4o-mini", "o1-mini"]

# Create prompt template
PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks.
You will be provided documents, specifically bank statements.
You are expected to summarize and organize the transactions information based on the question that will be asked. Your primary goal will be to provide insights into the transactions in the bank statement. As such, it is absolutely crucial that you do not miss out any rows of transactions in the bank statement, and be extremely accurate. This means that you should not create additional transactions that do not exist.
When asked to display or organize transactions, always organize it in ascending order by their date. 

The fields of the transactions should be "date", "description", and "amount".
Dates should always be formatted in the following way - dd MMM YYYY, for example "22 SEP 2024".
Descriptions should always be formatted such that there are no spaces in the descriptions.

THIS IS IMPORTANT, PLEASE ENFORCE THIS RULE!!! There are NO ERRORS OR DUPLICATES in the document, even if 2 transactions have the same description, date, and amount, they should BOTH be extracted. Extract ALL transactions related to the ones mentioned in the prompt from the bank statement, even if multiple transactions occurred on the same date. Make sure to return all of them even if some are identical or happen on the same day.

The transactions may also be split up into multiple pages or chunks, be sure to include all of them if relevant.

Please provide the output in a json format because I want to be able to parse the output as a json object in Python using "json.loads".


Use the following pieces of retrieved context to answer the question.


{context}

---

Answer the question based on the above context: {question}
"""

# Create prompt
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)





##### ========== Streamlit app setup ========== #####

st.set_page_config(page_title='🦜🔗 Bank Statement QnA App')
st.title('🦜🔗 Bank Statement QnA App')

# Initialize session state for template selection and query text
if "selected_template" not in st.session_state:
    st.session_state.selected_template = ""
if "query_text" not in st.session_state:
    st.session_state.query_text = ""

# Define available template questions
template_options = [
    "----- Default: Leave blank -----",  # Blank option
    "Can you extract all of the transactions from this bank statement that contain <INSERT TRANSACTION NAME HERE>."
]

# Update session state when the user selects a template question
def update_template():
    st.session_state.selected_template = st.session_state.template_selectbox

# Read Me section
st.markdown("<br>**:red-background[GUIDE]**", unsafe_allow_html=True)
st.markdown("**:red[There is no 'best model' to use. Generally, gpt-4o-mini is more accurate, but if there are multiple records with the exact same date/description/amount, 4o tends to see those as duplicates and removes them. In this case o1-mini is better.]**")
st.markdown("**:red[In some rare cases using both works better. This app is designed to combine the outputs of both models while removing overlapping records.]**")

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

    # Add a dropdown or radio button for model selection
    options = ["gpt-4o-mini", "o1-mini", "Use all"]
    model_option = st.selectbox(
        "Choose LLM Model", 
        options, 
        index=0  # Set default index to the first option
    )
    

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
    llm_models = []
    with st.spinner('Processing...'):
        # If user selects to use all models
        if model_option == options[-1]: # corresponds with "Use all"
            for model in models_avail:
                llm_models.append(ChatOpenAI(model=model, api_key=OPENAI_API_KEY))
        # Else if user selects specific model to use
        else: 
            llm_models.append(ChatOpenAI(model=model_option, api_key=OPENAI_API_KEY))


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
            documents = all_chunks['documents']  # Returns all the stored documents (chunks)

            # Concatenate context text
            context_text = "\n\n---\n\n".join([doc for doc in documents])

            # Create prompt
            prompt = prompt_template.format(context=context_text, question=query_text)
            
            # Get response from models selected and store them in contents list
            contents = []
            for model in llm_models:
                response = model.invoke(prompt)
                contents.append(response.content)

            # Initialise lists to store output dataframes and transactions extracted from each LLM
            dfs = []
            transactions_lists = []
            
            # If response not empty
            if contents[0]:
                # Loop through each response content
                for i, content in enumerate(contents):
                    # Clean output to convert to JSON
                    clean_content = content.replace("\n", "").replace("\\", "").strip()
                    clean_content = content.strip("```json").strip("```").strip()

                    # Convert output to JSON
                    transactions = json.loads(clean_content)
                    transactions_lists.append(transactions)
                    
                    # In case the output json uses 'transactions' as a key
                    if 'transactions' in transactions:
                        transactions = transactions['transactions']
    
                    # Convert data to Pandas DataFrame
                    df = pd.DataFrame(transactions)
                    dfs.append(df)

                # Merge all DataFrames in the list if applicable
                merged_df = dfs[0]
                if len(dfs) > 1:
                    for df in dfs[1:]:
                        merged_df = pd.merge(merged_df, df, on=['date', 'description', 'amount'], how='outer', suffixes=('_A', '_B'))

                # Extract amounts and calculate total amount
                amounts = merged_df.iloc[:,2].tolist()
                total_amt = round(sum(amounts), 2)

                # Rename columns
                merged_df.columns = ['Date', 'Transaction Details', 'Amount (SGD)']

                # Output final table to screen
                st.markdown("**:blue-background[Here are the list of transactions requested]**")
                st.write(merged_df)
                st.markdown(f"The total amount is: :red[**${total_amt}**]  ")


                # Display additional information about the prompts output for each transaction
                st.markdown("<br><br><br>", unsafe_allow_html=True)
                st.markdown("**:blue-background[Additional information regarding prompt output]**")
                for i, transaction in enumerate(transactions_lists):
                    st.markdown("+"*50)
                    st.markdown(f"**Raw transactions extracted: *attempt <{i+1}>* **")
                    st.markdown(transaction)

        finally:
            shutil.rmtree(temp_vectorstore_dir)  # Clean up the temporary directory
            