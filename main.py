
# import os
# import streamlit as st
# import pickle
# import requests
# import yfinance as yf
# from bs4 import BeautifulSoup
# import time
# from groq import Groq
# from langchain.chains import RetrievalQAWithSourcesChain
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from dotenv import load_dotenv
# import plotly.graph_objs as go
# import base64
# import json

# # Load environment variables
# load_dotenv()

# # Streamlit UI setup
# st.title("Trading Insight Tool 📈")

# # Initialize Groq client
# client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# # Constants
# COMMUNITY_POSTS_FILE = "community_posts.json"
# file_path = "faiss_store_groq.pkl"

# # Initialize session state variables
# if 'query' not in st.session_state:
#     st.session_state['query'] = ''
# if 'answer' not in st.session_state:
#     st.session_state['answer'] = ''
# if 'source_urls' not in st.session_state:
#     st.session_state['source_urls'] = []
# if 'page' not in st.session_state:
#     st.session_state['page'] = 'main'
# if 'stock_symbol' not in st.session_state:
#     st.session_state['stock_symbol'] = ''

# # Load community posts from file
# def load_community_posts():
#     if os.path.exists(COMMUNITY_POSTS_FILE):
#         with open(COMMUNITY_POSTS_FILE, 'r') as file:
#             return json.load(file)
#     return []

# # Save community posts to file
# def save_community_posts(posts):
#     with open(COMMUNITY_POSTS_FILE, 'w') as file:
#         json.dump(posts, file)

# # Community Post Feature
# def community_post_page():
#     st.subheader("Community Posts")

#     # Load existing posts
#     community_posts = load_community_posts()

#     # Form to submit a new post
#     with st.form("new_post_form"):
#         name = st.text_input("Your Name:", max_chars=50)
#         content = st.text_area("Write your post here:", height=150)
#         image_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
#         keyword = st.text_input("Keyword/Stock Name for Post Filtering:")

#         submitted = st.form_submit_button("Submit Post")
#         if submitted and content and name:
#             # Handle the image upload
#             image_url = None
#             if image_file is not None:
#                 image_bytes = image_file.read()
#                 image_url = f"data:image/{image_file.type.split('/')[1]};base64,{base64.b64encode(image_bytes).decode()}"
            
#             # Add the new post to the list
#             new_post = {"name": name, "content": content, "image": image_url, "keyword": keyword}
#             community_posts.append(new_post)
#             save_community_posts(community_posts)  # Save posts to file
            
#             st.success("Post submitted!")

#     # Filter posts by keyword
#     search_keyword = st.text_input("Search Posts by Keyword/Stock Name:")

#     # Button to see filtered posts
#     see_filtered_posts = st.button("See Filtered Posts")
    
#     # Show filtered posts based on the keyword
#     if see_filtered_posts:
#         if not search_keyword:
#             st.warning("Please enter a keyword or stock name to filter the posts.")
#         else:
#             filtered_posts = [post for post in community_posts if search_keyword.lower() in post.get('keyword', '').lower()]
#             if not filtered_posts:
#                 st.write(f"No posts found for keyword: {search_keyword}")
#             else:
#                 st.write("### Filtered Posts:")
#                 for idx, post in enumerate(filtered_posts):
#                     display_post(post)

#     # Show all posts if the "See All Posts" button is clicked
#     if st.button("See All Posts"):
#         if community_posts:
#             st.write("### All Posts:")
#             for idx, post in enumerate(community_posts):
#                 display_post(post)
#         else:
#             st.write("No posts available yet.")

#     # Back to Main Page button
#     if st.button("Back to Main Page"):
#         st.session_state['page'] = 'main'



# # Function to display individual posts in a styled format
# def display_post(post):
#     card_style = """
#     <div style='border: 2px solid #4CAF50; border-radius: 10px; padding: 10px; display: inline-block;'>
#         <h5 style='margin: 0; color: #333;'>{name} posted:</h5>
#         <p style='color: #555;'>{content}</p>
#         {image_tag}
#     </div>
#     """.format(name=post['name'], content=post['content'], image_tag=f"<img src='{post['image']}' width='300' style='border-radius: 5px;'>" if post.get('image') else '')

#     st.markdown(card_style, unsafe_allow_html=True)
#     st.write("---")

# # Stock History Feature
# def stock_history_page():
#     st.subheader("Stock History")

#     stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA):", value=st.session_state['stock_symbol'])
#     st.session_state['stock_symbol'] = stock_symbol  # Update session state

#     if st.button("Get Stock History"):
#         if stock_symbol:
#             # Fetch stock history using yfinance
#             stock = yf.Ticker(stock_symbol)
#             stock_info = stock.info
#             hist_data = stock.history(period="3mo")

#             if not hist_data.empty:
#                 # Plot stock history with candlestick chart
#                 fig = go.Figure(data=[go.Candlestick(
#                     x=hist_data.index,
#                     open=hist_data['Open'],
#                     high=hist_data['High'],
#                     low=hist_data['Low'],
#                     close=hist_data['Close'],
#                     increasing_line_color='green',
#                     decreasing_line_color='red'
#                 )])
#                 fig.update_layout(title=f"{stock_symbol} Candlestick Chart (Last 3 Months)",
#                                   xaxis_title='Date',
#                                   yaxis_title='Price',
#                                   xaxis_rangeslider_visible=True)
#                 st.plotly_chart(fig)

#                 # Display basic stock information in an expander (collapsible)
#                 with st.expander("📊 Stock Information", expanded=True):
#                     st.write(f"**Market Cap:** {stock_info.get('marketCap', 'N/A')}")
#                     st.write(f"**Volume:** {stock_info.get('volume', 'N/A')}")
#                    # st.write(f"**Current Price:** {stock_info.get('regularMarketPrice', 'N/A')}")
#                     st.write(f"**Sector:** {stock_info.get('sector', 'N/A')}")
#                     st.write(f"**Industry:** {stock_info.get('industry', 'N/A')}")
#             else:
#                 st.error(f"No data found for stock symbol: {stock_symbol}")
#         else:
#             st.error("Please enter a valid stock symbol.")

# # Main Page Navigation
# def main_page():
#     urls = []
#     for i in range(3):
#         url = st.sidebar.text_input(f"URL {i + 1}")
#         if url:
#             urls.append(url)

#     process_url_clicked = st.sidebar.button("Process URLs")

#     # Buttons for showing community posts and stock history
#     if st.sidebar.button("Show Community Posts"):
#         st.session_state['page'] = 'community_post'

#     if st.sidebar.button("Show Stock History"):
#         st.session_state['page'] = 'stock_history'

#     if process_url_clicked:
#         if not urls:
#             st.error("Please provide at least one valid URL.")
#         else:
#             data = []
#             metadata = []  # To store associated URLs
#             for url in urls:
#                 content, source_url = fetch_url_content(url)
#                 if content:
#                     data.append(content)
#                     metadata.append({"url": source_url})  # Append URL as metadata
#                     st.write(f"Fetched content from {url}")
#                 else:
#                     st.warning(f"Failed to fetch content from {url}")

#             if data:
#                 text_splitter = RecursiveCharacterTextSplitter(
#                     separators=['\n\n', '\n', '.', ','],
#                     chunk_size=1000
#                 )
#                 docs = text_splitter.create_documents(data, metadatas=metadata)  # Add metadata to chunks

#                 if docs:
#                     st.write(f"Text split successfully: {len(docs)} chunks created.")

#                     # Create embeddings for the documents
#                     with st.spinner("Building embedding vectors..."):
#                         embeddings = HuggingFaceEmbeddings()
#                         vectorstore_groq = FAISS.from_documents(docs, embeddings)

#                     # Save FAISS index to a file
#                     save_faiss_index(vectorstore_groq, file_path)
#                 else:
#                     st.error("No documents were generated after text splitting.")
#             else:
#                 st.error("No data was loaded from the URLs. Please check the URLs and try again.")

#     st.session_state['query'] = st.text_input("Enter your question:", key="query_input")

#     if st.button("Get Answer"):
#         if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
#             vectorstore = load_faiss_index(file_path)
#             if vectorstore:
#                 # Retrieve relevant documents from FAISS based on the query
#                 retriever = vectorstore.as_retriever()
#                 retrieved_docs = retriever.get_relevant_documents(st.session_state['query'])
#                 if retrieved_docs:
#                     # Initialize QA chain with sources
#                     chain = RetrievalQAWithSourcesChain.from_chain_type(
#                         llm=HuggingFaceEmbeddings(), retriever=retriever
#                     )
#                     st.session_state['answer'] = chain(st.session_state['query'])

#                     # Display answer and sources
#                     st.write("### Answer:")
#                     st.write(st.session_state['answer']['result'])
#                     st.write("### Sources:")
#                     for source in st.session_state['answer']['source_documents']:
#                         st.write(source.metadata['url'])  # Display the source URL
#                 else:
#                     st.error("No relevant documents found for the query.")
#             else:
#                 st.error("Failed to load vector store.")
#         else:
#             st.error("No index available. Please process some URLs first.")

# # Function to fetch URL content
# def fetch_url_content(url):
#     try:
#         response = requests.get(url)
#         if response.status_code == 200:
#             soup = BeautifulSoup(response.content, 'html.parser')
#             return soup.get_text(), url  # Return plain text and the URL
#     except Exception as e:
#         print(f"Error fetching {url}: {e}")
#     return None, None

# # Function to save FAISS index
# def save_faiss_index(vectorstore, file_path):
#     with open(file_path, "wb") as f:
#         pickle.dump(vectorstore, f)

# # Function to load FAISS index
# def load_faiss_index(file_path):
#     if os.path.exists(file_path):
#         with open(file_path, "rb") as f:
#             return pickle.load(f)
#     return None

# # Determine which page to display based on user selection
# if st.session_state['page'] == 'community_post':
#     community_post_page()
# elif st.session_state['page'] == 'stock_history':
#     stock_history_page()
# else:
#     main_page()











import os
import streamlit as st
import pickle
import requests
import yfinance as yf
from bs4 import BeautifulSoup
import time
from groq import Groq
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import plotly.graph_objs as go
import base64
import json

# Load environment variables
load_dotenv()

# Streamlit UI setup
st.title("Trading Insight Tool 📈")

# Initialize Groq client
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
else:
    client = Groq(api_key=api_key)

# Constants
COMMUNITY_POSTS_FILE = "community_posts.json"
file_path = "faiss_store_groq.pkl"

# Initialize session state variables
if 'query' not in st.session_state:
    st.session_state['query'] = ''
if 'answer' not in st.session_state:
    st.session_state['answer'] = ''
if 'source_urls' not in st.session_state:
    st.session_state['source_urls'] = []
if 'page' not in st.session_state:
    st.session_state['page'] = 'main'
if 'stock_symbol' not in st.session_state:
    st.session_state['stock_symbol'] = ''

# Load community posts from file
def load_community_posts():
    if os.path.exists(COMMUNITY_POSTS_FILE):
        with open(COMMUNITY_POSTS_FILE, 'r') as file:
            return json.load(file)
    return []

# Save community posts to file
def save_community_posts(posts):
    with open(COMMUNITY_POSTS_FILE, 'w') as file:
        json.dump(posts, file)

# Community Post Feature
def community_post_page():
    st.subheader("Community Posts")

    # Load existing posts
    community_posts = load_community_posts()

    # Form to submit a new post
    with st.form("new_post_form"):
        name = st.text_input("Your Name:", max_chars=50)
        content = st.text_area("Write your post here:", height=150)
        image_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
        keyword = st.text_input("Keyword/Stock Name for Post Filtering:")

        submitted = st.form_submit_button("Submit Post")
        if submitted and content and name:
            # Handle the image upload
            image_url = None
            if image_file is not None:
                image_bytes = image_file.read()
                image_url = f"data:image/{image_file.type.split('/')[1]};base64,{base64.b64encode(image_bytes).decode()}"
            
            # Add the new post to the list
            new_post = {"name": name, "content": content, "image": image_url, "keyword": keyword}
            community_posts.append(new_post)
            save_community_posts(community_posts)  # Save posts to file
            
            st.success("Post submitted!")

    # Filter posts by keyword
    search_keyword = st.text_input("Search Posts by Keyword/Stock Name:")

    # Button to see filtered posts
    see_filtered_posts = st.button("See Filtered Posts")
    
    # Show filtered posts based on the keyword
    if see_filtered_posts:
        if not search_keyword:
            st.warning("Please enter a keyword or stock name to filter the posts.")
        else:
            filtered_posts = [post for post in community_posts if search_keyword.lower() in post.get('keyword', '').lower()]
            if not filtered_posts:
                st.write(f"No posts found for keyword: {search_keyword}")
            else:
                st.write("### Filtered Posts:")
                for idx, post in enumerate(filtered_posts):
                    display_post(post)

    # Show all posts if the "See All Posts" button is clicked
    if st.button("See All Posts"):
        if community_posts:
            st.write("### All Posts:")
            for idx, post in enumerate(community_posts):
                display_post(post)
        else:
            st.write("No posts available yet.")

    # Back to Main Page button
    if st.button("Back to Main Page"):
        st.session_state['page'] = 'main'

# Function to display individual posts in a styled format
def display_post(post):
    card_style = """
    <div style='border: 2px solid #4CAF50; border-radius: 10px; padding: 10px; display: inline-block;'>
        <h5 style='margin: 0; color: #333;'>{name} posted:</h5>
        <p style='color: #555;'>{content}</p>
        {image_tag}
    </div>
    """.format(name=post['name'], content=post['content'], image_tag=f"<img src='{post['image']}' width='300' style='border-radius: 5px;'>" if post.get('image') else '')

    st.markdown(card_style, unsafe_allow_html=True)
    st.write("---")

# Stock History Feature
def stock_history_page():
    st.subheader("Stock History")

    stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA):", value=st.session_state['stock_symbol'])
    st.session_state['stock_symbol'] = stock_symbol  # Update session state

    if st.button("Get Stock History"):
        if stock_symbol:
            # Fetch stock history using yfinance
            stock = yf.Ticker(stock_symbol)
            stock_info = stock.info
            hist_data = stock.history(period="3mo")

            if not hist_data.empty:
                # Plot stock history with candlestick chart
                fig = go.Figure(data=[go.Candlestick(
                    x=hist_data.index,
                    open=hist_data['Open'],
                    high=hist_data['High'],
                    low=hist_data['Low'],
                    close=hist_data['Close'],
                    increasing_line_color='green',
                    decreasing_line_color='red'
                )])
                fig.update_layout(title=f"{stock_symbol} Candlestick Chart (Last 3 Months)",
                                  xaxis_title='Date',
                                  yaxis_title='Price',
                                  xaxis_rangeslider_visible=True)
                st.plotly_chart(fig)

                # Display basic stock information in an expander (collapsible)
                with st.expander("📊 Stock Information", expanded=True):
                    st.write(f"**Market Cap:** {stock_info.get('marketCap', 'N/A')}")
                    st.write(f"**Volume:** {stock_info.get('volume', 'N/A')}")
                    st.write(f"**Sector:** {stock_info.get('sector', 'N/A')}")
                    st.write(f"**Industry:** {stock_info.get('industry', 'N/A')}")
            else:
                st.error(f"No data found for stock symbol: {stock_symbol}")
        else:
            st.error("Please enter a valid stock symbol.")

# Main Page Navigation
def main_page():
    urls = []
    for i in range(3):
        url = st.sidebar.text_input(f"URL {i + 1}")
        if url:
            urls.append(url)

    process_url_clicked = st.sidebar.button("Process URLs")

    # Buttons for showing community posts and stock history
    if st.sidebar.button("Show Community Posts"):
        st.session_state['page'] = 'community_post'

    if st.sidebar.button("Show Stock History"):
        st.session_state['page'] = 'stock_history'

    if process_url_clicked:
        if not urls:
            st.error("Please provide at least one valid URL.")
        else:
            data = []
            metadata = []  # To store associated URLs
            for url in urls:
                content, source_url = fetch_url_content(url)
                if content:
                    data.append(content)
                    metadata.append({"url": source_url})  # Append URL as metadata
                    st.write(f"Fetched content from {url}")
                else:
                    st.warning(f"Failed to fetch content from {url}")

            if data:
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=['\n\n', '\n', '.', ','],
                    chunk_size=1000
                )
                docs = text_splitter.create_documents(data, metadatas=metadata)  # Add metadata to chunks

                if docs:
                    st.write(f"Text split successfully: {len(docs)} chunks created.")

                    # Create embeddings for the documents
                    with st.spinner("Building embedding vectors..."):
                        embeddings = HuggingFaceEmbeddings()
                        vectorstore_groq = FAISS.from_documents(docs, embeddings)

                    # Save FAISS index to a file
                    save_faiss_index(vectorstore_groq, file_path)
                else:
                    st.error("No documents were generated after text splitting.")
            else:
                st.error("No data was loaded from the URLs. Please check the URLs and try again.")

    st.session_state['query'] = st.text_input("Enter your question:", key="query_input")

    if st.button("Get Answer"):
        if st.session_state['query']:
            # Initialize GroqLLM
            groq_llm = GroqLLM(client)

            # Example usage of Langchain RetrievalQAWithSourcesChain
            with st.spinner("Processing your question..."):
                qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
                    groq_llm,
                    chain_type="stuff",
                    retriever=vectorstore_groq.as_retriever()
                )
                result = qa_chain({"question": st.session_state['query']})

                st.session_state['answer'] = result['answer']
                st.session_state['source_urls'] = result['source_urls']

        st.write("### Answer:", st.session_state['answer'])
        st.write("### Sources:")
        for url in st.session_state['source_urls']:
            st.write(f"- {url}")

# Routing between pages
if st.session_state['page'] == 'main':
    main_page()
elif st.session_state['page'] == 'community_post':
    community_post_page()
elif st.session_state['page'] == 'stock_history':
    stock_history_page()
