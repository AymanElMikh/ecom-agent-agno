import streamlit as st
from conversation import ConversationAgent
from shopping_team import ShoppingTeam
from image_processing import ProductImageProcessingAgent


st.set_page_config(
    page_title="E-commerce Shopping Assistant", 
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Report a bug": "https://github.com/aymanElMikh/ecom-agent-agno/issues"
    }
)

# Title and Header
st.title("🛒 AI-Powered Shopping Assistant")
st.write("Chat with our AI assistant to find the best products and compare them effortlessly.")

if "are_keys_avaibale" not in st.session_state:
    st.session_state["are_keys_avaibale"] = False 
        
image_container = st.container()

with st.sidebar:
    st.header("🔧 Configuration")
    llm_mode = 'OpenAI'
    
    api_key_llm = st.text_input(
        "OpenAI API Key - [OpenAI](https://console.groq.com/keys)", key="api_key_llm", type="password" )
   
    web_search_mode = st.radio("Search & Scraping Tools", ["Tavily", "SerpApi"], horizontal = True)
    
    api_key_search_tool = st.text_input(
        "Travily Or SerpApi Key - [Tavily](https://app.tavily.com) | [SerpApi](https://serpapi.com/)", key="api_key_search_tool", type="password"
        
    )
   
    api_key_firecrawl = st.text_input(
        "Firecrawl API Key - [Firecrawl](https://www.firecrawl.dev/)", key="api_key_firecrawl", type="password"
        
    )
    
    
    if st.button("Set keys"):
        
       
        st.session_state["conversation_agent"] = ConversationAgent(api_key=api_key_llm,llm_mode=llm_mode) 
        st.session_state["shopping_team"] = ShoppingTeam(api_key_llm=api_key_llm,api_key_search_tool=api_key_search_tool,search_tool=web_search_mode,llm_mode=llm_mode,firecrawl_api_key=api_key_firecrawl)
        st.session_state["image_processor"] = ProductImageProcessingAgent(api_key=api_key_llm,llm_mode=llm_mode)
        
        st.session_state["are_keys_avaibale"] = True
        
    if not st.session_state["are_keys_avaibale"]:
        st.warning("Please enter keys to get started.")
    else:
        st.success("API keys are avaibale.")
        
    st.divider()
     
    if st.button("Clear Conversation"):
        st.session_state.messages.clear() 
        if "conversation_agent" in st.session_state:
            st.session_state["conversation_agent"].reset()
        
      
    st.markdown("👨‍💻 Developed as an AI experiment. Contribute on [GitHub](https://github.com/vivekpathania/ai-experiments)")
            
  
# Main App Logic
# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_state" not in st.session_state:
    st.session_state.conversation_state = {}

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User input
user_query = st.chat_input("Ask anything about your shopping needs",accept_file=True,file_type=["jpg", "jpeg", "png"])

if user_query:
    # Append user message
    user_input = ""
    if len(user_query["text"]) > 0:
        user_input = user_query["text"]
    if len(user_query["files"]) > 0:
        f = user_query["files"][0]
       
        
        st.image(
            f,
            width = 200,
            use_container_width=False
        )
        bytes_data = f.getvalue()
        with st.spinner("Processing image..."):
            try:
                
                resp = st.session_state["image_processor"].process_image(image_data=bytes_data,user_input=user_input) 
                user_input = resp.content
                
            except Exception as e:
                st.error(f"Analysis error: {e}")
            
    
        
    if st.session_state["are_keys_avaibale"]:
        
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        response =  st.session_state['conversation_agent'].process_query(user_input)
        
        # Append assistant response
        if response['have_further_conversation']:
            st.session_state.messages.append({"role": "assistant", "content": response["message"]})
            with st.chat_message("assistant"):
                st.markdown(response["message"])
        else:
            with st.chat_message("assistant"):
                st.markdown(response["message"])
        
            with st.spinner("Fetching product recommendations..."):
                shopping_result = st.session_state["shopping_team"].run(payload=response["data"])
                
            with st.chat_message("assistant"):
                st.html(shopping_result.content)
               
    else:
        st.toast('Please enter both keys to get started.', icon='⚠️')