from consultent_agent import main
import streamlit as st

def streamlit_ui():
    st.title("Business Information Chatbot")
    st.sidebar.title("Settings")

    # Initialize session state
    if 'business_info' not in st.session_state:
        st.session_state.business_info = ""
    if 'finished' not in st.session_state:
        st.session_state.finished = False

    # Collect business information
    st.write("Please provide information about your business:")
    business_info = st.text_area("Business Information", value=st.session_state.business_info)

    if st.button("Submit"):
        st.session_state.business_info = business_info
        st.session_state.finished = True

    if st.session_state.finished:
        st.write("Thank you for providing the information. Processing...")
        # Call the main function to process the data
        main()

if __name__ == "__main__":
    streamlit_ui()