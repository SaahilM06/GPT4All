    import streamlit as st
    from pathlib import Path
    from dotenv import load_dotenv
    from PyPDF2 import PdfReader
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.embeddings import HuggingFaceInstructEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain
    from htmlTemplates import css, bot_template, user_template
    from langchain.llms import HuggingFaceHub
    env_path = Path('.') / '.env'
    load_dotenv(dotenv_path=env_path)


    css = '''
    <style>
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
    }
    .chat-message.user {
        background-color: #2b313e
    }
    .chat-message.bot {
        background-color: #475063
    }
    .chat-message .avatar {
    width: 20%;
    }
    .chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
    }
    .chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
    color: #fff;
    }
    '''

    bot_template = '''
    <div class="chat-message bot">
        <div class="avatar">
            <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
        </div>
        <div class="message">{{MSG}}</div>
    </div>
    '''

    user_template = '''
    <div class="chat-message user">
        <div class="avatar">
            <img src="https://static.vecteezy.com/system/resources/thumbnails/001/840/612/small/picture-profile-icon-male-icon-human-or-people-sign-and-symbol-free-vector.jpg">
        </div>    
        <div class="message">{{MSG}}</div>
    </div>
    '''


    def get_pdf_text(pdf_docs):
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text








    def get_text_chunks(text):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks








    def get_vectorstore(text_chunks):
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore








    def get_conversation_chain(vectorstore):
        llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})




        memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        return conversation_chain








    def handle_userinput(user_question):
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']




        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)








    def main():
        
        st.set_page_config(page_title="Test program")
        st.write(css, unsafe_allow_html=True)




        if "conversation" not in st.session_state:
            st.session_state.conversation = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = None




        st.header("Test Program")
        user_question = st.text_input("Ask a question you may have:")
        if user_question:
            handle_userinput(user_question)




        with st.sidebar:
            st.subheader("Your documents")
            pdf_docs = st.file_uploader(
                "Upload your PDFs here and click on 'Scan'", accept_multiple_files=True)
            if st.button("Scan"):
                with st.spinner("Scanning"):
                    # get pdf text
                    raw_text = get_pdf_text(pdf_docs)




                    # get the text chunks
                    text_chunks = get_text_chunks(raw_text)




                    # create vector store
                    vectorstore = get_vectorstore(text_chunks)




                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(
                        vectorstore)








    if __name__ == '__main__':
        main()



