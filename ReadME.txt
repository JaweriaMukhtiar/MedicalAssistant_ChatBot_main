Medical Assistant Chatbot

Overview:
This project develops a medical assistant chatbot capable of processing and retrieving accurate information from biomedical documents. 
The chatbot combines cutting-edge NLP techniques, including LangChain, KeyBERT, BioBERT, GPT-2, and FAISS, to provide users with contextually 
relevant and coherent responses.

Features:
1. Efficient keyword extraction using KeyBERT.
2. Semantic vector embeddings with HuggingFace's transformer models.
3. Fast and scalable similarity search using FAISS.
4. Contextual understanding and response generation with GPT-2.
5. Biomedical-specific enhancements through BioBERT.
6. Interactive user interface built with Streamlit.

Steps to Run:
1. Install the required dependencies using:
   pip install langchain langchain-community langchain-huggingface transformers keybert faiss-cpu streamlit python-dotenv

2. Add a `.env` file with your Hugging Face API token:
   HF_TOKEN=your_huggingface_token_here

3. Place your biomedical PDF documents in the `data/` directory.

4. Execute the Jupyter Notebook (`MedicalAssistant_Chatbot.ipynb`) to preprocess data, create embeddings, and set up the QA system.

5. To deploy the chatbot interface, run the Streamlit app using the command:
   streamlit run MedicalAssistantBot.py

Output:
- A functional chatbot capable of answering biomedical queries.
- A FAISS vector store for efficient similarity searches.
- Contextually accurate responses generated using GPT-2.

Evaluation:
The chatbot has been tested to deliver accurate and relevant responses to complex biomedical queries. Its modular design ensures scalability 
and adaptability to diverse datasets.

Contributors:
- JaweriaMukhtiar(023-21-0283)
- Kajal Kattpall(023-21-0005)

Date:
January 10, 2025

Contact:
For queries or suggestions, please reach out to jaweriamukhtiar.bscsf21@iba-suk.edu.pk.

Project Demo Presentation: https://drive.google.com/file/d/1lFzuWW_ruvwTfbDW0mfarVoCxTIDZStp/view?usp=sharing