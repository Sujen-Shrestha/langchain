from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient

#paste your huggingfacehub_api_token here
HUGGINGFACEHUB_API_TOKEN=''

custom_prompt_template = """Answer the user's question using the provided information. If you're unsure, refrain from guessing.

Context: {context}
Question: {question}

Provide helpful answers only. End your response with a courteous "Thank You."
"""

url = 'http://localhost:6333'
collection_name = 'FA_db'

client = QdrantClient(
    url=url,
    prefer_grpc= False
)


def set_custom_prompts():
    """
    Prompt template for retrival of each vector store
    """
    prompt = PromptTemplate(input_variables=['context','question'],
                            template=custom_prompt_template,
                            template_format='f-string')
    
    return prompt

def load_llm():
    print("Loading the language model from Hugging Face Hub...🚀")
    llm = HuggingFaceHub(
        repo_id='mistralai/Mixtral-8x7B-Instruct-v0.1',
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        model_kwargs={'temperature':1,"max_length": 64,"max_new_tokens":512},
        
    )
    print("Language model loaded successfully.")
    return llm

def retrival_qa_chain(llm, prompt, db):
    # print(f'{llm},{prompt},{db}')
    print("Setting up the retrieval-based QA chain...🔗")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever = db.as_retriever(),
        return_source_documents = True,
        chain_type_kwargs={'prompt': prompt}
    )
    print("Retrieval-based QA chain ready.")
    return qa_chain

def qa_bot():
    print("Setting up the QA bot ...🤖")
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs = {'device':'cpu'} )
    
    db = Qdrant(client=client,
                embeddings=embeddings,
                collection_name="FA_db",)
    
    llm = load_llm()
    qa_prompt = set_custom_prompts()
    qa = retrival_qa_chain(llm, qa_prompt, db)
    print("QA bot ready.")
    return qa


def final_result(query):
    qa_result =qa_bot()
    response = qa_result.invoke({'query': query})

    return response

def format_output(answer):
    result = f"**Question:** {answer['query']} 🤔\n\n"
    result += f"**Answer:** 💡 {answer['result']} 🙏\n\n"

    if 'source_documents' in answer:
        result += "**Source Documents:**\n"
        for doc in answer['source_documents']:
            metadata = doc.metadata
            result += f"- Page {metadata['page']} from {metadata['source']}\n"
        result += "\n"
    
    result += 'This is only generated answer, if you have actual health problems,\n \
        please consult to an actual health professional.❗ \n'

    return result

def main():
    qa_result = qa_bot()

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        response = qa_result.invoke({'query': user_input})
        formatted_output = format_output(response)
        print("Bot:", formatted_output)

if __name__ == '__main__':
    main()