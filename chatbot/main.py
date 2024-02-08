from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient

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

    llm = CTransformers(model = '../llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type = 'llama',
                        max_new_tokens = 512,
                        temperature = 0.5)
    
    return llm

def retrival_qa_chain(llm, prompt, db):
    print(f'{llm},{prompt},{db}')
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever = db.as_retriever(),
        return_source_documents = True,
        chain_type_kwargs={'prompt': prompt}
    )

    return qa_chain

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs = {'device':'cpu'} )
    
    db = Qdrant(client=client,
                embeddings=embeddings,
                collection_name="FA_db",)
    
    llm = load_llm()
    qa_prompt = set_custom_prompts()
    qa = retrival_qa_chain(llm, qa_prompt, db)

    return qa


def final_result(query):
    qa_result =qa_bot()
    response = qa_result.invoke({'query': query})
    return response

if __name__ == '__main__':
    query_from_user = input("Enter your question:")
    answer = final_result(query_from_user)
    print(answer)