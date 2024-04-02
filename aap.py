from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, ServiceContext, StorageContext, load_index_from_storage
from llama_index.core.query_engine import SubQuestionQueryEngine
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile,Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse



import numpy as np
import uuid
import urllib.parse
from typing import List, Optional
import os

Settings.llm = OpenAI(model="gpt-4", temperature=0)
Settings.chunk_size = 512

from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")

from llama_index.core.query_engine import FLAREInstructQueryEngine
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata


from llama_index.core.agent import ReActAgent

import requests


app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost:3000",
    "http://localhost:8000",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)





import uvloop
import asyncio

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

reports_in_progress = set()

import uvicorn 

agent = None

agent_available = {}  

reports= {}
user_ids = set()

####################################  Upload Routes #####################################

@app.get("/")
def hi():
    return "Go to /docs for documentation Ashsish"

@app.post("/upload/medical_docs/{user_id}")
async def medical_docs_download(user_id: str, pdf_urls: List[str]):
    global user_ids
    try:
        user_folder_path = f"./db/medical_docs/{user_id}"
        if not os.path.exists(user_folder_path):
            os.makedirs(user_folder_path)

        file_paths = [f"doc_{i + 1}.pdf" for i in range(len(pdf_urls))]
        downloaded_files = []

        for index, pdf_url in enumerate(pdf_urls):
            # Encode special characters in the URL
            encoded_url = urllib.parse.quote(pdf_url, safe="%/:=&?~#+!$,;'@()*[]")

            response = requests.get(encoded_url, allow_redirects=True)

            if response.status_code == 200:
                file_path = os.path.join(user_folder_path, file_paths[index])
                with open(file_path, 'wb') as pdf_file:
                    pdf_file.write(response.content)
                downloaded_files.append(file_path)


                ### DEBUG: Load medical documents HERE itself

                medical_dir = f"./db/medical_docs/{user_id}/"
                if os.path.exists(medical_dir):
                    medical_docs = SimpleDirectoryReader(medical_dir).load_data()
                    print("Len of Transcript docs: ",len(medical_docs))
                    print(medical_docs)
                else:
                    medical_docs = []
                
                
                if medical_docs:
                    print("Creating medical records index and persisting to storage.")
                    # Create medical records index and persist to storage
                    medical_index = VectorStoreIndex.from_documents(
                        medical_docs)
                    
                    medical_index.storage_context.persist(persist_dir=f"./storage_context/medical_docs/{user_id}")
                else:
                    print("No medical documents found.So no indexing embedding done")

            else:
                raise HTTPException(status_code=response.status_code, detail=f"Failed to download PDF at index {index}. Status code: {response.status_code}")

        user_ids.add(user_id)
        print(f"Downloaded files: {downloaded_files}")
        return [FileResponse(file, media_type='application/pdf', filename=os.path.basename(file)) for file in downloaded_files]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")





@app.post("/upload/transcript_docs/{user_id}")
async def download_transcript_pdfs(user_id: str, pdf_urls: List[str]):
    global user_ids
    try:
        user_folder_path = f"./db/transcript_docs/{user_id}"
        if not os.path.exists(user_folder_path):
            os.makedirs(user_folder_path)

        file_paths = [f"doc_{i + 1}.pdf" for i in range(len(pdf_urls))]
        downloaded_files = []

        for index, pdf_url in enumerate(pdf_urls):
            # Encode special characters in the URL
            encoded_url = urllib.parse.quote(pdf_url, safe="%/:=&?~#+!$,;'@()*[]")

            response = requests.get(encoded_url, allow_redirects=True)

            if response.status_code == 200:
                file_path = os.path.join(user_folder_path, file_paths[index])
                with open(file_path, 'wb') as pdf_file:
                    pdf_file.write(response.content)
                downloaded_files.append(file_path)

            
                ### DEBUG: Load transcript documents HERE itself
                transcript_dir = f"./db/transcript_docs/{user_id}/"
                if os.path.exists(transcript_dir):
                    transcript_docs = SimpleDirectoryReader(transcript_dir).load_data()
                    print("Len of Transcript docs: ",len(transcript_docs))
                else:
                    transcript_docs = []

                print(f"Number of transcript documents loaded: {len(transcript_docs)}")

                if transcript_docs:
                    print("Creating transcripts index and persisting to storage.")
                    # Create transcripts index and persist to storage
                    transcript_index = VectorStoreIndex.from_documents(
                        transcript_docs)
                    transcript_index.storage_context.persist(persist_dir=f"./storage_context/transcript/{user_id}")
                    print("Transcript persisted..")

            else:
                raise HTTPException(status_code=response.status_code, detail=f"Failed to download PDF at index {index}. Status code: {response.status_code}")

        user_ids.add(user_id)
        print(f"Downloaded files: {downloaded_files}")
        return [FileResponse(file, media_type='application/pdf', filename=os.path.basename(file)) for file in downloaded_files]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/upload/pleading_docs/{user_id}")
async def download_pleading_pdfs(user_id: str, pdf_urls: List[str]):
    global user_ids
    try:
        user_folder_path = f"./db/pleading_docs/{user_id}"
        if not os.path.exists(user_folder_path):
            os.makedirs(user_folder_path)

        file_paths = [f"doc_{i + 1}.pdf" for i in range(len(pdf_urls))]
        downloaded_files = []

        for index, pdf_url in enumerate(pdf_urls):
            # Encode special characters in the URL
            encoded_url = urllib.parse.quote(pdf_url, safe="%/:=&?~#+!$,;'@()*[]")

            response = requests.get(encoded_url, allow_redirects=True)

            if response.status_code == 200:
                file_path = os.path.join(user_folder_path, file_paths[index])
                with open(file_path, 'wb') as pdf_file:
                    pdf_file.write(response.content)
                downloaded_files.append(file_path)

            
                ### DEBUG: Load transcript documents HERE itself
                pleading_dir = f"./db/pleading_docs/{user_id}/"
                if os.path.exists(pleading_dir):
                    pleading_docs = SimpleDirectoryReader(pleading_dir).load_data()
                    print("Len of pleading docs: ",len(pleading_docs))
                else:
                    pleading_docs = []

                print(f"Number of pleading documents loaded: {len(pleading_docs)}")

                if pleading_docs:
                    print("Creating pleadings index and persisting to storage.")
                    # Create transcripts index and persist to storage
                    pleading_index = VectorStoreIndex.from_documents(
                        pleading_docs)
                    pleading_index.storage_context.persist(persist_dir=f"./storage_context/pleading/{user_id}")
                    print("pleading index persisted..")

            else:
                raise HTTPException(status_code=response.status_code, detail=f"Failed to download PDF at index {index}. Status code: {response.status_code}")

        user_ids.add(user_id)
        print(f"Downloaded files: {downloaded_files}")
        return [FileResponse(file, media_type='application/pdf', filename=os.path.basename(file)) for file in downloaded_files]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.post("/upload/misc_docs/{user_id}")
async def download_misc_pdfs(user_id: str, pdf_urls: List[str]):
    global user_ids
    try:
        user_folder_path = f"./db/misc_docs/{user_id}"
        if not os.path.exists(user_folder_path):
            os.makedirs(user_folder_path)

        file_paths = [f"doc_{i + 1}.pdf" for i in range(len(pdf_urls))]
        downloaded_files = []

        for index, pdf_url in enumerate(pdf_urls):
            # Encode special characters in the URL
            encoded_url = urllib.parse.quote(pdf_url, safe="%/:=&?~#+!$,;'@()*[]")

            response = requests.get(encoded_url, allow_redirects=True)

            if response.status_code == 200:
                file_path = os.path.join(user_folder_path, file_paths[index])
                with open(file_path, 'wb') as pdf_file:
                    pdf_file.write(response.content)
                downloaded_files.append(file_path)

            
                ### DEBUG: Load transcript documents HERE itself
                misc_dir = f"./db/misc_docs/{user_id}/"
                if os.path.exists(misc_dir):
                    misc_docs = SimpleDirectoryReader(misc_dir).load_data()
                    print("Len of misc docs: ",len(misc_docs))
                else:
                    misc_docs = []

                print(f"Number of misc documents loaded: {len(misc_docs)}")

                if misc_docs:
                    print("Creating miscs index and persisting to storage.")
                    # Create transcripts index and persist to storage
                    misc_index = VectorStoreIndex.from_documents(
                        misc_docs)
                    misc_index.storage_context.persist(persist_dir=f"./storage_context/misc/{user_id}")
                    print("misc index persisted")

            else:
                raise HTTPException(status_code=response.status_code, detail=f"Failed to download PDF at index {index}. Status code: {response.status_code}")

        user_ids.add(user_id)
        print(f"Downloaded files: {downloaded_files}")
        return [FileResponse(file, media_type='application/pdf', filename=os.path.basename(file)) for file in downloaded_files]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")




@app.post("/load_chat/{user_id}")
def load(user_id: str, background_tasks: BackgroundTasks):
    global agent_available

    # Background process to load documents and create tools
    def load_documents_and_create_tools():
        print("Starting document loading and tool creation process.")

        openai = OpenAI(model="gpt-4-1106-preview", openai_api_key=openai_api_key)
        function_llm = OpenAI(model="gpt-4-1106-preview", openai_api_key=openai_api_key)
        

        ### DEBUG: Loadthe index from storage HERE itself\

        print("**** \n load user_id: ",user_id)
        print("")
        medical_storage_c = StorageContext.from_defaults(persist_dir=f"./storage_context/medical_docs/{user_id}/")
        print(medical_storage_c)
        medical_index = load_index_from_storage(medical_storage_c)
        transcript_storage_c = StorageContext.from_defaults(persist_dir=f"./storage_context/transcript/{user_id}/")
        transcript_index = load_index_from_storage(transcript_storage_c)
        pleading_storage_c = StorageContext.from_defaults(persist_dir=f"./storage_context/pleading/{user_id}/")
        pleading_index = load_index_from_storage(pleading_storage_c)
        misc_storage_c = StorageContext.from_defaults(persist_dir=f"./storage_context/misc/{user_id}/")
        misc_index = load_index_from_storage(misc_storage_c)




        medical_metadata = ToolMetadata(name="MedicalSearchTool", description=" useful for when you need to search medical records documents for queries")
        transcript_metadata = ToolMetadata(name="TranscriptSearchTool", description="useful for when you need to search transcripts of case hearings for queries")
        pleading_metadata = ToolMetadata(name="PleadingSearchTool", description="useful for when you need to Search court pleadings filings for more details on court pleadings or court dynamics")
        misc_metadata = ToolMetadata(name="MiscSearchTool", description="useful for when you need to Search extra documents related for more details on legal case")


        medical_search_engine = medical_index.as_query_engine()
        medical_flare_query_engine = FLAREInstructQueryEngine(
             query_engine=medical_search_engine,
                max_iterations=15,
                verbose=True,
            )
        medical_search_tool = QueryEngineTool(metadata=medical_metadata, query_engine=medical_flare_query_engine)

        transcript_search_engine = transcript_index.as_query_engine()
        transcript_flare_query_engine = FLAREInstructQueryEngine(
             query_engine=transcript_search_engine,
                max_iterations=15,
                verbose=True,
            )
        transcript_search_tool = QueryEngineTool(metadata=transcript_metadata, query_engine=transcript_flare_query_engine)

        pleading_search_engine = pleading_index.as_query_engine()
        pleading_flare_query_engine = FLAREInstructQueryEngine(
             query_engine=pleading_search_engine,
                max_iterations=15,
                verbose=True,
            )
        pleading_search_tool = QueryEngineTool(metadata=pleading_metadata, query_engine=pleading_flare_query_engine)

        misc_search_engine = misc_index.as_query_engine()
        misc_flare_query_engine = FLAREInstructQueryEngine(
             query_engine=misc_search_engine,
                max_iterations=15,
                verbose=True,
            )
        misc_search_tool = QueryEngineTool(metadata=misc_metadata, query_engine=misc_flare_query_engine)

        case_tools = [medical_search_tool,transcript_search_tool,pleading_search_tool,misc_search_tool]#,petition_search_tool]

        query_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=case_tools,
        )

        query_engine_tool = QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name="sub_question_query_engine",
                description="useful for when you want to answer queries that require analyzing multiple documents related to a case",
            ),
        )

        print("Creating and storing the agent.")
        # Creating and storing the agent
        if case_tools:
            tools = case_tools + [query_engine_tool]
            agent = ReActAgent.from_tools(
                tools,
                llm=function_llm,
                verbose=True,
                max_function_calls=100,
                system_prompt=f"""\
        You are a specialized agent named Compfox designed to answer queries about Legal Cases.
        You must ALWAYS use at least one of the tools provided when answering a question for context, and try to use all the tools for context, for answering.

        """,)

            agent_available[user_id] = agent
            print("Agent created and stored in memory.")
            return "Done creating agent."

        print("Document loading and tool creation process completed.")
        return {"msg": "Documents loaded and tools created."}


    # Add the background process to the tasks
    background_tasks.add_task(load_documents_and_create_tools)

    return {"msg": "Load process initiated."}



@app.post("/chat/{user_id}")
async def chat(user_id: str, message: str):
    global agent_available
    agent = agent_available[user_id]
    rep = ""
    response = agent.stream_chat(message)
    response_gen = response.response_gen
    for token in response_gen:
        rep += str(token)
    return {"response": str(rep)}


@app.post("/report-generate/{user_id}")
async def report_generate(user_id: str, background_tasks: BackgroundTasks):
    global agent_available
    agent = agent_available[user_id]

    report_id = str(uuid.uuid4())

    def generate_report_task():
        response =  agent.chat('''
            Using the tools, you have access to, make a report based on these tasks:
            1. *Comprehend the Courtroom Dynamics:*
               - Dive into the court transcript to understand the case dynamics.
               - What were the main arguments presented during the trial?

            2. *Extract Crucial Legal Points:*
               - Identify key legal points discussed in the courtroom.
               - What legal issues or claims emerged from the proceedings?

            3. *Analyze the Medical Report:*
               - Scrutinize the medical report for relevant details.
               - How do the medical facts correlate with legal aspects?

            4. *Bridge Medical Information with Legal Context:*
               - Connect medical information to legal arguments.
               - What legal implications arise from the medical findings?

            5. *Identify Legal Standards in Medical Testimony:*
               - Explore legal standards related to medical testimony.
               - How does the medical evidence align with these standards?

            6. ** Suggested Segments:**
               - Based on this research, suggest some points/questions that can be looked into as part of Legal Research
               - Suggest points that can be looked into as part of Legal Research for a Lawyer representing this case.
            After you have looked into these tasks, make an elaborate report so far for my legal Client. You are my savior legal buddy.
        ''')


        report_id = str(uuid.uuid4())
        reports[report_id] = response
        return response

        # Assuming you want to do something with the response, you can add your logic here.
        

    background_tasks.add_task(generate_report_task)
    return {"status": "Report generation started.", "report_id": report_id}

@app.get("/report-status/{report_id}")
async def report_status(report_id: str):
    if report_id not in reports_in_progress:
        raise HTTPException(status_code=404, detail="Report not found or already completed.")
    if report_id in reports:
        return {"status": "Report completed.", "report": str(reports[report_id])}
    else:
        # raise HTTPException(status_code=202, detail="Report Still in progress...")
        return {"status": "Report in progress."}


@app.get("/load_chat/status/{user_id}")
async def get_load_status(user_id: str):
    if user_id in agent_available:
        return JSONResponse(content={"status": "Agent created and loaded successfully."}, status_code=200)
    else:
        if user_id not in user_ids:
            return JSONResponse(content={"status": "No Data available to load man!."}, status_code=404)
        return JSONResponse(content={"status": "Agent creation and loading in progress or failed."}, status_code=404)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0",port=8080)