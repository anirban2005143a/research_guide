from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import get_answer
from fastapi import FastAPI, UploadFile, File, HTTPException
from uploaded_pdf_storing import process_file
import shutil
import tempfile
import os

app = FastAPI()

TEMP_DIR = tempfile.mkdtemp()

# Add CORS middleware to allow all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Allows all origins
    allow_credentials=True,    # Allows cookies/auth headers
    allow_methods=["*"],      # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],      # Allows all headers
)

# ---------- Request Model ----------
class QueryRequest(BaseModel):
    query: str

@app.get("/")
async def root():
    return {"message": "CORS is enabled for all!"}

# endpoint for file upload
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    # check file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    try:
        # Generate a temporary file path for saving the PDF
        temp_file_path = os.path.join(TEMP_DIR, file.filename)

        # Save the file directly to disk without reading it into memory
        with open(temp_file_path, "wb") as f:
            # Write the file content directly to disk
            # 'file.file' is a file-like object
            shutil.copyfileobj(file.file, f)

        # Pass the file path to the function for further processing
        process_file(temp_file_path)

        # Optionally, remove the file after processing (cleanup)
        os.remove(temp_file_path)

        return {
            "status": "success",
            "message": "PDF uploaded successfully",
            "filename": file.filename
        }

    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=500,
            detail="PDF upload failed"
        )

# endpoint for query and answer
# ---------- Hardcoded Answer Endpoint ----------
@app.post("/ask-question")
def ask_question(data: QueryRequest):
    user_query = data.query

    # 🔒 Hardcoded answer for now
    # answer = "This is a hardcoded answer. AI integration will be added later."
    answer = get_answer(user_query)

    return {
        "query": user_query,
        "answer": answer,
        "status": "success"
    }