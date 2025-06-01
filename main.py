from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import io
from PyPDF2 import PdfReader
from PIL import Image

load_dotenv()

app = FastAPI(title="My API", version="1.0.0")

# CORS configuration for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React default port
        "http://127.0.0.1:3000",
        "http://localhost:3001",  # Backup React port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI!", "environment": "development"}

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "service": "backend"}

@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        # Read the file content
        content = await file.read()
        
        # Try to parse the PDF to validate it
        pdf_file = io.BytesIO(content)
        try:
            PdfReader(pdf_file)
            return {
                "status": "success",
                "message": "Valid PDF file uploaded",
                "filename": file.filename
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid PDF file")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error processing the file")
    finally:
        await file.close()

@app.post("/api/upload-image")
async def upload_image(file: UploadFile):
    # List of allowed image extensions
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    
    # Get file extension and convert to lowercase
    file_ext = os.path.splitext(file.filename.lower())[1]
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"File must be an image ({', '.join(allowed_extensions)})"
        )
    
    try:
        # Read the file content
        content = await file.read()
        
        # Try to open and validate the image
        image_stream = io.BytesIO(content)
        try:
            with Image.open(image_stream) as img:
                # Get image information
                return {
                    "status": "success",
                    "message": "Valid image file uploaded",
                    "filename": file.filename,
                    "format": img.format,
                    "mode": img.mode,
                    "size": {"width": img.width, "height": img.height}
                }
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid image file")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error processing the file")
    finally:
        await file.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)