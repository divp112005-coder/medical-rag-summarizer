from fastapi import FastAPI

app = FastAPI(title="Medical Report Summarizer API")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Medical Report Summarizer API"}