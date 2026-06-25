from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.predict import router

app = FastAPI(
    title="BISINDO API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "message": "BISINDO API Running"
    }


@app.get("/health")
def health():
    return {
        "status": "online"
    }