import uvicorn
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from handlers import image
from constants import configs

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=configs.ALLOW_ORIGINS,
  allow_credentials=True,
  allow_methods=configs.ALLOW_METHODS,
  allow_headers=configs.ALLOW_HEADERS,
)

app.include_router(image.router, prefix="/api/v1/image", tags=["image"])

@app.get("/readyz")
async def readyz():
  return { "message": "service is ready", "cuda": torch.cuda.is_available() }

if __name__ == "__main__":
  uvicorn.run("main:app", host="0.0.0.0", port=5000, log_level="info", reload=True) 