import logging

import uvicorn
from dotenv import load_dotenv

load_dotenv(".env")

from fastapi import FastAPI

from routers.orchestration_router import router
from utils.config import AUGMENTED_IMAGES_DIR, TRAINED_MODELS_DIR

logging.basicConfig(level=logging.DEBUG)
logging.debug(f"Logging initialized with level {logging.getLevelName(logging.getLogger().level)}")
app = FastAPI()
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    logging.debug("startup_event triggered")
    logging.debug("startup_event: ensuring directories exist")
    AUGMENTED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Directories ensured: {AUGMENTED_IMAGES_DIR}, {TRAINED_MODELS_DIR}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
