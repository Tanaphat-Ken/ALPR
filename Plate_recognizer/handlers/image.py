from typing import List
from fastapi import APIRouter, HTTPException, File, UploadFile, Body
from constants import configs
from models.image_processor import ImageProcessor
from libs import utils
from libs.logging import logger

router = APIRouter()
image_processor = ImageProcessor()

@router.post("/process")
async def process_image(file: UploadFile = File(...)):
  try:
    if file.content_type not in configs.ALLOWED_IMAGE_TYPES:
      raise ValueError({ "message": f"Invalid file type. {configs.ALLOWED_IMAGE_TYPES}" })

    file_content = await file.read()
    image = utils.image_open_and_verify(file_content)
    return image_processor.read(image)

  except ValueError as e:
    raise HTTPException(status_code=400, detail=e.args[0])
  except Exception as e:
    raise HTTPException(status_code=500, detail={ "message": f"An unexpected error occurred: {str(e)}" })

@router.post("/process/skip/car")
async def process_image_skip_car(car_bbox: List[float] = Body(...), file: UploadFile = File(...)):
  try:
    if file.content_type not in configs.ALLOWED_IMAGE_TYPES:
      raise ValueError({ "message": f"Invalid file type. {configs.ALLOWED_IMAGE_TYPES}" })

    file_content = await file.read()
    image = utils.image_open_and_verify(file_content)
    return image_processor.read(image, car_bbox)

  except ValueError as e:
    raise HTTPException(status_code=400, detail=e.args[0])
  except Exception as e:
    logger.info(str(e))
    raise HTTPException(status_code=500, detail={ "message": f"An unexpected error occurred: {str(e)}" })