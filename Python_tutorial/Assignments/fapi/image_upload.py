from fastapi import APIRouter, File, UploadFile
from starlette.background import BackgroundTasks

router = APIRouter(tags=['Upload_Image'])

@router.post("/image_upload")
async def UploadImage(background_tasks: BackgroundTasks, file: UploadFile):
    background_tasks.add_task(upload_image, file)
    return {"message": "your upload is initiated"}
    


def upload_image(file):
    with open(f'images/{file.filename}','wb') as buffer:
        print("successfully uploaded")
    return 'got it'


