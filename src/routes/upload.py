from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.services.upload import UploadResponse, create_upload, get_upload

router = APIRouter(prefix="/upload", tags=["upload"])


@router.post("", response_model=UploadResponse, status_code=201)
async def upload_image(file: Annotated[UploadFile, File()]) -> UploadResponse:
    return await create_upload(file)


@router.get("/{upload_id}", response_model=UploadResponse)
async def get_upload_info(upload_id: str) -> UploadResponse:
    result = await get_upload(upload_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "UPLOAD_NOT_FOUND", "message": "존재하지 않는 업로드입니다"},
        )
    return result
