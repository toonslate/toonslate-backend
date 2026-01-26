from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.services import upload as upload_service

router = APIRouter(prefix="/upload", tags=["upload"])


@router.post("", response_model=upload_service.UploadResponse, status_code=201)
async def create_upload(file: Annotated[UploadFile, File()]) -> upload_service.UploadResponse:
    return await upload_service.create_upload(file)


@router.get("/{upload_id}", response_model=upload_service.UploadResponse)
async def read_upload(upload_id: str) -> upload_service.UploadResponse:
    result = await upload_service.get_upload(upload_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "UPLOAD_NOT_FOUND", "message": f"Upload not found: {upload_id}"},
        )
    return result
