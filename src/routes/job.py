from fastapi import APIRouter, HTTPException, Request, status

from src.services import job as job_service

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("", response_model=job_service.JobResponse, status_code=status.HTTP_201_CREATED)
async def create_job(
    request: job_service.JobCreateRequest, req: Request
) -> job_service.JobResponse:
    # TODO: 프록시 환경 배포 시 X-Forwarded-For 헤더 처리 필요
    client_ip = req.client.host if req.client else "unknown"

    try:
        return await job_service.create_job(request, client_ip)
    except job_service.InvalidUploadError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "INVALID_UPLOAD_ID", "message": str(e), "uploadId": e.upload_id},
        ) from None
    except job_service.RateLimitExceededError:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"code": "RATE_LIMIT_EXCEEDED", "message": "Daily limit exceeded"},
        ) from None


@router.get("/{job_id}", response_model=job_service.JobResponse)
async def read_job(job_id: str) -> job_service.JobResponse:
    job = await job_service.get_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "JOB_NOT_FOUND", "message": f"Job not found: {job_id}"},
        )

    return job
