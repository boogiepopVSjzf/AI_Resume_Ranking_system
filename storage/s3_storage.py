from __future__ import annotations

from dataclasses import dataclass
from mimetypes import guess_type

from config import settings
from utils.errors import DatabaseError
from utils.logger import get_logger

logger = get_logger("s3_storage")
_client = None


@dataclass
class StorageUploadResult:
    bucket: str
    key: str
    mime_type: str


def _require_storage_config() -> None:
    if settings.ENABLE_S3_STORAGE:
        return
    raise DatabaseError(
        "S3 storage is not configured. Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, and S3_BUCKET_NAME."
    )


def _get_s3_client():
    global _client
    _require_storage_config()
    if _client is None:
        try:
            import boto3
        except ModuleNotFoundError as exc:
            raise DatabaseError(
                "boto3 is not installed. Run `pip install -r requirements.txt` first."
            ) from exc

        kwargs = {
            "aws_access_key_id": settings.AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": settings.AWS_SECRET_ACCESS_KEY,
            "region_name": settings.AWS_REGION,
        }
        if settings.S3_ENDPOINT_URL:
            kwargs["endpoint_url"] = settings.S3_ENDPOINT_URL
        _client = boto3.client("s3", **kwargs)
    return _client


def _guess_mime_type(filename: str, fallback_ext: str) -> str:
    guessed, _ = guess_type(filename)
    if guessed:
        return guessed
    if fallback_ext.lower() == ".pdf":
        return "application/pdf"
    if fallback_ext.lower() == ".docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return "application/octet-stream"


def build_resume_storage_key(resume_id: str, ext: str) -> str:
    normalized_ext = ext.lower()
    if not normalized_ext.startswith("."):
        normalized_ext = f".{normalized_ext}"
    return f"raw/{resume_id}{normalized_ext}"


def upload_resume_source_file(
    *,
    resume_id: str,
    filename: str,
    ext: str,
    content: bytes,
) -> StorageUploadResult:
    """Upload the original resume file into AWS S3."""
    bucket = settings.S3_BUCKET_NAME
    key = build_resume_storage_key(resume_id, ext)
    mime_type = _guess_mime_type(filename, ext)
    client = _get_s3_client()

    try:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=content,
            ContentType=mime_type,
        )
    except Exception as exc:
        raise DatabaseError(f"Failed to upload resume to S3: {exc}") from exc

    logger.info("Uploaded resume source file %s to S3 bucket %s", resume_id, bucket)
    return StorageUploadResult(bucket=bucket, key=key, mime_type=mime_type)
