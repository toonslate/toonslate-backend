import re


class TranslateId:
    PREFIX = "tr_"
    PATTERN = re.compile(r"^tr_[a-f0-9]{8}$")


class BatchId:
    PREFIX = "batch_"
    PATTERN = re.compile(r"^batch_[a-f0-9]{8}$")


class TTL:
    DATA = 60 * 60 * 2  # 2시간 (upload/translate/batch 공통)
    CELERY_RESULT = 60 * 60 * 2
    LEGACY_USAGE = 60 * 60 * 24  # quota.py 이전 전까지 임시 유지


class RedisPrefix:
    UPLOAD = "upload"
    TRANSLATE = "translate"
    BATCH = "batch"
    USAGE = "usage"


class Limits:
    WEEKLY_IMAGES = 20  # 주 20장 (단일/배치 공유 쿼터)
    MAX_BATCH_SIZE = 10
