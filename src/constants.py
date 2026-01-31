class TTL:
    UPLOAD = 60 * 60 * 24
    JOB = 60 * 60 * 24
    USAGE = 60 * 60 * 24
    CELERY_RESULT = 60 * 60 * 24


class RedisPrefix:
    UPLOAD = "upload"
    JOB = "job"
    USAGE = "usage"


class Limits:
    DAILY_JOB = 100
