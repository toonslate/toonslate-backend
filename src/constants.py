# TODO: 현실적인 정책으로 변경 필요
# 현재 값은 개발/테스트용. 포트폴리오 서비스 특성상 일주일에 웹툰 1화 정도가 적절.
# 예: DAILY_TRANSLATE = 10 (약 2화/주)


class TTL:
    UPLOAD = 60 * 60 * 24
    TRANSLATE = 60 * 60 * 24
    USAGE = 60 * 60 * 24
    CELERY_RESULT = 60 * 60 * 24


class RedisPrefix:
    UPLOAD = "upload"
    TRANSLATE = "translate"
    USAGE = "usage"


class Limits:
    DAILY_TRANSLATE = 100
