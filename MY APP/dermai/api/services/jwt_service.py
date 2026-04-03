
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from jose import JWTError, jwt

from api.config import get_settings

settings = get_settings()


_revoked_refresh_tokens: set[str] = set()


class JWTService:
   

    ACCESS_TOKEN_TYPE = "access"
    REFRESH_TOKEN_TYPE = "refresh"

    def __init__(self) -> None:
        self._secret = settings.jwt_secret_key
        self._algorithm = settings.jwt_algorithm
        self._access_ttl = timedelta(minutes=settings.access_token_ttl_minutes)
        self._refresh_ttl = timedelta(hours=settings.refresh_token_ttl_hours)

   

    def create_access_token(self, user_id: UUID, email: str) -> str:
        
        return self._encode(
            subject=str(user_id),
            token_type=self.ACCESS_TOKEN_TYPE,
            ttl=self._access_ttl,
            extra={"email": email},
        )

    def create_refresh_token(self, user_id: UUID) -> str:
        
        return self._encode(
            subject=str(user_id),
            token_type=self.REFRESH_TOKEN_TYPE,
            ttl=self._refresh_ttl,
        )

    def create_token_pair(self, user_id: UUID, email: str) -> tuple[str, str]:
        
        return (
            self.create_access_token(user_id, email),
            self.create_refresh_token(user_id),
        )

    

    def verify_access_token(self, token: str) -> dict:
        
        payload = self._decode(token)
        if payload.get("type") != self.ACCESS_TOKEN_TYPE:
            raise JWTError("Invalid token type.")
        return payload

    def verify_refresh_token(self, token: str) -> dict:
       
        if token in _revoked_refresh_tokens:
            raise JWTError("Refresh token has been revoked.")
        payload = self._decode(token)
        if payload.get("type") != self.REFRESH_TOKEN_TYPE:
            raise JWTError("Invalid token type.")
        return payload

    def revoke_refresh_token(self, token: str) -> None:
        
        _revoked_refresh_tokens.add(token)

    def get_user_id_from_token(self, token: str) -> str:
        
        payload = self.verify_access_token(token)
        return payload["sub"]

    

    def _encode(self, subject: str, token_type: str, ttl: timedelta, extra: dict = None) -> str:
        now = datetime.now(timezone.utc)
        payload = {
            "sub": subject,
            "type": token_type,
            "iat": now,
            "exp": now + ttl,
        }
        if extra:
            payload.update(extra)
        return jwt.encode(payload, self._secret, algorithm=self._algorithm)

    def _decode(self, token: str) -> dict:
        try:
            return jwt.decode(token, self._secret, algorithms=[self._algorithm])
        except JWTError as exc:
            raise JWTError(str(exc)) from exc
