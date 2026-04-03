
import time
from collections import defaultdict
from datetime import datetime, timezone

import bcrypt
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from api.config import get_settings
from api.database import get_db
from api.repositories.user_repository import UserRepository
from api.schemas.auth import (
    LoginRequest, MessageResponse, RefreshRequest,
    RegisterRequest, TokenResponse, UserResponse,
)
from api.services.jwt_service import JWTService

router = APIRouter(prefix="/auth", tags=["Authentication"])
settings = get_settings()
jwt_service = JWTService()


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password[:72].encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain[:72].encode("utf-8"), hashed.encode("utf-8"))


_login_attempts: dict[str, list[float]] = defaultdict(list)

def _check_rate_limit(ip: str) -> None:
    now = time.time()
    window = settings.login_rate_limit_window_seconds
    max_attempts = settings.login_rate_limit_max
    _login_attempts[ip] = [t for t in _login_attempts[ip] if now - t < window]
    if len(_login_attempts[ip]) >= max_attempts:
        retry_after = int(window - (now - _login_attempts[ip][0]))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed login attempts. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )

def _record_failed_login(ip: str) -> None:
    _login_attempts[ip].append(time.time())

def _clear_failed_logins(ip: str) -> None:
    _login_attempts[ip] = []


from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError

security = HTTPBearer()

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> UserResponse:
    token = credentials.credentials
    try:
        user_id = jwt_service.get_user_id_from_token(token)
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        )
    repo = UserRepository(db)
    from uuid import UUID
    user = repo.get_by_id(UUID(user_id))
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found.")
    return UserResponse(
        user_id=str(user.user_id),
        email=user.email,
        display_name=user.display_name,
    )



@router.post("/register", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    repo = UserRepository(db)
    if repo.email_exists(payload.email):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This email address is already registered.",
        )
    hashed = hash_password(payload.password)
    repo.create(email=payload.email, display_name=payload.display_name, hashed_password=hashed)
    return MessageResponse(message="Account created successfully. You can now log in.")


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, request: Request, db: Session = Depends(get_db)):
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)
    repo = UserRepository(db)
    user = repo.get_by_email(payload.email)
    if not user or not verify_password(payload.password, user.hashed_password):
        _record_failed_login(client_ip)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password.")
    _clear_failed_logins(client_ip)
    repo.update_last_login(user)
    access, refresh = jwt_service.create_token_pair(user.user_id, user.email)
    return TokenResponse(access_token=access, refresh_token=refresh)


@router.post("/refresh", response_model=TokenResponse)
def refresh_token(payload: RefreshRequest):
    try:
        data = jwt_service.verify_refresh_token(payload.refresh_token)
    except JWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))
    jwt_service.revoke_refresh_token(payload.refresh_token)
    from uuid import UUID
    user_id = UUID(data["sub"])
    access, new_refresh = jwt_service.create_token_pair(user_id, "")
    return TokenResponse(access_token=access, refresh_token=new_refresh)


@router.post("/logout", response_model=MessageResponse)
def logout(payload: RefreshRequest, _: UserResponse = Depends(get_current_user)):
    jwt_service.revoke_refresh_token(payload.refresh_token)
    return MessageResponse(message="Logged out successfully.")


@router.get("/me", response_model=UserResponse)
def get_me(current_user: UserResponse = Depends(get_current_user)):
    return current_user