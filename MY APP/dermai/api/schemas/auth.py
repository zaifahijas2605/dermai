
import re
from pydantic import BaseModel, EmailStr, field_validator, model_validator


PASSWORD_POLICY_RE = re.compile(
    r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*()_+\-=\[\]{};\':\"\\|,.<>\/?]).{8,}$'
)


class RegisterRequest(BaseModel):
    email: EmailStr
    display_name: str
    password: str

    @field_validator("display_name")
    @classmethod
    def validate_display_name(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2:
            raise ValueError("Display name must be at least 2 characters.")
        if len(v) > 100:
            raise ValueError("Display name must be 100 characters or fewer.")
        return v

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if not PASSWORD_POLICY_RE.match(v):
            raise ValueError(
                "Password must be at least 8 characters and contain: "
                "one uppercase letter, one lowercase letter, one digit, "
                "and one special character (!@#$%^&* etc.)."
            )
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    user_id: str
    email: str
    display_name: str

    model_config = {"from_attributes": True}


class MessageResponse(BaseModel):
    message: str
