
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy.orm import Session

from api.repositories.models import User


class UserRepository:
   

    def __init__(self, db: Session) -> None:
        self._db = db

    

    def get_by_email(self, email: str) -> Optional[User]:
        
        return (
            self._db.query(User)
            .filter(User.email == email.lower().strip())
            .first()
        )

    def get_by_id(self, user_id: UUID) -> Optional[User]:
        
        return self._db.query(User).filter(User.user_id == user_id).first()

    def email_exists(self, email: str) -> bool:
        
        return (
            self._db.query(User.user_id)
            .filter(User.email == email.lower().strip())
            .first()
        ) is not None

   

    def create(self, email: str, display_name: str, hashed_password: str) -> User:
        
        if self.email_exists(email):
            raise ValueError(f"Email '{email}' is already registered.")

        user = User(
            email=email.lower().strip(),
            display_name=display_name.strip(),
            hashed_password=hashed_password,
        )
        self._db.add(user)
        self._db.commit()
        self._db.refresh(user)
        return user

    def update_last_login(self, user: User) -> None:
        
        user.last_login = datetime.now(timezone.utc)
        self._db.commit()
