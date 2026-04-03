
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from api.config import get_settings

settings = get_settings()

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,       
    pool_size=10,
    max_overflow=20,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
   
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    
    from api.repositories import models  
    Base.metadata.create_all(bind=engine)
