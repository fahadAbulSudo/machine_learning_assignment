from pydantic import BaseSettings


class Settings(BaseSettings):
    secret_key: str
    algorithm: str
    access_token_expire_minutes: int
    api_key: str

    class Config:
        env_file = ".env"


settings = Settings()