from pydantic_settings import BaseSettings,SettingsConfigDict
class Settings(BaseSettings):
    SECRET_KEY: str
    DATABASE_URL:str
    SMTP_EMAIL: str
    SMTP_APP_PASSWORD: str
    SMTP_SERVER: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    ADMIN_EMAIL: str
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow"
    )
    
settings = Settings()