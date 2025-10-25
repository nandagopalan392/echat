"""
Pydantic schemas for authentication API requests and responses
"""
from pydantic import BaseModel, Field
from typing import Optional


class LoginRequest(BaseModel):
    """Login request schema"""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=3)
    
    class Config:
        json_schema_extra = {
            "example": {
                "username": "admin",
                "password": "admin123"
            }
        }


class LoginResponse(BaseModel):
    """
    Login response schema
    
    Note: access_token is deprecated and only provided for backward compatibility.
    Authentication should use httpOnly cookies instead.
    """
    access_token: Optional[str] = Field(None, description="Deprecated: Use httpOnly cookies instead")
    token_type: str = "bearer"
    username: str
    is_admin: bool = False
    role: str = "Engineer"
    expires_in: int = Field(..., description="Token expiration time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "Deprecated - tokens are in httpOnly cookies",
                "token_type": "bearer",
                "username": "admin",
                "is_admin": True,
                "role": "Admin",
                "expires_in": 3600
            }
        }


class RegisterRequest(BaseModel):
    """User registration request schema"""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    email: Optional[str] = None
    role: str = Field(default="Engineer")
    
    class Config:
        json_schema_extra = {
            "example": {
                "username": "john_doe",
                "password": "securepass123",
                "email": "john@example.com",
                "role": "Engineer"
            }
        }


class RegisterResponse(BaseModel):
    """User registration response schema"""
    message: str
    username: str
    role: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "User registered successfully",
                "username": "john_doe",
                "role": "Engineer"
            }
        }


class TokenRefreshResponse(BaseModel):
    """Token refresh response schema"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Token expiration time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 3600
            }
        }

