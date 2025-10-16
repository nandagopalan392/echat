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
    """Login response schema"""
    access_token: str
    token_type: str = "bearer"
    username: str
    is_admin: bool = False
    role: str = "Engineer"
    
    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "username": "admin",
                "is_admin": True,
                "role": "Admin"
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
