"""
JWT token handling utilities with refresh token support.
"""
import jwt
import datetime
import secrets
from typing import Dict, Optional, Tuple
from app.config import settings


def create_access_token(data: Dict[str, str], expires_delta: Optional[datetime.timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Dictionary containing the token payload (usually {"sub": username})
        expires_delta: Optional custom expiration time
        
    Returns:
        Encoded JWT token as a string
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.datetime.utcnow(),
        "type": "access"
    })
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    
    return encoded_jwt


def create_refresh_token(data: Dict[str, str]) -> str:
    """
    Create a JWT refresh token with longer expiration.
    
    Args:
        data: Dictionary containing the token payload (usually {"sub": username})
        
    Returns:
        Encoded JWT refresh token as a string
    """
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + datetime.timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    
    # Add a unique identifier to prevent token reuse
    to_encode.update({
        "exp": expire,
        "iat": datetime.datetime.utcnow(),
        "type": "refresh",
        "jti": secrets.token_urlsafe(32)  # JWT ID for token tracking
    })
    
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def create_token_pair(username: str) -> Tuple[str, str]:
    """
    Create both access and refresh tokens for a user.
    
    Args:
        username: The username to create tokens for
        
    Returns:
        Tuple of (access_token, refresh_token)
    """
    token_data = {"sub": username}
    
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)
    
    return access_token, refresh_token


def verify_token(token: str, token_type: str = "access") -> Dict[str, str]:
    """
    Verify and decode a JWT token.
    
    Args:
        token: JWT token string to verify
        token_type: Expected token type ("access" or "refresh")
        
    Returns:
        Decoded token payload
        
    Raises:
        jwt.InvalidTokenError: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        
        # Verify token type
        if payload.get("type") != token_type:
            raise jwt.InvalidTokenError(f"Invalid token type. Expected {token_type}, got {payload.get('type')}")
        
        return payload
    except jwt.ExpiredSignatureError:
        raise jwt.InvalidTokenError("Token has expired")
    except jwt.InvalidTokenError as e:
        raise jwt.InvalidTokenError(f"Invalid token: {str(e)}")

