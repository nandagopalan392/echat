"""
Authentication API Endpoints
Handles user login, registration, and token management with secure cookie-based auth
"""
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response, Cookie
from fastapi.responses import JSONResponse
from datetime import timedelta
from typing import Optional
import logging
import jwt
import time

from app.config import settings
from app.dependencies import get_user_repository, get_current_user
from app.db.repositories.user_repository import UserRepository
from app.db.models.user import User
from app.core.auth.jwt_handler import (
    create_token_pair, 
    verify_token,
    create_access_token
)
from app.api.v1.schemas.auth import (
    LoginRequest,
    LoginResponse,
    RegisterRequest,
    RegisterResponse,
    TokenRefreshResponse
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/login", response_model=LoginResponse)
async def login(
    response: Response,
    request: Request,
    login_data: LoginRequest,
    user_repo: UserRepository = Depends(get_user_repository)
):
    """
    User login endpoint with secure cookie-based authentication
    
    Authenticates user and sets httpOnly cookies for tokens
    
    Args:
        response: FastAPI response object for setting cookies
        request: FastAPI request object
        login_data: Login credentials
        user_repo: User repository instance
        
    Returns:
        LoginResponse with access token and user info
        
    Raises:
        HTTPException: If authentication fails
    """
    try:
        logger.info(f"Login attempt for user: {login_data.username}")
        
        # Authenticate user
        authenticated = user_repo.authenticate_user(
            login_data.username,
            login_data.password
        )
        
        if not authenticated:
            logger.warning(f"Failed login attempt for user: {login_data.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Get user details
        user = user_repo.get_user_by_username(login_data.username)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is inactive"
            )
        
        # Create access and refresh tokens
        access_token, refresh_token = create_token_pair(user.username)
        
        # Set httpOnly cookies for tokens
        # Access token cookie (short-lived)
        response.set_cookie(
            key="access_token",
            value=access_token,
            httponly=settings.COOKIE_HTTPONLY,
            secure=settings.COOKIE_SECURE,
            samesite=settings.COOKIE_SAMESITE,
            max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            domain=settings.COOKIE_DOMAIN
        )
        
        # Refresh token cookie (long-lived)
        response.set_cookie(
            key="refresh_token",
            value=refresh_token,
            httponly=settings.COOKIE_HTTPONLY,
            secure=settings.COOKIE_SECURE,
            samesite=settings.COOKIE_SAMESITE,
            max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
            domain=settings.COOKIE_DOMAIN
        )
        
        # Calculate token expiry timestamp (current time + token lifetime in milliseconds)
        current_time_ms = int(time.time() * 1000)
        expires_at_ms = current_time_ms + (settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60 * 1000)
        
        # Also store token expiry in a readable cookie for frontend
        response.set_cookie(
            key="token_expires_at",
            value=str(expires_at_ms),
            httponly=False,  # Readable by JavaScript for refresh logic
            secure=settings.COOKIE_SECURE,
            samesite=settings.COOKIE_SAMESITE,
            max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            domain=settings.COOKIE_DOMAIN
        )
        
        logger.info(f"Successful login for user: {login_data.username}")
        
        # Return response WITHOUT access_token in body (security best practice)
        # Tokens are now only in httpOnly cookies
        return LoginResponse(
            access_token=None,  # Removed: Use httpOnly cookies instead
            token_type="bearer",
            username=user.username,
            is_admin=user.is_admin,
            role=user.role,
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during login"
        )


@router.post("/register", response_model=RegisterResponse)
async def register(
    register_data: RegisterRequest,
    user_repo: UserRepository = Depends(get_user_repository)
):
    """
    User registration endpoint
    
    Creates a new user account
    
    Args:
        register_data: Registration data
        user_repo: User repository instance
        
    Returns:
        RegisterResponse with success message
        
    Raises:
        HTTPException: If registration fails
    """
    try:
        logger.info(f"Registration attempt for user: {register_data.username}")
        
        # Check if user already exists
        existing_user = user_repo.get_user_by_username(register_data.username)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        # Create new user
        success = user_repo.create_user(
            username=register_data.username,
            password=register_data.password,
            role=register_data.role,
            email=register_data.email,
            is_admin=False
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user"
            )
        
        logger.info(f"Successfully registered user: {register_data.username}")
        
        return RegisterResponse(
            message="User registered successfully",
            username=register_data.username,
            role=register_data.role
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during registration"
        )


@router.options("/login")
async def login_options():
    """Handle OPTIONS request for CORS preflight"""
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        }
    )


@router.options("/register")
async def register_options():
    """Handle OPTIONS request for CORS preflight"""
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        }
    )


@router.post("/refresh", response_model=TokenRefreshResponse)
async def refresh_token(
    response: Response,
    request: Request,
    refresh_token: Optional[str] = Cookie(None),
    user_repo: UserRepository = Depends(get_user_repository)
):
    """
    Refresh access token using refresh token
    
    Validates refresh token and issues new access token
    
    Args:
        response: FastAPI response object for setting cookies
        request: FastAPI request object
        refresh_token: Refresh token from httpOnly cookie
        user_repo: User repository instance
        
    Returns:
        TokenRefreshResponse with new access token
        
    Raises:
        HTTPException: If refresh token is invalid or expired
    """
    try:
        if not refresh_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Verify refresh token
        try:
            payload = verify_token(refresh_token, token_type="refresh")
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid refresh token: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e),
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        username = payload.get("sub")
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        
        # Verify user still exists and is active
        user = user_repo.get_user_by_username(username)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is inactive"
            )
        
        # Create new access token (refresh token stays the same)
        access_token = create_access_token(data={"sub": username})
        
        # Set new access token cookie
        response.set_cookie(
            key="access_token",
            value=access_token,
            httponly=settings.COOKIE_HTTPONLY,
            secure=settings.COOKIE_SECURE,
            samesite=settings.COOKIE_SAMESITE,
            max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            domain=settings.COOKIE_DOMAIN
        )
        
        # Update token expiry cookie
        import time
        response.set_cookie(
            key="token_expires_at",
            value=str(int(time.time() * 1000) + settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60 * 1000),
            httponly=False,
            secure=settings.COOKIE_SECURE,
            samesite=settings.COOKIE_SAMESITE,
            max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            domain=settings.COOKIE_DOMAIN
        )
        
        logger.info(f"Token refreshed for user: {username}")
        
        return TokenRefreshResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during token refresh"
        )


@router.post("/logout")
async def logout(
    response: Response,
    current_user: User = Depends(get_current_user)
):
    """
    Logout endpoint - clears authentication cookies
    
    Args:
        response: FastAPI response object
        current_user: Currently authenticated user
        
    Returns:
        Success message
    """
    try:
        # Clear all authentication cookies
        response.delete_cookie(
            key="access_token",
            domain=settings.COOKIE_DOMAIN,
            samesite=settings.COOKIE_SAMESITE
        )
        response.delete_cookie(
            key="refresh_token",
            domain=settings.COOKIE_DOMAIN,
            samesite=settings.COOKIE_SAMESITE
        )
        response.delete_cookie(
            key="token_expires_at",
            domain=settings.COOKIE_DOMAIN,
            samesite=settings.COOKIE_SAMESITE
        )
        
        logger.info(f"User logged out: {current_user.username}")
        
        return {"message": "Successfully logged out"}
        
    except Exception as e:
        logger.error(f"Logout error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during logout"
        )

