"""
Authentication API Endpoints
Handles user login, registration, and token management
"""
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import JSONResponse
from datetime import timedelta
import logging

from app.config import settings
from app.dependencies import get_user_repository
from app.db.repositories.user_repository import UserRepository
from app.core.auth.jwt_handler import create_access_token
from app.api.v1.schemas.auth import (
    LoginRequest,
    LoginResponse,
    RegisterRequest,
    RegisterResponse
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/login", response_model=LoginResponse)
async def login(
    request: Request,
    login_data: LoginRequest,
    user_repo: UserRepository = Depends(get_user_repository)
):
    """
    User login endpoint
    
    Authenticates user and returns JWT access token
    
    Args:
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
        
        # Create access token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username},
            expires_delta=access_token_expires
        )
        
        logger.info(f"Successful login for user: {login_data.username}")
        
        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            username=user.username,
            is_admin=user.is_admin,
            role=user.role
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
