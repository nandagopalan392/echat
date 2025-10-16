"""
User Database Model
Defines the structure for user data
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class User:
    """User model representing a user in the system"""
    
    username: str
    password_hash: str
    created_at: datetime
    is_admin: bool = False
    is_active: bool = True
    role: str = "Engineer"
    email: Optional[str] = None
    last_login: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate user data after initialization"""
        if self.role not in ['Engineer', 'Manager', 'Business Development', 'Associate', 'Admin']:
            raise ValueError(f"Invalid role: {self.role}")
    
    def to_dict(self) -> dict:
        """Convert user to dictionary"""
        return {
            'username': self.username,
            'is_admin': self.is_admin,
            'is_active': self.is_active,
            'role': self.role,
            'email': self.email,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            'last_login': self.last_login.isoformat() if isinstance(self.last_login, datetime) else self.last_login
        }
    
    @classmethod
    def from_db_row(cls, row) -> 'User':
        """Create User instance from database row"""
        return cls(
            username=row['username'],
            password_hash=row['password_hash'],
            created_at=row['created_at'],
            is_admin=bool(row.get('is_admin', False)),
            is_active=bool(row.get('is_active', True)),
            role=row.get('role', 'Engineer'),
            email=row.get('email'),
            last_login=row.get('last_login')
        )


# Valid roles in the system
VALID_ROLES = [
    'Engineer',
    'Manager',
    'Business Development',
    'Associate',
    'Admin'
]
