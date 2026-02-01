"""
Result Pattern Implementation
Type-safe error handling following AudioLab patterns
"""

from typing import Generic, TypeVar, Optional, Union
from pydantic import BaseModel

T = TypeVar('T')


class Result(Generic[T], BaseModel):
    """Result type for type-safe error handling"""
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None

    @classmethod
    def ok(cls, data: T) -> 'Result[T]':
        """Create a successful result"""
        return cls(success=True, data=data)

    @classmethod
    def err(cls, error: str) -> 'Result[T]':
        """Create an error result"""
        return cls(success=False, error=error)

    def is_ok(self) -> bool:
        """Check if result is successful"""
        return self.success

    def is_err(self) -> bool:
        """Check if result is an error"""
        return not self.success

    def unwrap(self) -> T:
        """Unwrap successful result or raise on error"""
        if self.success and self.data is not None:
            return self.data
        raise ValueError(f"Result unwrap failed: {self.error}")

    def unwrap_or(self, default: T) -> T:
        """Unwrap successful result or return default"""
        if self.success and self.data is not None:
            return self.data
        return default

    def map(self, func):
        """Map successful result through a function"""
        if self.success and self.data is not None:
            try:
                return Result.ok(func(self.data))
            except Exception as e:
                return Result.err(str(e))
        return self

    class Config:
        arbitrary_types_allowed = True