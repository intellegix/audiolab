"""
Base Repository
Common CRUD operations for all entities
"""

import uuid
from typing import Generic, TypeVar, Type, Optional, List, Dict, Any
from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError, NoResultFound
from sqlalchemy.orm import selectinload, joinedload

from ..connection import Base

# Type variables for generic repository
ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType")
UpdateSchemaType = TypeVar("UpdateSchemaType")


class RepositoryError(Exception):
    """Base repository error"""
    pass


class NotFoundError(RepositoryError):
    """Entity not found error"""
    pass


class ConflictError(RepositoryError):
    """Data conflict error"""
    pass


class BaseRepository(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """Base repository with common CRUD operations"""

    def __init__(self, model: Type[ModelType], session: AsyncSession):
        self.model = model
        self.session = session

    async def create(self, obj_in: CreateSchemaType, **kwargs) -> ModelType:
        """Create a new entity"""
        try:
            # Convert Pydantic model to dict
            if hasattr(obj_in, 'model_dump'):
                obj_data = obj_in.model_dump()
            else:
                obj_data = obj_in.dict()

            # Add any additional kwargs
            obj_data.update(kwargs)

            # Create model instance
            db_obj = self.model(**obj_data)
            self.session.add(db_obj)
            await self.session.flush()
            await self.session.refresh(db_obj)
            return db_obj

        except IntegrityError as e:
            await self.session.rollback()
            raise ConflictError(f"Data conflict: {str(e)}")

    async def get(self, id: uuid.UUID) -> Optional[ModelType]:
        """Get entity by ID"""
        try:
            result = await self.session.execute(
                select(self.model).where(self.model.id == id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            raise RepositoryError(f"Error getting entity: {str(e)}")

    async def get_or_404(self, id: uuid.UUID) -> ModelType:
        """Get entity by ID or raise NotFoundError"""
        obj = await self.get(id)
        if obj is None:
            raise NotFoundError(f"{self.model.__name__} with id {id} not found")
        return obj

    async def get_multi(
        self,
        skip: int = 0,
        limit: int = 100,
        order_by: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ModelType]:
        """Get multiple entities with pagination and filtering"""
        try:
            query = select(self.model)

            # Apply filters
            if filters:
                for field, value in filters.items():
                    if hasattr(self.model, field):
                        column = getattr(self.model, field)
                        if isinstance(value, list):
                            query = query.where(column.in_(value))
                        else:
                            query = query.where(column == value)

            # Apply ordering
            if order_by and hasattr(self.model, order_by):
                column = getattr(self.model, order_by)
                query = query.order_by(column)
            elif hasattr(self.model, 'created_at'):
                query = query.order_by(self.model.created_at.desc())

            # Apply pagination
            query = query.offset(skip).limit(limit)

            result = await self.session.execute(query)
            return result.scalars().all()

        except Exception as e:
            raise RepositoryError(f"Error getting entities: {str(e)}")

    async def update(self, id: uuid.UUID, obj_in: UpdateSchemaType) -> Optional[ModelType]:
        """Update entity by ID"""
        try:
            # Get existing entity
            db_obj = await self.get_or_404(id)

            # Convert Pydantic model to dict, excluding unset values
            if hasattr(obj_in, 'model_dump'):
                update_data = obj_in.model_dump(exclude_unset=True)
            else:
                update_data = obj_in.dict(exclude_unset=True)

            # Update fields
            for field, value in update_data.items():
                if hasattr(db_obj, field):
                    setattr(db_obj, field, value)

            await self.session.flush()
            await self.session.refresh(db_obj)
            return db_obj

        except NotFoundError:
            raise
        except IntegrityError as e:
            await self.session.rollback()
            raise ConflictError(f"Data conflict: {str(e)}")
        except Exception as e:
            await self.session.rollback()
            raise RepositoryError(f"Error updating entity: {str(e)}")

    async def delete(self, id: uuid.UUID) -> bool:
        """Delete entity by ID"""
        try:
            # Check if entity exists
            db_obj = await self.get_or_404(id)

            # Delete entity
            await self.session.delete(db_obj)
            await self.session.flush()
            return True

        except NotFoundError:
            raise
        except Exception as e:
            await self.session.rollback()
            raise RepositoryError(f"Error deleting entity: {str(e)}")

    async def exists(self, id: uuid.UUID) -> bool:
        """Check if entity exists"""
        try:
            result = await self.session.execute(
                select(func.count(self.model.id)).where(self.model.id == id)
            )
            count = result.scalar()
            return count > 0
        except Exception as e:
            raise RepositoryError(f"Error checking entity existence: {str(e)}")

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count entities with optional filters"""
        try:
            query = select(func.count(self.model.id))

            # Apply filters
            if filters:
                for field, value in filters.items():
                    if hasattr(self.model, field):
                        column = getattr(self.model, field)
                        if isinstance(value, list):
                            query = query.where(column.in_(value))
                        else:
                            query = query.where(column == value)

            result = await self.session.execute(query)
            return result.scalar()

        except Exception as e:
            raise RepositoryError(f"Error counting entities: {str(e)}")

    async def get_with_relations(
        self,
        id: uuid.UUID,
        relations: List[str]
    ) -> Optional[ModelType]:
        """Get entity with eager-loaded relationships"""
        try:
            query = select(self.model).where(self.model.id == id)

            # Add relationship loading
            for relation in relations:
                if hasattr(self.model, relation):
                    query = query.options(selectinload(getattr(self.model, relation)))

            result = await self.session.execute(query)
            return result.scalar_one_or_none()

        except Exception as e:
            raise RepositoryError(f"Error getting entity with relations: {str(e)}")

    async def bulk_create(self, objs_in: List[CreateSchemaType]) -> List[ModelType]:
        """Create multiple entities in bulk"""
        try:
            db_objs = []

            for obj_in in objs_in:
                # Convert Pydantic model to dict
                if hasattr(obj_in, 'model_dump'):
                    obj_data = obj_in.model_dump()
                else:
                    obj_data = obj_in.dict()

                db_obj = self.model(**obj_data)
                db_objs.append(db_obj)

            self.session.add_all(db_objs)
            await self.session.flush()

            # Refresh all objects to get IDs
            for db_obj in db_objs:
                await self.session.refresh(db_obj)

            return db_objs

        except IntegrityError as e:
            await self.session.rollback()
            raise ConflictError(f"Data conflict in bulk create: {str(e)}")
        except Exception as e:
            await self.session.rollback()
            raise RepositoryError(f"Error in bulk create: {str(e)}")

    async def bulk_update(
        self,
        updates: List[Dict[str, Any]]
    ) -> int:
        """Update multiple entities in bulk"""
        try:
            count = 0

            for update_data in updates:
                entity_id = update_data.pop('id', None)
                if not entity_id:
                    continue

                stmt = update(self.model).where(
                    self.model.id == entity_id
                ).values(**update_data)

                result = await self.session.execute(stmt)
                count += result.rowcount

            await self.session.flush()
            return count

        except IntegrityError as e:
            await self.session.rollback()
            raise ConflictError(f"Data conflict in bulk update: {str(e)}")
        except Exception as e:
            await self.session.rollback()
            raise RepositoryError(f"Error in bulk update: {str(e)}")

    async def bulk_delete(self, ids: List[uuid.UUID]) -> int:
        """Delete multiple entities by IDs"""
        try:
            stmt = delete(self.model).where(self.model.id.in_(ids))
            result = await self.session.execute(stmt)
            await self.session.flush()
            return result.rowcount

        except Exception as e:
            await self.session.rollback()
            raise RepositoryError(f"Error in bulk delete: {str(e)}")

    async def search(
        self,
        search_term: str,
        search_fields: List[str],
        skip: int = 0,
        limit: int = 100
    ) -> List[ModelType]:
        """Search entities across multiple text fields"""
        try:
            query = select(self.model)

            # Build search conditions
            search_conditions = []
            for field in search_fields:
                if hasattr(self.model, field):
                    column = getattr(self.model, field)
                    if hasattr(column.type, 'python_type') and column.type.python_type == str:
                        search_conditions.append(column.ilike(f"%{search_term}%"))

            if search_conditions:
                from sqlalchemy import or_
                query = query.where(or_(*search_conditions))

            # Apply pagination
            query = query.offset(skip).limit(limit)

            # Order by relevance (created_at as fallback)
            if hasattr(self.model, 'created_at'):
                query = query.order_by(self.model.created_at.desc())

            result = await self.session.execute(query)
            return result.scalars().all()

        except Exception as e:
            raise RepositoryError(f"Error searching entities: {str(e)}")