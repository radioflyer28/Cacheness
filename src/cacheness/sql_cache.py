"""
SQL Pull-Through Cache Implementation

This module provides a caching layer that sits between your application and SQL databases,
automatically fetching and caching query results with intelligent cache invalidation.

Key Features:
- SQLAlchemy Table object integration for type-safe schema definitions
- Automatic upsert operations with cross-database compatibility
- Intelligent missing data detection and partial fetching
- Configurable TTL (Time To Live) with automatic cleanup
- Support for complex query patterns (ranges, filters, etc.)
- Built-in cache statistics and monitoring

Example Usage:
    >>> from sqlalchemy import Table, Column, String, Date, Float, MetaData
    >>> from cacheness.sql_cache import SQLAlchemyPullThroughCache, SQLAlchemyDataAdapter
    >>> 
    >>> # Define your table schema
    >>> metadata = MetaData()
    >>> stock_table = Table(
    ...     'stock_prices',
    ...     metadata,
    ...     Column('symbol', String(10), primary_key=True),
    ...     Column('date', Date, primary_key=True),
    ...     Column('close', Float),
    ... )
    >>> 
    >>> # Create a data adapter
    >>> class StockAdapter(SQLAlchemyDataAdapter):
    ...     def get_table_definition(self):
    ...         return stock_table
    ...     
    ...     def fetch_data(self, **kwargs):
    ...         # Your API/database fetch logic here
    ...         return pandas_dataframe
    >>> 
    >>> # Create and use the cache
    >>> cache = SQLAlchemyPullThroughCache("stocks.db", stock_table, StockAdapter())
    >>> data = cache.get_data(symbol="AAPL", start_date="2024-01-01", end_date="2024-01-31")
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from sqlalchemy import Table
    from sqlalchemy.orm import Session

try:
    from sqlalchemy import (
        Column, DateTime, Table, and_, between, create_engine, 
        delete, func, insert, or_, select, update
    )
    from sqlalchemy.orm import Session, sessionmaker
    
    HAS_SQLALCHEMY = True
    
except ImportError:
    HAS_SQLALCHEMY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class SQLCacheError(Exception):
    """Base exception for SQL cache operations"""
    pass


class MissingDependencyError(SQLCacheError):
    """Raised when required dependencies are not available"""
    pass


def check_dependencies():
    """Check if required dependencies are available"""
    if not HAS_SQLALCHEMY:
        raise MissingDependencyError(
            "SQLAlchemy is required for SQL cache functionality. "
            "Install with: pip install 'cacheness[sql]'"
        )
    
    if not HAS_PANDAS:
        raise MissingDependencyError(
            "pandas is required for SQL cache functionality. "
            "Install with: pip install 'cacheness[dataframes]'"
        )


class SQLAlchemyDataAdapter(ABC):
    """
    Abstract adapter for fetching data from external sources.
    
    This class defines the interface that data adapters must implement
    to work with the SQLAlchemy pull-through cache system.
    """
    
    @abstractmethod
    def get_table_definition(self) -> 'Table':
        """
        Return the SQLAlchemy Table object that defines the cache schema.
        
        Returns:
            Table: SQLAlchemy Table object with columns, constraints, and indexes
        """
        pass
    
    @abstractmethod
    def fetch_data(self, **kwargs) -> 'pd.DataFrame':
        """
        Fetch data from the external source.
        
        Args:
            **kwargs: Query parameters for data fetching
            
        Returns:
            pd.DataFrame: Data fetched from the source, with columns matching
                         the table definition
        """
        pass
    
    @abstractmethod
    def parse_query_params(self, **kwargs) -> Dict[str, Any]:
        """
        Parse and validate query parameters.
        
        Args:
            **kwargs: Raw query parameters
            
        Returns:
            Dict[str, Any]: Parsed and validated parameters
        """
        pass


class SQLAlchemyPullThroughCache:
    """
    Pull-through cache implementation using SQLAlchemy.
    
    This cache automatically fetches missing data from external sources
    and stores it in a local database for fast subsequent access.
    """
    
    def __init__(
        self,
        db_url: str,
        table: "Table",
        data_adapter: SQLAlchemyDataAdapter,
        ttl_hours: int = 24,
        echo: bool = False,
        engine_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the SQL pull-through cache.
        
        Args:
            db_url: Database URL (supports DuckDB, SQLite, PostgreSQL)
            table: SQLAlchemy Table object defining the cache schema
            data_adapter: Adapter for fetching data from external sources
            ttl_hours: Cache TTL in hours (0 for no expiration)
            echo: Whether to echo SQL statements for debugging
            engine_kwargs: Additional arguments for SQLAlchemy engine
        """
        check_dependencies()
        
        if not HAS_SQLALCHEMY:
            raise MissingDependencyError("SQLAlchemy is required")
        
        # Normalize database URL
        if not any(db_url.startswith(prefix) for prefix in [
            'duckdb://', 'sqlite://', 'postgresql://', 'mysql://'
        ]):
            # Assume it's a file path, default to DuckDB
            db_url = f'duckdb:///{db_url}'
        
        # Create SQLAlchemy engine
        engine_kwargs = engine_kwargs or {}
        self.engine = create_engine(db_url, echo=echo, **engine_kwargs)  # type: ignore
        self.Session = sessionmaker(bind=self.engine)  # type: ignore
        
        # Store configuration
        self.data_adapter = data_adapter
        self.ttl_hours = ttl_hours
        
        # Set up table with cache metadata columns
        self.table = table
        self._add_cache_columns()
        
        # Create table in database
        self.table.create(self.engine, checkfirst=True)
        
        # Cache table information for performance
        self._setup_query_helpers()
        """
        Initialize the SQL pull-through cache.
        
        Args:
            db_url: Database URL (supports DuckDB, SQLite, PostgreSQL)
            table: SQLAlchemy Table object defining the cache schema
            data_adapter: Adapter for fetching data from external sources
            ttl_hours: Cache TTL in hours (0 for no expiration)
            echo: Whether to echo SQL statements for debugging
            engine_kwargs: Additional arguments for SQLAlchemy engine
        """
        check_dependencies()
        
        # Normalize database URL
        if not any(db_url.startswith(prefix) for prefix in [
            'duckdb://', 'sqlite://', 'postgresql://', 'mysql://'
        ]):
            # Assume it's a file path, default to DuckDB
            db_url = f'duckdb:///{db_url}'
        
        # Create SQLAlchemy engine
        engine_kwargs = engine_kwargs or {}
        self.engine = create_engine(db_url, echo=echo, **engine_kwargs)
        self.Session = sessionmaker(bind=self.engine)
        
        # Store configuration
        self.data_adapter = data_adapter
        self.ttl_hours = ttl_hours
        
        # Set up table with cache metadata columns
        self.table = table
        self._add_cache_columns()
        
        # Create table in database
        self.table.create(self.engine, checkfirst=True)
        
        # Cache table information for performance
        self._setup_query_helpers()
    
    def _add_cache_columns(self):
        """Add cache metadata columns to the table if not present"""
        if not HAS_SQLALCHEMY:
            return
            
        existing_columns = {col.name for col in self.table.columns}
        
        # Add cached_at timestamp
        if 'cached_at' not in existing_columns:
            self.table.append_column(
                Column('cached_at', DateTime, default=func.now(), nullable=False)  # type: ignore
            )
        
        # Add expires_at timestamp if TTL is configured
        if 'expires_at' not in existing_columns and self.ttl_hours > 0:
            self.table.append_column(
                Column('expires_at', DateTime, nullable=True)  # type: ignore
            )
    
    def _setup_query_helpers(self):
        """Set up cached information about the table structure"""
        self.primary_key_cols = [col.name for col in self.table.primary_key.columns]
        self.data_columns = [
            col.name for col in self.table.columns 
            if col.name not in ('cached_at', 'expires_at')
        ]
    
    def get_data(self, **query_params) -> 'pd.DataFrame':
        """
        Main pull-through cache method.
        
        Checks the cache first, fetches missing data from the external source,
        and returns the complete dataset.
        
        Args:
            **query_params: Query parameters for data retrieval
            
        Returns:
            pd.DataFrame: Complete dataset from cache
            
        Raises:
            SQLCacheError: If there's an error with cache operations
        """
        try:
            parsed_params = self.data_adapter.parse_query_params(**query_params)
            
            with self.Session() as session:
                # Check cache first
                cached_data = self._get_cached_data(session, parsed_params)
                
                # Find missing data ranges
                missing_ranges = self._find_missing_data(parsed_params, cached_data)
                
                # Fetch and store missing data
                if missing_ranges:
                    for missing_params in missing_ranges:
                        try:
                            fresh_data = self.data_adapter.fetch_data(**missing_params)
                            if not fresh_data.empty:
                                self._store_in_cache(session, fresh_data)
                        except Exception as e:
                            # Log the error but continue with other ranges
                            print(f"Warning: Failed to fetch data for {missing_params}: {e}")
                    
                    session.commit()
                
                # Return complete dataset from cache
                return self._get_cached_data(session, parsed_params)
                
        except Exception as e:
            raise SQLCacheError(f"Error in get_data: {e}") from e
    
    def _get_cached_data(self, session: "Session", query_params: Dict[str, Any]) -> "pd.DataFrame":
        """Retrieve data from cache based on query parameters"""
        # Select only data columns (excluding cache metadata)
        data_cols = [
            self.table.c[col] for col in self.data_columns 
            if col in self.table.c
        ]
        
        query = select(*data_cols)  # type: ignore
        conditions = self._build_query_conditions(query_params)
        
        if conditions:
            query = query.where(and_(*conditions))  # type: ignore
        
        # Execute query and convert to DataFrame
        result = session.execute(query)
        df = pd.DataFrame(  # type: ignore
            result.fetchall(), 
            columns=[col.name for col in data_cols]
        )
        
        return df
    
    def _build_query_conditions(self, query_params: Dict[str, Any]) -> List:
        """Build WHERE conditions from query parameters"""
        conditions = []
        
        for key, value in query_params.items():
            if key not in self.table.c:
                continue
                
            column = self.table.c[key]
            
            # Handle different value types
            if isinstance(value, dict):
                # Range queries
                if 'start' in value and 'end' in value:
                    conditions.append(between(column, value['start'], value['end']))  # type: ignore
                elif 'gte' in value:
                    conditions.append(column >= value['gte'])
                elif 'lte' in value:
                    conditions.append(column <= value['lte'])
                elif 'gt' in value:
                    conditions.append(column > value['gt'])
                elif 'lt' in value:
                    conditions.append(column < value['lt'])
                elif 'in' in value:
                    conditions.append(column.in_(value['in']))
                elif 'not_in' in value:
                    conditions.append(~column.in_(value['not_in']))
            elif isinstance(value, (list, tuple)):
                # IN clause
                conditions.append(column.in_(value))
            else:
                # Exact match
                conditions.append(column == value)
        
        # Add TTL condition if expires_at column exists
        if 'expires_at' in self.table.c:
            conditions.append(
                or_(  # type: ignore
                    self.table.c.expires_at.is_(None),
                    self.table.c.expires_at > func.now()  # type: ignore
                )
            )
        
        return conditions
    
    def _store_in_cache(self, session: Session, data: 'pd.DataFrame'):
        """Store data in cache with upsert behavior"""
        if data.empty:
            return
        
        # Add cache metadata
        data = data.copy()
        if 'cached_at' in self.table.c:
            data['cached_at'] = datetime.now()
        if 'expires_at' in self.table.c and self.ttl_hours > 0:
            data['expires_at'] = datetime.now() + timedelta(hours=self.ttl_hours)
        
        # Filter to only include columns that exist in the table
        table_columns = {col.name for col in self.table.columns}
        data_columns = [col for col in data.columns if col in table_columns]
        data_filtered = data[data_columns]
        
        # Convert to records for insertion
        records = data_filtered.to_dict('records')
        
        # Attempt upsert operation
        try:
            self._upsert_records(session, records)
        except Exception as e:
            # Fallback to individual row operations
            print(f"Upsert failed, using fallback method: {e}")
            self._fallback_upsert(session, records)
    
    def _upsert_records(self, session: Session, records: List[Dict]):
        """Perform bulk upsert operation"""
        if not records:
            return
        
        dialect = self.engine.dialect.name
        
        if dialect in ('sqlite', 'duckdb'):
            # Use SQLite-style upsert
            from sqlalchemy.dialects.sqlite import insert as sqlite_insert
            stmt = sqlite_insert(self.table)
            
            if self.primary_key_cols:
                update_dict = {
                    col.name: stmt.excluded[col.name] 
                    for col in self.table.columns 
                    if col.name not in self.primary_key_cols
                }
                
                stmt = stmt.on_conflict_do_update(
                    index_elements=self.primary_key_cols,
                    set_=update_dict
                )
                
        elif dialect == 'postgresql':
            # Use PostgreSQL-style upsert
            from sqlalchemy.dialects.postgresql import insert as postgresql_insert
            stmt = postgresql_insert(self.table)
            
            if self.primary_key_cols:
                update_dict = {
                    col.name: stmt.excluded[col.name] 
                    for col in self.table.columns 
                    if col.name not in self.primary_key_cols
                }
                
                stmt = stmt.on_conflict_do_update(
                    index_elements=self.primary_key_cols,
                    set_=update_dict
                )
        else:
            # Fallback to regular insert for other databases
            stmt = insert(self.table)  # type: ignore
        
        session.execute(stmt, records)
    
    def _fallback_upsert(self, session: "Session", records: List[Dict]):
        """Fallback upsert using individual operations"""
        for record in records:
            if not self.primary_key_cols:
                # No primary key, just insert
                session.execute(insert(self.table).values(**record))  # type: ignore
                continue
            
            # Build primary key conditions
            pk_conditions = [
                self.table.c[col] == record[col] 
                for col in self.primary_key_cols 
                if col in record
            ]
            
            if not pk_conditions:
                session.execute(insert(self.table).values(**record))  # type: ignore
                continue
            
            # Check if record exists
            existing = session.execute(
                select(self.table).where(and_(*pk_conditions))  # type: ignore
            ).first()
            
            if existing:
                # Update existing record
                update_stmt = (
                    update(self.table)  # type: ignore
                    .where(and_(*pk_conditions))  # type: ignore
                    .values(**record)
                )
                session.execute(update_stmt)
            else:
                # Insert new record
                session.execute(insert(self.table).values(**record))  # type: ignore
    
    def _find_missing_data(
        self, 
        query_params: Dict[str, Any], 
        cached_data: 'pd.DataFrame'
    ) -> List[Dict[str, Any]]:
        """
        Find missing data ranges - this method should be overridden by
        specific implementations or the data adapter should provide this logic.
        
        Default implementation assumes all data is missing if cache is empty.
        """
        if cached_data.empty:
            # Convert query params back to fetch format
            fetch_params = {}
            for key, value in query_params.items():
                if isinstance(value, dict):
                    if 'start' in value and 'end' in value:
                        fetch_params[f'{key}_start'] = value['start']
                        fetch_params[f'{key}_end'] = value['end']
                    else:
                        fetch_params[key] = value
                else:
                    fetch_params[key] = value
            
            return [fetch_params]
        
        # If we have cached data, assume it's complete
        # Subclasses should override this for more sophisticated logic
        return []
    
    def invalidate_cache(self, **query_params):
        """
        Invalidate specific cache entries.
        
        Args:
            **query_params: Parameters to identify entries to invalidate
        """
        try:
            parsed_params = self.data_adapter.parse_query_params(**query_params)
            
            with self.Session() as session:
                conditions = self._build_query_conditions(parsed_params)
                
                if conditions:
                    # Remove TTL condition for invalidation
                    data_conditions = [
                        cond for cond in conditions 
                        if 'expires_at' not in str(cond)
                    ]
                    
                    if data_conditions:
                        stmt = delete(self.table).where(and_(*data_conditions))  # type: ignore
                        result = session.execute(stmt)
                        session.commit()
                        return result.rowcount
                
                return 0
                
        except Exception as e:
            raise SQLCacheError(f"Error invalidating cache: {e}") from e
    
    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries.
        
        Returns:
            int: Number of expired entries removed
        """
        if 'expires_at' not in self.table.c:
            return 0
            
        try:
            with self.Session() as session:
                stmt = delete(self.table).where(  # type: ignore
                    self.table.c.expires_at < func.now()  # type: ignore
                )
                result = session.execute(stmt)
                session.commit()
                return result.rowcount
                
        except Exception as e:
            raise SQLCacheError(f"Error cleaning up expired entries: {e}") from e
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict[str, Any]: Statistics about the cache
        """
        try:
            with self.Session() as session:
                # Total records
                total_records = session.execute(
                    select(func.count()).select_from(self.table)  # type: ignore
                ).scalar()
                
                stats = {
                    'total_records': total_records,
                    'table_name': self.table.name,
                    'primary_keys': self.primary_key_cols
                }
                
                # Expired records count
                if 'expires_at' in self.table.c:
                    expired_count = session.execute(
                        select(func.count()).select_from(self.table)  # type: ignore
                        .where(self.table.c.expires_at < func.now())  # type: ignore
                    ).scalar()
                    stats['expired_records'] = expired_count
                
                # Latest cache timestamp
                if 'cached_at' in self.table.c:
                    latest_cache = session.execute(
                        select(func.max(self.table.c.cached_at))  # type: ignore
                    ).scalar()
                    stats['latest_cached_at'] = latest_cache
                
                return stats
                
        except Exception as e:
            raise SQLCacheError(f"Error getting cache stats: {e}") from e
    
    def clear_cache(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            int: Number of entries removed
        """
        try:
            with self.Session() as session:
                result = session.execute(delete(self.table))  # type: ignore
                session.commit()
                return result.rowcount
                
        except Exception as e:
            raise SQLCacheError(f"Error clearing cache: {e}") from e
    
    def close(self):
        """Close the database connection"""
        if hasattr(self, 'engine'):
            self.engine.dispose()
