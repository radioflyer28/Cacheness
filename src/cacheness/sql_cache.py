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
    >>> from cacheness.sql_cache import SqlCache, SqlCacheAdapter
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
    >>> class StockAdapter(SqlCacheAdapter):
    ...     def get_table_definition(self):
    ...         return stock_table
    ...     
    ...     def fetch_data(self, **kwargs):
    ...         # Your API/database fetch logic here
    ...         return pandas_dataframe
    >>> 
    >>> # Create and use the cache
    >>> cache = SqlCache("stocks.db", stock_table, StockAdapter())
    >>> data = cache.get_data(symbol="AAPL", start_date="2024-01-01", end_date="2024-01-31")
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone, date
from typing import Any, Dict, List, Optional, Union, Callable, TYPE_CHECKING

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


class SqlCacheAdapter(ABC):
    """
    Abstract adapter for fetching data from external sources for SQL pull-through cache.
    
    This class defines the interface that data adapters must implement
    to work with the SQL pull-through cache system.
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


class SqlCache:
    """
    Pull-through cache implementation using SQL databases.
    
    This cache automatically fetches missing data from external sources
    and stores it in a local database for fast subsequent access.
    
    Supported Database Backends:
    - DuckDB: Optimized for analytical/columnar workloads with fast aggregations
    - SQLite: Optimized for transactional/row-wise operations with ACID guarantees
    - PostgreSQL: Full-featured RDBMS for production environments
    - MySQL: Alternative production RDBMS option
    """
    
    @classmethod
    def with_duckdb(
        cls,
        db_path: str,
        table: "Table",
        data_adapter: SqlCacheAdapter,
        ttl_hours: int = 24,
        time_increment: Optional[Union[str, timedelta, int]] = None,
        ordered_increment: Optional[Union[int, float]] = None,
        gap_detector: Optional[Callable] = None,
        **kwargs
    ):
        """
        Create a cache using DuckDB backend (optimized for analytical workloads).
        
        DuckDB is ideal for:
        - Time-series data analysis
        - Large datasets with aggregations
        - Columnar data processing
        - Fast analytical queries
        
        DuckDB Limitations:
        - No support for auto-incrementing Integer primary keys (SERIAL type)
        - Use composite primary keys or set autoincrement=False
        - Optimized for read-heavy analytical workloads
        
        Args:
            db_path: Path to DuckDB database file
            table: SQLAlchemy Table object (avoid autoincrement=True for primary keys)
            data_adapter: Data adapter for fetching external data
            ttl_hours: Cache TTL in hours
            time_increment: Optional time increment (e.g., timedelta(minutes=5), "5min")
            ordered_increment: Optional ordered data increment (e.g., 1, 10, 100)
            gap_detector: Optional custom gap detection function
            **kwargs: Additional arguments for SQLAlchemy engine
            
        Returns:
            SqlCache: Cache instance with DuckDB backend
            
        Example:
            >>> # Good: Composite primary key or no autoincrement
            >>> table = Table('data', metadata,
            ...     Column('symbol', String(10), primary_key=True),
            ...     Column('date', Date, primary_key=True),
            ...     Column('value', Float)
            ... )
            >>> cache = SqlCache.with_duckdb("data.db", table, adapter, 
            ...                               time_increment=timedelta(minutes=5))
        """
        db_url = f"duckdb:///{db_path}"
        return cls(db_url, table, data_adapter, ttl_hours, 
                  engine_kwargs=kwargs.get('engine_kwargs'),
                  echo=kwargs.get('echo', False),
                  time_increment=time_increment, 
                  ordered_increment=ordered_increment,
                  gap_detector=gap_detector)
    
    @classmethod
    def with_sqlite(
        cls,
        db_path: str,
        table: "Table", 
        data_adapter: SqlCacheAdapter,
        ttl_hours: int = 24,
        time_increment: Optional[Union[str, timedelta, int]] = None,
        ordered_increment: Optional[Union[int, float]] = None,
        gap_detector: Optional[Callable] = None,
        **kwargs
    ):
        """
        Create a cache using SQLite backend (optimized for transactional workloads).
        
        SQLite is ideal for:
        - Row-wise operations and updates
        - ACID transaction requirements
        - Concurrent read access
        - Simple deployment scenarios
        
        Args:
            db_path: Path to SQLite database file (use ":memory:" for in-memory)
            table: SQLAlchemy Table object
            data_adapter: Data adapter for fetching external data
            ttl_hours: Cache TTL in hours
            table: SQLAlchemy Table object
            data_adapter: Data adapter for fetching external data
            ttl_hours: Cache TTL in hours
            time_increment: Optional time increment (e.g., timedelta(minutes=5), "5min")
            ordered_increment: Optional ordered data increment (e.g., 1, 10, 100)
            **kwargs: Additional arguments for SQLAlchemy engine
            
        Returns:
            SqlCache: Cache instance with SQLite backend
        """
        if db_path == ":memory:":
            db_url = "sqlite:///:memory:"
        else:
            db_url = f"sqlite:///{db_path}"
        return cls(db_url, table, data_adapter, ttl_hours,
                  engine_kwargs=kwargs.get('engine_kwargs'),
                  echo=kwargs.get('echo', False),
                  time_increment=time_increment,
                  ordered_increment=ordered_increment,
                  gap_detector=gap_detector)
    
    @classmethod
    def with_postgresql(
        cls,
        connection_string: str,
        table: "Table",
        data_adapter: SqlCacheAdapter,
        ttl_hours: int = 24,
        time_increment: Optional[Union[str, timedelta, int]] = None,
        ordered_increment: Optional[Union[int, float]] = None,
        gap_detector: Optional[Callable] = None,
        **kwargs
    ):
        """
        Create a cache using PostgreSQL backend (production-ready RDBMS).
        
        PostgreSQL is ideal for:
        - Production environments
        - Advanced SQL features
        - High concurrency
        - Complex data types
        
        Args:
            connection_string: PostgreSQL connection string
                             (e.g., "postgresql://user:pass@localhost/dbname")
            table: SQLAlchemy Table object
            data_adapter: Data adapter for fetching external data
            ttl_hours: Cache TTL in hours
            time_increment: Optional time increment (e.g., timedelta(minutes=5), "5min")
            ordered_increment: Optional ordered data increment (e.g., 1, 10, 100)
            **kwargs: Additional arguments for SQLAlchemy engine
            
        Returns:
            SqlCache: Cache instance with PostgreSQL backend
        """
        return cls(connection_string, table, data_adapter, ttl_hours,
                  engine_kwargs=kwargs.get('engine_kwargs'),
                  echo=kwargs.get('echo', False),
                  time_increment=time_increment,
                  ordered_increment=ordered_increment,
                  gap_detector=gap_detector)
    
    def __init__(
        self,
        db_url: str,
        table: "Table",
        data_adapter: SqlCacheAdapter,
        ttl_hours: int = 24,
        echo: bool = False,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        time_increment: Optional[Union[str, timedelta, int]] = None,
        ordered_increment: Optional[Union[int, float]] = None,
        gap_detector: Optional[Callable] = None
    ):
        """
        Initialize the SQL pull-through cache.
        
        Database Backend Selection:
        - Use class methods (.with_duckdb(), .with_sqlite(), .with_postgresql()) 
          for explicit backend selection
        - Or provide a full SQLAlchemy URL for custom configuration
        - Simple file paths default to DuckDB for analytical workloads
        
        Args:
            db_url: Database URL or file path
                   - "stocks.db" → DuckDB (analytical/columnar)
                   - "sqlite:///stocks.db" → SQLite (transactional/row-wise)
                   - "duckdb:///stocks.db" → DuckDB (explicit)
                   - "postgresql://user:pass@host/db" → PostgreSQL
            table: SQLAlchemy Table object defining the cache schema
            data_adapter: Adapter for fetching data from external sources
            ttl_hours: Cache TTL in hours (0 for no expiration)
            echo: Whether to echo SQL statements for debugging
            engine_kwargs: Additional arguments for SQLAlchemy engine
            time_increment: Optional time increment for time-series data:
                          - timedelta(minutes=5) for 5-minute intervals
                          - "5min", "30sec", "2hour" for common intervals
                          - 300 (seconds) for numeric specification
                          - None to auto-detect from data (default)
            ordered_increment: Optional increment for ordered data (e.g., 1, 10, 100)
                             - None to auto-detect from data (default)
            gap_detector: Optional custom gap detection function. If provided, this function
                        will be called instead of the built-in gap detection logic.
                        Signature: gap_detector(query_params, cached_data, cache_instance) -> List[Dict]
                        Should return a list of parameter dictionaries for missing data ranges.
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
        
        # Store user-specified increments for intelligent gap detection
        self.time_increment = self._parse_time_increment(time_increment)
        self.ordered_increment = ordered_increment
        self.gap_detector = gap_detector
        
        # Set up table with cache metadata columns
        self.table = table
        self._add_cache_columns()
        
        # Handle DuckDB-specific compatibility issues
        self._handle_duckdb_compatibility()
        
        # Create table in database
        self.table.create(self.engine, checkfirst=True)
        
        # Cache table information for performance
        self._setup_query_helpers()
    
    def _parse_time_increment(self, time_increment: Optional[Union[str, timedelta, int]]) -> Optional[timedelta]:
        """Parse user-provided time increment into timedelta object."""
        if time_increment is None:
            return None
        
        if isinstance(time_increment, timedelta):
            return time_increment
        
        if isinstance(time_increment, (int, float)):
            # Assume seconds
            return timedelta(seconds=time_increment)
        
        if isinstance(time_increment, str):
            # Parse string formats like "5min", "30sec", "2hour"
            time_increment = time_increment.lower().strip()
            
            # Extract number and unit
            import re
            match = re.match(r'^(\d+(?:\.\d+)?)([a-z]+)$', time_increment)
            if not match:
                raise ValueError(f"Invalid time increment format: {time_increment}")
            
            value, unit = match.groups()
            value = float(value)
            
            # Map units to timedelta arguments
            unit_map = {
                'sec': {'seconds': value},
                'second': {'seconds': value},
                'seconds': {'seconds': value},
                'min': {'minutes': value},
                'minute': {'minutes': value},
                'minutes': {'minutes': value},
                'hour': {'hours': value},
                'hours': {'hours': value},
                'day': {'days': value},
                'days': {'days': value},
                'week': {'weeks': value},
                'weeks': {'weeks': value},
            }
            
            if unit not in unit_map:
                raise ValueError(f"Unsupported time unit: {unit}")
            
            return timedelta(**unit_map[unit])
        
        raise ValueError(f"Unsupported time increment type: {type(time_increment)}")
    
    def _handle_duckdb_compatibility(self):
        """
        Handle DuckDB-specific compatibility issues.
        
        DuckDB Limitations:
        - No SERIAL type support (auto-incrementing columns)
        - Limited support for some SQLAlchemy features
        - Optimized for analytical/read-heavy workloads
        """
        if self.engine.dialect.name == 'duckdb':
            # Check for potential compatibility issues
            for column in self.table.columns:
                if (hasattr(column.type, 'python_type') and 
                    column.type.python_type is int and 
                    column.primary_key and 
                    getattr(column, 'autoincrement', None) is not False):
                    
                    import warnings
                    warnings.warn(
                        f"Column '{column.name}' may cause issues with DuckDB due to "
                        "auto-incrementing Integer primary keys. DuckDB doesn't support "
                        "SERIAL types. Consider using composite primary keys or "
                        "explicitly set autoincrement=False for DuckDB compatibility.",
                        UserWarning,
                        stacklevel=3
                    )
    
    def _add_cache_columns(self):
        """Add cache metadata columns to the table if not present"""
        if not HAS_SQLALCHEMY:
            return
            
        existing_columns = {col.name for col in self.table.columns}
        
        # Add cached_at timestamp (UTC timezone-aware)
        if 'cached_at' not in existing_columns:
            self.table.append_column(
                Column('cached_at', DateTime(timezone=True), nullable=False)  # type: ignore
            )
        
        # Add expires_at timestamp if TTL is configured (UTC timezone-aware)
        if 'expires_at' not in existing_columns and self.ttl_hours > 0:
            self.table.append_column(
                Column('expires_at', DateTime(timezone=True), nullable=True)  # type: ignore
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
        
        # Add TTL condition if expires_at column exists (using UTC)
        if 'expires_at' in self.table.c:
            current_utc = datetime.now(timezone.utc)
            conditions.append(
                or_(  # type: ignore
                    self.table.c.expires_at.is_(None),
                    self.table.c.expires_at > current_utc  # type: ignore
                )
            )
        
        return conditions
    
    def _store_in_cache(self, session: Session, data: 'pd.DataFrame'):
        """Store data in cache with upsert behavior"""
        if data.empty:
            return
        
        # Add cache metadata (using UTC timestamps)
        data = data.copy()
        if 'cached_at' in self.table.c:
            data['cached_at'] = datetime.now(timezone.utc)
        if 'expires_at' in self.table.c and self.ttl_hours > 0:
            data['expires_at'] = datetime.now(timezone.utc) + timedelta(hours=self.ttl_hours)
        
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
        Find missing data ranges by analyzing query parameters vs cached data.
        
        This implementation provides multiple ways to customize gap detection:
        
        1. Custom Gap Detector Function: Pass a gap_detector function to the constructor
           that implements your specific logic for finding missing data ranges.
           
        2. Subclass Override: Override this method in a subclass for complex scenarios
           that need access to the full SqlCache instance state.
           
        3. Built-in Intelligence: Use the sophisticated built-in gap detection that
           handles time-series data, ordered data, arbitrary increments, etc.
        
        Args:
            query_params: Parsed query parameters from the adapter
            cached_data: Current cached data matching the query
            
        Returns:
            List of parameter dictionaries for fetching missing data ranges
        """
        # Use custom gap detector if provided
        if self.gap_detector is not None:
            try:
                return self.gap_detector(query_params, cached_data, self)
            except Exception as e:
                # Fallback to built-in logic if custom detector fails
                print(f"Warning: Custom gap detector failed ({e}), using built-in logic")
        
        # Use built-in intelligent gap detection
        if cached_data.empty:
            # No cached data - fetch everything
            return [self._convert_query_to_fetch_params(query_params)]
        
        # Check if cached data covers the requested query parameters
        missing_ranges = self._detect_range_gaps(query_params, cached_data)
        
        if missing_ranges:
            return [self._convert_query_to_fetch_params(params) for params in missing_ranges]
        
        # All requested data is already cached
        return []
    
    def _detect_range_gaps(self, query_params: Dict[str, Any], cached_data: 'pd.DataFrame') -> List[Dict[str, Any]]:
        """
        Intelligent gap detection for time series data with overlapping and non-overlapping ranges.
        
        This implementation:
        1. Identifies the data granularity (hourly, daily, etc.)
        2. Detects gaps within cached data
        3. Handles overlapping and adjacent ranges intelligently
        4. Adds safety margins to avoid accidental gaps
        5. Consolidates nearby ranges to minimize fetch calls
        
        Returns list of missing range parameters that need to be fetched.
        """
        missing_ranges = []
        
        # Find the primary time/date column for gap analysis
        column_info = self._identify_time_column(query_params, cached_data)
        if not column_info:
            # No ordered column found - use simple overlap detection
            return self._simple_overlap_detection(query_params, cached_data)
        
        column_name = column_info['column']
        column_type = column_info['column_type']
        requested_start = column_info['requested_start']
        requested_end = column_info['requested_end']
        
        # Analyze cached data coverage
        cache_analysis = self._analyze_cached_data(cached_data, column_name, query_params, column_type)
        
        if not cache_analysis['has_data']:
            # No relevant cached data - fetch the requested range with margin
            if column_type == 'time':
                expanded_range = self._add_safety_margin(
                    requested_start, requested_end, cache_analysis['granularity']
                )
            else:
                # For ordered data, add small buffer
                expanded_range = self._add_ordered_margin(requested_start, requested_end)
            return [self._create_fetch_params(query_params, column_name, expanded_range)]
        
        # Detect gaps and missing ranges
        gaps = self._detect_gaps(
            requested_start, requested_end,
            cache_analysis['cached_start'], cache_analysis['cached_end'],
            column_type, cache_analysis.get('granularity', 'day'), 
            cached_data, column_name, query_params
        )
        
        # Consolidate nearby gaps to minimize fetch calls
        if column_type == 'time':
            consolidated_gaps = self._consolidate_gaps(gaps, cache_analysis.get('granularity', 'day'))
        else:
            consolidated_gaps = self._consolidate_ordered_gaps(gaps)
        
        # Convert gaps to fetch parameters
        for gap_start, gap_end in consolidated_gaps:
            if column_type == 'time':
                expanded_range = self._add_safety_margin(gap_start, gap_end, cache_analysis.get('granularity', 'day'))
            else:
                expanded_range = self._add_ordered_margin(gap_start, gap_end)
            missing_ranges.append(self._create_fetch_params(query_params, column_name, expanded_range))
        
        return missing_ranges
    
    def _identify_time_column(self, query_params: Dict[str, Any], cached_data: 'pd.DataFrame') -> Optional[Dict[str, Any]]:
        """Identify the primary time/date column and extract range information."""
        for param_name, param_value in query_params.items():
            if not isinstance(param_value, dict):
                continue
                
            if 'start' not in param_value or 'end' not in param_value:
                continue
            
            # Check if this looks like a time column
            if any(keyword in param_name.lower() for keyword in ['date', 'time', 'timestamp']):
                return {
                    'column': param_name,
                    'requested_start': param_value['start'],
                    'requested_end': param_value['end'],
                    'column_type': 'time'
                }
                
            # If column exists in cached data, check if it contains time-like data
            if param_name in cached_data.columns and not cached_data.empty:
                sample_value = cached_data[param_name].iloc[0]
                if self._is_time_like(sample_value):
                    return {
                        'column': param_name,
                        'requested_start': param_value['start'],
                        'requested_end': param_value['end'],
                        'column_type': 'time'
                    }
                    
            # Check if this looks like an ordered/incrementing column (order_id, sequence, etc.)
            if any(keyword in param_name.lower() for keyword in ['order', 'id', 'sequence', 'number', 'index']):
                return {
                    'column': param_name,
                    'requested_start': param_value['start'],
                    'requested_end': param_value['end'],
                    'column_type': 'ordered'
                }
                
            # If column exists and contains numeric/ordered data, treat as ordered
            if param_name in cached_data.columns and not cached_data.empty:
                sample_value = cached_data[param_name].iloc[0]
                if self._is_ordered_like(sample_value):
                    return {
                        'column': param_name,
                        'requested_start': param_value['start'],
                        'requested_end': param_value['end'],
                        'column_type': 'ordered'
                    }
        
        return None
    
    def _is_time_like(self, value) -> bool:
        """Check if a value looks like a time/date."""
        return isinstance(value, (date, datetime)) or str(type(value)).find('datetime') != -1
    
    def _is_ordered_like(self, value) -> bool:
        """Check if a value looks like an ordered/incrementing value (int, float, etc.)."""
        return isinstance(value, (int, float)) or str(type(value)).find('int') != -1 or str(type(value)).find('float') != -1
    
    def _analyze_cached_data(self, cached_data: 'pd.DataFrame', column_name: str, query_params: Dict[str, Any], column_type: str = 'time') -> Dict[str, Any]:
        """Analyze cached data to understand granularity and coverage."""
        
        # Filter cached data to match other query parameters (exclude the ordered column)
        filtered_cache = self._filter_cached_data_by_non_time_params(cached_data, query_params, column_name)
        
        if filtered_cache.empty:
            return {
                'has_data': False,
                'granularity': 'day' if column_type == 'time' else 'unit',
                'cached_start': None,
                'cached_end': None
            }
        
        # Get the actual range of cached data
        cached_start = filtered_cache[column_name].min()
        cached_end = filtered_cache[column_name].max()
        
        # Detect data granularity
        if column_type == 'time':
            granularity = self._detect_granularity(filtered_cache, column_name)
        else:
            granularity = self._detect_ordered_granularity(filtered_cache, column_name)
        
        return {
            'has_data': True,
            'granularity': granularity,
            'cached_start': cached_start,
            'cached_end': cached_end,
            'filtered_data': filtered_cache
        }
    
    def _filter_cached_data_by_non_time_params(self, cached_data: 'pd.DataFrame', query_params: Dict[str, Any], time_column: str) -> 'pd.DataFrame':
        """Filter cached data by non-time parameters to get relevant subset."""
        filtered = cached_data.copy()
        
        for param_name, param_value in query_params.items():
            if param_name == time_column:
                continue  # Skip time column
                
            if param_name not in cached_data.columns:
                continue
                
            if isinstance(param_value, dict):
                # Skip complex conditions for now - could be enhanced later
                continue
            elif isinstance(param_value, (list, tuple)):
                # IN clause
                filtered = filtered[filtered[param_name].isin(param_value)]
            else:
                # Exact match
                filtered = filtered[filtered[param_name] == param_value]
        
        return filtered
    
    def _detect_granularity(self, data: 'pd.DataFrame', time_column: str) -> Union[str, timedelta]:
        """Detect the time granularity of the data (hour, day, week, etc.) or use user-specified increment."""
        
        # If user provided a time increment, use it directly
        if self.time_increment is not None:
            return self.time_increment
        
        # Auto-detect from data
        if len(data) < 2:
            return 'day'  # default
        
        # Sort by time column and calculate differences
        sorted_data = data.sort_values(time_column)
        time_values = sorted_data[time_column].values
        
        if len(time_values) < 2:
            return 'day'
        
        # Calculate time differences
        try:
            # Try to get time differences
            if hasattr(time_values[0], '__sub__'):
                # Direct subtraction for datetime objects
                time_diffs = [time_values[i] - time_values[i-1] for i in range(1, len(time_values))]
                
                # Find the most common difference (mode)
                if time_diffs:
                    # Calculate actual timedelta differences
                    valid_diffs = [diff for diff in time_diffs if diff is not None]
                    if valid_diffs:
                        # Use the most frequent difference
                        from collections import Counter
                        diff_counter = Counter(valid_diffs)
                        most_common_diff = diff_counter.most_common(1)[0][0]
                        
                        # Return the actual timedelta for precise gap detection
                        if hasattr(most_common_diff, 'total_seconds'):
                            return most_common_diff
            
            # Fallback to legacy string-based detection
            return self._legacy_granularity_detection(data, time_column)
                
        except Exception:
            # If anything goes wrong, default to daily
            return 'day'
    
    def _legacy_granularity_detection(self, data: 'pd.DataFrame', time_column: str) -> str:
        """Legacy granularity detection for backward compatibility."""
        try:
            sorted_data = data.sort_values(time_column)
            time_values = sorted_data[time_column].values
            
            if len(time_values) < 2:
                return 'day'
            
            time_diffs = [time_values[i] - time_values[i-1] for i in range(1, len(time_values))]
            most_common_diff = time_diffs[0] if time_diffs else None
            
            if most_common_diff is None:
                return 'day'
            
            # Convert to timedelta for analysis
            if hasattr(most_common_diff, 'days'):
                days = most_common_diff.days
                seconds = getattr(most_common_diff, 'seconds', 0)
            else:
                # Try to convert to numeric days
                try:
                    days = float(most_common_diff)
                    seconds = 0
                except (ValueError, TypeError):
                    return 'day'
            
            # Classify granularity based on time intervals
            total_seconds = days * 86400 + seconds
            
            if total_seconds <= 60:         # <= 1 minute
                return 'minute'
            elif total_seconds <= 3600:     # <= 1 hour  
                return 'hour'
            elif total_seconds <= 86400:    # <= 1 day
                return 'day'
            elif 6 * 86400 <= total_seconds <= 8 * 86400:  # Around a week
                return 'week'
            elif 28 * 86400 <= total_seconds <= 32 * 86400:  # Around a month
                return 'month'
            else:
                return 'day'  # default fallback
                
        except Exception:
            return 'day'
    
    def _detect_gaps(self, requested_start, requested_end, cached_start, cached_end, 
                     column_type: str, granularity: str, cached_data: 'pd.DataFrame', 
                     column_name: str, query_params: Dict[str, Any]) -> List[tuple]:
        """Detect specific gaps that need to be fetched (works for time and ordered data)."""
        gaps = []
        
        # Gap before cached data
        if requested_start < cached_start:
            gaps.append((requested_start, min(requested_end, cached_start)))
        
        # Gap after cached data  
        if requested_end > cached_end:
            gaps.append((max(requested_start, cached_end), requested_end))
        
        # For ordered data, we might want to check for internal gaps
        # For time data, gaps within cached data range (missing data points)
        if requested_start < cached_end and requested_end > cached_start:
            if column_type == 'time':
                internal_gaps = self._find_internal_gaps(
                    max(requested_start, cached_start),
                    min(requested_end, cached_end),
                    cached_data, column_name, granularity, query_params
                )
                gaps.extend(internal_gaps)
            # For ordered data, we assume continuous ranges for now
            # Could be enhanced to detect missing order IDs within ranges
        
        return gaps
    
    def _detect_ordered_granularity(self, data: 'pd.DataFrame', column_name: str) -> Union[str, float]:
        """Detect the granularity of ordered data (unit increment, batch size, etc.) or use user-specified increment."""
        
        # If user provided an ordered increment, use it directly
        if self.ordered_increment is not None:
            return self.ordered_increment
        
        # Auto-detect from data
        if len(data) < 2:
            return 'unit'  # default for ordered data
        
        # Sort by the ordered column and calculate differences
        sorted_data = data.sort_values(column_name)
        values = sorted_data[column_name].values
        
        if len(values) < 2:
            return 'unit'
        
        try:
            # Calculate differences between consecutive values
            diffs = [values[i] - values[i-1] for i in range(1, len(values))]
            
            # Find the most common difference (mode)
            if diffs:
                from collections import Counter
                diff_counter = Counter(diffs)
                most_common_diff = diff_counter.most_common(1)[0][0]
                
                # Return the actual increment value for precise gap detection
                if isinstance(most_common_diff, (int, float)) and most_common_diff > 0:
                    return float(most_common_diff)
                
                # Fallback to legacy classification
                avg_diff = sum(diffs) / len(diffs)
                
                if avg_diff <= 1:
                    return 'unit'      # Increment by 1
                elif avg_diff <= 10:
                    return 'batch_10'  # Increment by ~10
                elif avg_diff <= 100:
                    return 'batch_100' # Increment by ~100
                else:
                    return 'batch_large' # Large increments
            
            return 'unit'
            
        except Exception:
            return 'unit'
    
    def _add_ordered_margin(self, start, end) -> tuple:
        """Add safety margins for ordered data to avoid gaps."""
        try:
            # Add small margin for ordered data
            margin = max(1, int((end - start) * 0.01))  # 1% margin, minimum 1
            safe_start = start - margin
            safe_end = end + margin
            return (safe_start, safe_end)
        except (TypeError, ValueError):
            # If we can't calculate margin, return original range
            return (start, end)
    
    def _consolidate_ordered_gaps(self, gaps: List[tuple]) -> List[tuple]:
        """Consolidate nearby gaps for ordered data to minimize fetch calls."""
        if not gaps:
            return []
        
        # Sort gaps by start value
        sorted_gaps = sorted(gaps, key=lambda x: x[0])
        consolidated = []
        
        current_start, current_end = sorted_gaps[0]
        
        for gap_start, gap_end in sorted_gaps[1:]:
            try:
                # For ordered data, consolidate if gaps are close (within 20% of range)
                gap_size = gap_start - current_end
                range_size = current_end - current_start
                
                if gap_size <= max(10, range_size * 0.2):  # Consolidate if close
                    current_end = max(current_end, gap_end)
                else:
                    consolidated.append((current_start, current_end))
                    current_start, current_end = gap_start, gap_end
            except (TypeError, ValueError):
                # If we can't calculate, don't consolidate
                consolidated.append((current_start, current_end))
                current_start, current_end = gap_start, gap_end
        
        consolidated.append((current_start, current_end))
        return consolidated
    
    def _find_internal_gaps(self, start, end, cached_data: 'pd.DataFrame', 
                           time_column: str, granularity: str, query_params: Dict[str, Any]) -> List[tuple]:
        """Find gaps within the cached data range."""
        # This is simplified - in practice, you might want to:
        # 1. Generate expected time series based on granularity
        # 2. Compare with actual cached data
        # 3. Identify missing time points
        # 4. Group consecutive missing points into ranges
        
        # For now, assume no internal gaps (conservative approach)
        # Real implementation would check for missing data points
        return []
    
    def _add_safety_margin(self, start, end, granularity: Union[str, timedelta, float]) -> tuple:
        """Add safety margins to avoid accidental gaps."""
        from datetime import timedelta
        
        # Determine margin based on granularity type
        if isinstance(granularity, timedelta):
            # Use the user-specified time increment
            margin = granularity
        elif isinstance(granularity, (int, float)):
            # Ordered data - add a small numeric margin
            margin_size = max(1, int(granularity * 0.1))  # 10% of increment, minimum 1
            safe_start = start - margin_size if isinstance(start, (int, float)) else start
            safe_end = end + margin_size if isinstance(end, (int, float)) else end
            return (safe_start, safe_end)
        else:
            # Legacy string-based granularity
            margin_map = {
                'minute': timedelta(minutes=1),
                'hour': timedelta(hours=1),
                'day': timedelta(days=1), 
                'week': timedelta(days=1),
                'month': timedelta(days=2),
                'unit': 1,          # For ordered data
                'batch_10': 2,      # Small margin
                'batch_100': 10,    # Medium margin
                'batch_large': 50   # Large margin
            }
            
            margin = margin_map.get(granularity, timedelta(days=1))
            
            # Handle ordered data margins
            if isinstance(margin, (int, float)):
                safe_start = start - margin if isinstance(start, (int, float)) else start
                safe_end = end + margin if isinstance(end, (int, float)) else end
                return (safe_start, safe_end)
        
        # Add time-based margin
        safe_start = start - margin if hasattr(start, '__sub__') else start
        safe_end = end + margin if hasattr(end, '__add__') else end
        
        return (safe_start, safe_end)
    
    def _consolidate_gaps(self, gaps: List[tuple], granularity: Union[str, timedelta, float]) -> List[tuple]:
        """Consolidate nearby gaps to minimize fetch calls."""
        if not gaps:
            return []
        
        # Sort gaps by start time
        sorted_gaps = sorted(gaps, key=lambda x: x[0])
        consolidated = []
        
        current_start, current_end = sorted_gaps[0]
        
        for gap_start, gap_end in sorted_gaps[1:]:
            # Check if gaps are close enough to consolidate
            if self._should_consolidate_gaps(current_end, gap_start, granularity):
                # Extend current gap
                current_end = max(current_end, gap_end)
            else:
                # Start new gap
                consolidated.append((current_start, current_end))
                current_start, current_end = gap_start, gap_end
        
        consolidated.append((current_start, current_end))
        return consolidated
    
    def _should_consolidate_gaps(self, end1, start2, granularity: Union[str, timedelta, float]) -> bool:
        """Determine if two gaps should be consolidated."""
        from datetime import timedelta
        
        try:
            # Calculate gap size
            gap_size = start2 - end1
            
            # Determine threshold based on granularity type
            if isinstance(granularity, timedelta):
                # Use 3x the time increment as threshold
                threshold = granularity * 3
                return gap_size <= threshold
            elif isinstance(granularity, (int, float)):
                # For ordered data, consolidate if gap is small relative to increment
                threshold = granularity * 3
                return gap_size <= threshold
            else:
                # Legacy string-based thresholds
                threshold_map = {
                    'minute': timedelta(minutes=10),
                    'hour': timedelta(hours=6),    # Consolidate if < 6 hours apart
                    'day': timedelta(days=3),      # Consolidate if < 3 days apart
                    'week': timedelta(days=7),     # Consolidate if < 1 week apart
                    'month': timedelta(days=14),   # Consolidate if < 2 weeks apart
                    'unit': 5,                     # For ordered data
                    'batch_10': 30,
                    'batch_100': 300,
                    'batch_large': 1000
                }
                
                threshold = threshold_map.get(granularity, timedelta(days=3))
                
                # Handle ordered data thresholds
                if isinstance(threshold, (int, float)):
                    return gap_size <= threshold
                else:
                    return gap_size <= threshold
                    
        except (TypeError, AttributeError):
            # Can't calculate difference - don't consolidate
            return False
    
    def _create_fetch_params(self, query_params: Dict[str, Any], time_column: str, time_range: tuple) -> Dict[str, Any]:
        """Create fetch parameters for a specific time range."""
        fetch_params = {}
        
        for key, value in query_params.items():
            if key == time_column:
                # Replace with the specific time range
                fetch_params[f'{key}_start'] = time_range[0]
                fetch_params[f'{key}_end'] = time_range[1]
            elif isinstance(value, dict):
                if 'start' in value and 'end' in value:
                    fetch_params[f'{key}_start'] = value['start']
                    fetch_params[f'{key}_end'] = value['end']
                else:
                    fetch_params[key] = value
            else:
                fetch_params[key] = value
        
        return fetch_params
    
    def _simple_overlap_detection(self, query_params: Dict[str, Any], cached_data: 'pd.DataFrame') -> List[Dict[str, Any]]:
        """Fallback to simple overlap detection when no time column is identified."""
        # Check each range parameter in the query
        for param_name, param_value in query_params.items():
            if not isinstance(param_value, dict):
                continue
                
            if 'start' not in param_value or 'end' not in param_value:
                continue
                
            # Check if this column exists in cached data
            if param_name not in cached_data.columns:
                # Column doesn't exist in cache - need to fetch
                return [query_params]
                
            # Get the range of cached data for this column
            cached_min = cached_data[param_name].min()
            cached_max = cached_data[param_name].max()
            
            requested_start = param_value['start']
            requested_end = param_value['end']
            
            # Check if requested range is completely outside cached range
            if requested_end < cached_min or requested_start > cached_max:
                # Non-overlapping range - need to fetch
                return [query_params]
                
            # Check if requested range extends beyond cached range
            if requested_start < cached_min or requested_end > cached_max:
                # Partial overlap - fetch the entire requested range for safety
                return [query_params]
        
        return []
    
    def _convert_query_to_fetch_params(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert internal query parameters to fetch_data format."""
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
        
        return fetch_params
    
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
                current_utc = datetime.now(timezone.utc)
                stmt = delete(self.table).where(  # type: ignore
                    self.table.c.expires_at < current_utc  # type: ignore
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
                
                # Expired records count (using UTC)
                if 'expires_at' in self.table.c:
                    current_utc = datetime.now(timezone.utc)
                    expired_count = session.execute(
                        select(func.count()).select_from(self.table)  # type: ignore
                        .where(self.table.c.expires_at < current_utc)  # type: ignore
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

    @classmethod
    def for_timeseries(
        cls,
        db_path: str,
        table_name: str = "timeseries_cache",
        symbol_column: str = "symbol",
        date_column: str = "date", 
        data_fetcher: Optional[Callable] = None,
        ttl_hours: int = 24,
        **kwargs
    ) -> "SqlCache":
        """
        Create a SQL cache for analytical timeseries data (uses DuckDB).
        
        Optimized for:
        - Historical analysis and backtesting
        - Aggregations over time ranges
        - Bulk timeseries processing
        - Data science and reporting workloads
        
        Args:
            db_path: Database file path (will use DuckDB backend)
            table_name: Name for the cache table
            symbol_column: Column name for the symbol/ticker
            date_column: Column name for the date
            data_fetcher: Function that fetches data (symbol, start_date, end_date) -> DataFrame
            ttl_hours: Cache TTL in hours
            **kwargs: Additional columns as name:type pairs (e.g. price=Float, volume=Integer)
            
        Example:
            def fetch_stock_data(symbol, start_date, end_date):
                # Your API call here - returns large datasets
                return pd.DataFrame(...)
            
            cache = SqlCache.for_timeseries(
                "stocks.db",  # Uses DuckDB for analytics
                data_fetcher=fetch_stock_data,
                price=Float,
                volume=Integer
            )
        """
        from sqlalchemy import Table, Column, String, Date, MetaData
        
        # Force DuckDB for analytical timeseries (default behavior for simple paths)
        if db_path.startswith('sqlite://'):
            # If user explicitly wants SQLite, we respect that but warn
            import warnings
            warnings.warn(
                "SQLite specified for analytical timeseries. Consider for_realtime_timeseries() "
                "for real-time lookups or use DuckDB for better analytical performance.",
                UserWarning
            )
        
        # Create table schema
        metadata = MetaData()
        columns = [
            Column(symbol_column, String(20), primary_key=True),
            Column(date_column, Date, primary_key=True),
        ]
        
        # Add custom columns from kwargs
        for col_name, col_type in kwargs.items():
            if col_name not in ['ttl_hours', 'data_fetcher']:
                columns.append(Column(col_name, col_type))
        
        table = Table(table_name, metadata, *columns)
        
        # Create simple adapter using the function
        class SimpleTimeseriesAdapter(SqlCacheAdapter):
            def get_table_definition(self):
                return table
                
            def fetch_data(self, **fetch_kwargs):
                if data_fetcher is None:
                    raise ValueError("data_fetcher function is required")
                return data_fetcher(**fetch_kwargs)
                
            def parse_query_params(self, **query_kwargs):
                return query_kwargs
        
        return cls(db_path, table, SimpleTimeseriesAdapter(), ttl_hours=ttl_hours)

    @classmethod
    def for_realtime_timeseries(
        cls,
        db_path: str,
        table_name: str = "realtime_timeseries_cache",
        symbol_column: str = "symbol",
        date_column: str = "date", 
        data_fetcher: Optional[Callable] = None,
        ttl_hours: int = 1,  # Shorter TTL for real-time data
        **kwargs
    ) -> "SqlCache":
        """
        Create a SQL cache for real-time timeseries data (uses SQLite).
        
        Optimized for:
        - Recent data lookups (last hour, day, week)
        - Real-time updates and streaming data
        - Quick individual symbol/date queries
        - Low-latency transactional access
        
        Args:
            db_path: Database file path (will use SQLite backend)
            table_name: Name for the cache table
            symbol_column: Column name for the symbol/ticker
            date_column: Column name for the date
            data_fetcher: Function that fetches data (symbol, start_date, end_date) -> DataFrame
            ttl_hours: Cache TTL in hours (default 1hr for real-time)
            **kwargs: Additional columns as name:type pairs
            
        Example:
            def fetch_live_prices(symbol, start_date, end_date):
                # Your real-time API call - small, frequent updates
                return pd.DataFrame(...)
            
            cache = SqlCache.for_realtime_timeseries(
                "live_prices.db",  # Uses SQLite for fast lookups
                data_fetcher=fetch_live_prices,
                ttl_hours=1,  # Fresh data
                price=Float,
                bid=Float,
                ask=Float
            )
        """
        from sqlalchemy import Table, Column, String, Date, MetaData
        
        # Force SQLite backend for real-time access
        if not db_path.startswith(('sqlite://', 'postgresql://')):
            db_path = f"sqlite:///{db_path}"
        
        # Create table schema
        metadata = MetaData()
        columns = [
            Column(symbol_column, String(20), primary_key=True),
            Column(date_column, Date, primary_key=True),
        ]
        
        # Add custom columns from kwargs
        for col_name, col_type in kwargs.items():
            if col_name not in ['ttl_hours', 'data_fetcher']:
                columns.append(Column(col_name, col_type))
        
        table = Table(table_name, metadata, *columns)
        
        # Create simple adapter using the function
        class SimpleRealtimeAdapter(SqlCacheAdapter):
            def get_table_definition(self):
                return table
                
            def fetch_data(self, **fetch_kwargs):
                if data_fetcher is None:
                    raise ValueError("data_fetcher function is required")
                return data_fetcher(**fetch_kwargs)
                
            def parse_query_params(self, **query_kwargs):
                return query_kwargs
        
        return cls(db_path, table, SimpleRealtimeAdapter(), ttl_hours=ttl_hours)

    @classmethod  
    def for_lookup_table(
        cls,
        db_path: str,
        table_name: str = "lookup_cache",
        primary_keys: Optional[List[str]] = None,
        data_fetcher: Optional[Callable] = None,
        ttl_hours: int = 12,
        **columns
    ) -> "SqlCache":
        """
        Create a SQL cache for row-wise lookups (uses SQLite).
        
        Optimized for:
        - Individual record lookups by primary key
        - User profiles, product details, session data
        - Transactional access patterns
        - Real-time updates and small queries
        
        Args:
            db_path: Database file path (will use SQLite backend)
            table_name: Name for the cache table  
            primary_keys: List of column names to use as primary keys
            data_fetcher: Function that fetches data (**kwargs) -> DataFrame
            ttl_hours: Cache TTL in hours
            **columns: Column definitions as name=type pairs
            
        Example:
            def fetch_user_profile(user_id):
                return pd.DataFrame([{
                    'user_id': user_id,
                    'name': 'John Doe', 
                    'email': 'john@example.com'
                }])
            
            cache = SqlCache.for_lookup_table(
                "users.db",
                primary_keys=["user_id"],
                data_fetcher=fetch_user_profile,
                user_id=Integer,
                name=String(100),
                email=String(255)
            )
        """
        from sqlalchemy import Table, Column, MetaData
        
        if not columns:
            raise ValueError("At least one column must be specified")
        if primary_keys is None:
            primary_keys = [list(columns.keys())[0]]  # Use first column as PK
            
        # Force SQLite backend for row-wise access
        if not db_path.startswith(('sqlite://', 'postgresql://')):
            db_path = f"sqlite:///{db_path}"
            
        # Create table schema
        metadata = MetaData()
        table_columns = []
        
        for col_name, col_type in columns.items():
            is_primary = col_name in primary_keys
            table_columns.append(Column(col_name, col_type, primary_key=is_primary))
        
        table = Table(table_name, metadata, *table_columns)
        
        # Create simple adapter
        class SimpleLookupAdapter(SqlCacheAdapter):
            def get_table_definition(self):
                return table
                
            def fetch_data(self, **fetch_kwargs):
                if data_fetcher is None:
                    raise ValueError("data_fetcher function is required")
                return data_fetcher(**fetch_kwargs)
                
            def parse_query_params(self, **query_kwargs):
                return query_kwargs
        
        return cls(db_path, table, SimpleLookupAdapter(), ttl_hours=ttl_hours)

    @classmethod  
    def for_analytics_table(
        cls,
        db_path: str,
        table_name: str = "analytics_cache",
        primary_keys: Optional[List[str]] = None,
        data_fetcher: Optional[Callable] = None,
        ttl_hours: int = 12,
        **columns
    ) -> "SqlCache":
        """
        Create a SQL cache for analytical queries (uses DuckDB).
        
        Optimized for:
        - Aggregations, GROUP BY, analytical queries
        - Bulk data processing and filtering
        - Reporting and data science workloads
        - Large dataset operations
        
        Args:
            db_path: Database file path (will use DuckDB backend)
            table_name: Name for the cache table  
            primary_keys: List of column names to use as primary keys
            data_fetcher: Function that fetches data (**kwargs) -> DataFrame
            ttl_hours: Cache TTL in hours
            **columns: Column definitions as name=type pairs
            
        Example:
            def fetch_sales_data(start_date, end_date):
                return pd.DataFrame([...])  # Large dataset
            
            cache = SqlCache.for_analytics_table(
                "sales.db",
                primary_keys=["transaction_id"],
                data_fetcher=fetch_sales_data,
                transaction_id=Integer,
                product_id=Integer,
                amount=Float,
                sale_date=Date
            )
        """
        from sqlalchemy import Table, Column, MetaData
        
        if not columns:
            raise ValueError("At least one column must be specified")
        if primary_keys is None:
            primary_keys = [list(columns.keys())[0]]  # Use first column as PK
            
        # Force DuckDB backend for analytical access
        if not db_path.startswith(('duckdb://', 'postgresql://')):
            # Simple file path gets DuckDB
            if '://' not in db_path:
                pass  # Keep as simple path, defaults to DuckDB
            else:
                db_path = f"duckdb:///{db_path}"
            
        # Create table schema
        metadata = MetaData()
        table_columns = []
        
        for col_name, col_type in columns.items():
            is_primary = col_name in primary_keys
            table_columns.append(Column(col_name, col_type, primary_key=is_primary))
        
        table = Table(table_name, metadata, *table_columns)
        
        # Create simple adapter
        class SimpleAnalyticsAdapter(SqlCacheAdapter):
            def get_table_definition(self):
                return table
                
            def fetch_data(self, **fetch_kwargs):
                if data_fetcher is None:
                    raise ValueError("data_fetcher function is required")
                return data_fetcher(**fetch_kwargs)
                
            def parse_query_params(self, **query_kwargs):
                return query_kwargs
        
        return cls(db_path, table, SimpleAnalyticsAdapter(), ttl_hours=ttl_hours)

    @classmethod  
    def for_table(
        cls,
        db_path: str,
        table_name: str = "table_cache",
        primary_keys: Optional[List[str]] = None,
        data_fetcher: Optional[Callable] = None,
        ttl_hours: int = 12,
        **columns
    ) -> "SqlCache":
        """
        Create a SQL cache for generic tabular data (backward compatibility).
        
        This method is kept for backward compatibility. Consider using:
        - for_lookup_table() for row-wise access (SQLite)
        - for_analytics_table() for analytical queries (DuckDB)
        
        Args:
            db_path: Database file path or URL
            table_name: Name for the cache table  
            primary_keys: List of column names to use as primary keys
            data_fetcher: Function that fetches data (**kwargs) -> DataFrame
            ttl_hours: Cache TTL in hours
            **columns: Column definitions as name=type pairs
        """
        import warnings
        warnings.warn(
            "for_table() is deprecated. Use for_lookup_table() for row-wise access "
            "or for_analytics_table() for analytical queries.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Default to lookup table behavior (SQLite)
        return cls.for_lookup_table(
            db_path, table_name, primary_keys, data_fetcher, ttl_hours, **columns
        )


# Backward compatibility aliases
SQLAlchemyDataAdapter = SqlCacheAdapter
