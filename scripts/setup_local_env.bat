@echo off
REM Setup script for local test environment on Windows

setlocal enabledelayedexpansion

echo.
echo ============================================================
echo.  Cacheness Local Test Environment Setup (Windows)
echo.
echo ============================================================
echo.

REM Check Docker
echo Checking Docker installation...
docker --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker not found or not running
    echo Install from: https://www.docker.com/products/docker-desktop
    exit /b 1
)
echo [OK] Docker is installed

REM Check Docker Compose
echo Checking Docker Compose installation...
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker Compose not found
    echo Docker Desktop includes Compose automatically
    exit /b 1
)
echo [OK] Docker Compose is installed

REM Start services
echo.
echo Starting Docker Compose services...
docker-compose up -d
if errorlevel 1 (
    echo ERROR: Failed to start services
    exit /b 1
)

REM Wait for services
echo.
echo Waiting for services to become healthy (max 60 seconds)...
setlocal enabledelayedexpansion
set "attempts=0"
set "max_attempts=30"

:wait_loop
if !attempts! geq !max_attempts! (
    echo ERROR: Services did not become healthy in time
    echo.
    echo Troubleshooting:
    echo   docker-compose logs postgres
    echo   docker-compose logs minio
    exit /b 1
)

docker-compose ps | findstr "healthy" >nul 2>&1
if errorlevel 1 (
    set /a "attempts=!attempts!+1"
    echo   Attempt !attempts!/!max_attempts! ...
    timeout /t 2 /nobreak >nul
    goto wait_loop
)

REM Display info
echo.
echo ============================================================
echo.  Connection Information
echo.
echo ============================================================
echo.
echo PostgreSQL:
echo   Host:     localhost
echo   Port:     5432
echo   Database: cacheness_test
echo   User:     cacheness
echo   Password: cacheness_dev_password
echo   URL:      postgresql://cacheness:cacheness_dev_password@localhost:5432/cacheness_test
echo.
echo MinIO (S3-compatible):
echo   Endpoint: http://localhost:9000
echo   Console:  http://localhost:9001
echo   Access:   minioadmin
echo   Secret:   minioadmin
echo   Buckets:  cache-bucket, test-bucket
echo.
echo ============================================================
echo.  Setup Complete!
echo.
echo ============================================================
echo.
echo Next steps:
echo   1. Install Python dependencies:
echo      pip install -e ".[dev,s3,postgresql,cloud]"
echo.
echo   2. Run integration tests:
echo      pytest tests/test_postgres_s3_integration.py -v
echo.
echo   3. View MinIO console:
echo      http://localhost:9001
echo.
echo   4. Full guide: docs\LOCAL_TEST_ENVIRONMENT.md
echo.
echo To stop services:
echo   docker-compose down
echo.
echo To stop and remove data:
echo   docker-compose down -v
echo.

endlocal
