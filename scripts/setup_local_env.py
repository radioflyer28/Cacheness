#!/usr/bin/env python3
"""
Interactive setup script for local test environment.
Helps configure PostgreSQL and Garage containers.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str = "") -> bool:
    """Run a command and report status."""
    if description:
        print(f"\n{'=' * 60}")
        print(f"▶ {description}")
        print("=" * 60)

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"✗ Command failed: {' '.join(cmd)}")
        print(f"  Error: {e}")
        return False
    except FileNotFoundError:
        print(f"✗ Command not found: {cmd[0]}")
        return False


def check_docker() -> bool:
    """Check if Docker is installed and running."""
    print("\n" + "=" * 60)
    print("✓ Checking Docker installation...")
    print("=" * 60)

    try:
        subprocess.run(["docker", "--version"], capture_output=True, check=True)
        print("✓ Docker is installed")

        subprocess.run(["docker", "ps"], capture_output=True, check=True)
        print("✓ Docker daemon is running")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ Docker not found or not running")
        print("  Install Docker from: https://www.docker.com/products/docker-desktop")
        return False


def check_docker_compose() -> bool:
    """Check if Docker Compose is installed."""
    print("\n" + "=" * 60)
    print("✓ Checking Docker Compose installation...")
    print("=" * 60)

    try:
        subprocess.run(["docker-compose", "--version"], capture_output=True, check=True)
        print("✓ Docker Compose is installed")
        return True
    except FileNotFoundError:
        print("✗ Docker Compose not found")
        print("  Docker Desktop includes Compose automatically")
        return False


def start_services() -> bool:
    """Start PostgreSQL and Garage services."""
    print("\n" + "=" * 60)
    print("▶ Starting services...")
    print("=" * 60)

    # Check if already running
    try:
        result = subprocess.run(
            ["docker-compose", "ps"],
            capture_output=True,
            text=True,
            check=False,
        )
        if "cacheness-postgres" in result.stdout and "healthy" in result.stdout:
            print("✓ PostgreSQL is already running")
        if "cacheness-garage" in result.stdout and "healthy" in result.stdout:
            print("✓ Garage is already running")
    except FileNotFoundError:
        pass

    return run_command(
        ["docker-compose", "up", "-d"],
        "Starting Docker Compose services",
    )


def wait_for_services() -> bool:
    """Wait for services to be healthy."""
    print("\n" + "=" * 60)
    print("⏳ Waiting for services to become healthy...")
    print("=" * 60)

    max_attempts = 30
    attempt = 0

    while attempt < max_attempts:
        try:
            result = subprocess.run(
                ["docker-compose", "ps"],
                capture_output=True,
                text=True,
                check=False,
            )

            postgres_healthy = (
                "cacheness-postgres" in result.stdout and "healthy" in result.stdout
            )
            garage_healthy = (
                "cacheness-garage" in result.stdout and "healthy" in result.stdout
            )

            if postgres_healthy and garage_healthy:
                print("✓ PostgreSQL is healthy")
                print("✓ Garage is healthy")
                return True

            attempt += 1
            remaining = max_attempts - attempt
            print(
                f"  Attempt {attempt}/{max_attempts} ({remaining} remaining)...",
                end="\r",
            )

            import time

            time.sleep(2)
        except Exception as e:
            print(f"✗ Error checking services: {e}")
            return False

    print("\n✗ Services did not become healthy in time")
    return False


def show_connection_info() -> None:
    """Display connection information."""
    print("\n" + "=" * 60)
    print("✓ Connection Information")
    print("=" * 60)

    print("\n📊 PostgreSQL:")
    print("  Host:     localhost")
    print("  Port:     5432")
    print("  Database: cacheness_test")
    print("  User:     cacheness")
    print("  Password: cacheness_dev_password")
    print(
        "  URL:      postgresql://cacheness:cacheness_dev_password@localhost:5432/cacheness_test"
    )

    print("\n💾 Garage (S3-compatible):")
    print("  Endpoint: http://localhost:3900")
    print("  Access:   GKdeadbeef02d4b4e901234567")
    print(
        "  Secret:   0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    )
    print("  Buckets:  cache-bucket, test-bucket")


def show_next_steps() -> None:
    """Display next steps."""
    print("\n" + "=" * 60)
    print("✓ Setup Complete!")
    print("=" * 60)

    print("\n📝 Next steps:")
    print("  1. Install Python dependencies:")
    print("     pip install -e '.[dev,s3,postgresql,cloud]'")
    print()
    print("  2. Run integration tests:")
    print("     pytest tests/test_postgres_s3_integration.py -v")
    print()
    print("  3. Or explore with psql:")
    print("     psql -h localhost -U cacheness -d cacheness_test")
    print()
    print("📖 Full guide: docs/LOCAL_TEST_ENVIRONMENT.md")

    print("\n⚠️  To stop services:")
    print("     docker-compose down")
    print()
    print("   To stop and remove data:")
    print("     docker-compose down -v")


def main():
    """Main setup flow."""
    print("\n" + "=" * 60)
    print("🚀 Cacheness Local Test Environment Setup")
    print("=" * 60)

    workspace_root = Path(__file__).parent.parent
    os.chdir(workspace_root)

    # Checks
    if not check_docker():
        sys.exit(1)

    if not check_docker_compose():
        sys.exit(1)

    # Start services
    if not start_services():
        print("\n✗ Failed to start services")
        sys.exit(1)

    # Wait for health
    if not wait_for_services():
        print("\n✗ Services failed to become healthy")
        print("\nTroubleshooting:")
        print("  docker-compose logs postgres")
        print("  docker-compose logs garage")
        sys.exit(1)

    # Success
    show_connection_info()
    show_next_steps()


if __name__ == "__main__":
    main()
