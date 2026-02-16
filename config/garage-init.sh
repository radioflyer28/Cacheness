#!/bin/sh
# Garage initialization script for local development/testing
# Waits for Garage to be ready, then configures layout, keys, and buckets.
#
# This script runs inside the Garage container as an entrypoint wrapper.
# It starts Garage in the background, runs setup, then foregrounds it.

set -e

# Known dev credentials (deterministic for testing — NOT for production)
DEV_ACCESS_KEY="GKdeadbeef02d4b4e901234567"
DEV_SECRET_KEY="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

# Generate RPC secret if it doesn't exist
if [ ! -f /var/lib/garage/rpc_secret ]; then
    openssl rand -hex 32 > /var/lib/garage/rpc_secret
fi
chmod 600 /var/lib/garage/rpc_secret

# Start Garage in the background
garage server &
GARAGE_PID=$!

echo "Waiting for Garage to start..."
for i in $(seq 1 30); do
    if garage status 2>/dev/null | grep -q "NO ROLE\|Gateway"; then
        echo "Garage is up after ${i}s"
        break
    fi
    # Already has a role assigned from previous run
    if garage status 2>/dev/null | grep -qE "[0-9a-f]{16}"; then
        echo "Garage is up after ${i}s"
        break
    fi
    sleep 1
done

# Get the node ID
NODE_ID=$(garage status 2>/dev/null | grep -oE '[0-9a-f]{16}' | head -1)
if [ -z "$NODE_ID" ]; then
    echo "ERROR: Could not determine Garage node ID"
    wait $GARAGE_PID
    exit 1
fi
echo "Node ID: $NODE_ID"

# Assign layout (zone=dc1, capacity=1GB) — idempotent
garage layout assign -z dc1 -c 1G "$NODE_ID" 2>/dev/null || true

# Apply layout if there are staged changes
# Apply layout — try a few version numbers (handles fresh and restarted containers)
APPLIED=false
for v in 1 2 3 4 5; do
    if garage layout apply --version "$v" 2>/dev/null; then
        echo "Layout applied (version $v)"
        APPLIED=true
        break
    fi
done
if [ "$APPLIED" = "false" ]; then
    echo "Layout already applied or no staged changes"
fi

# Wait for layout to propagate
sleep 2

# Import known API key (idempotent — errors if key name already exists)
garage key import --yes -n cacheness-dev-key "$DEV_ACCESS_KEY" "$DEV_SECRET_KEY" 2>/dev/null \
    || echo "Key 'cacheness-dev-key' already exists"

# Create buckets and grant access
for BUCKET in cache-bucket test-bucket; do
    garage bucket create "$BUCKET" 2>/dev/null || echo "Bucket '${BUCKET}' already exists"
    garage bucket allow --read --write --owner "$BUCKET" --key cacheness-dev-key 2>/dev/null || true
    echo "Bucket '${BUCKET}' ready"
done

echo ""
echo "============================================"
echo "Garage S3 ready for development/testing"
echo "  S3 endpoint:  http://localhost:3900"
echo "  Access key:   ${DEV_ACCESS_KEY}"
echo "  Secret key:   ${DEV_SECRET_KEY}"
echo "  Buckets:      cache-bucket, test-bucket"
echo "============================================"
echo ""

# Foreground Garage (keep container running)
wait $GARAGE_PID
