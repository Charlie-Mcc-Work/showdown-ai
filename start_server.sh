#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/pokemon-showdown"
echo "Starting Pokemon Showdown server..."
echo "Press Ctrl+C to stop."
node pokemon-showdown start --no-security
