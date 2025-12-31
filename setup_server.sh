#!/bin/bash
# Setup script for local Pokemon Showdown server
# Required for high-throughput RL training (no rate limits)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHOWDOWN_DIR="$SCRIPT_DIR/pokemon-showdown"

echo "=== Pokemon Showdown Server Setup ==="

# Check for Node.js
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed."
    echo "Please install Node.js (v18+) from https://nodejs.org/"
    exit 1
fi

echo "Node.js version: $(node --version)"

# Check for npm
if ! command -v npm &> /dev/null; then
    echo "Error: npm is not installed."
    exit 1
fi

echo "npm version: $(npm --version)"

# Clone Pokemon Showdown if not present
if [ ! -d "$SHOWDOWN_DIR" ]; then
    echo "Cloning Pokemon Showdown..."
    git clone https://github.com/smogon/pokemon-showdown.git "$SHOWDOWN_DIR"
else
    echo "Pokemon Showdown directory already exists."
    echo "To update, run: cd $SHOWDOWN_DIR && git pull"
fi

# Install dependencies
echo "Installing dependencies..."
cd "$SHOWDOWN_DIR"
npm install

# Copy config if not exists
if [ ! -f "$SHOWDOWN_DIR/config/config.js" ]; then
    cp "$SHOWDOWN_DIR/config/config-example.js" "$SHOWDOWN_DIR/config/config.js"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start the server for training, run:"
echo "  cd $SHOWDOWN_DIR && node pokemon-showdown start --no-security"
echo ""
echo "The server will be available at: localhost:8000"
echo "WebSocket endpoint: ws://localhost:8000/showdown/websocket"
echo ""
echo "Or use the start_server.sh script (created in this directory)."

# Create a convenience start script
cat > "$SCRIPT_DIR/start_server.sh" << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/pokemon-showdown"
echo "Starting Pokemon Showdown server..."
echo "Press Ctrl+C to stop."
node pokemon-showdown start --no-security
EOF

chmod +x "$SCRIPT_DIR/start_server.sh"
echo "Created start_server.sh convenience script."
