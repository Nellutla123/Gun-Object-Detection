#!/usr/bin/env python3
"""
Run the Guns Object Detection Web Application
"""

import uvicorn
import sys
import os

def main():
    """Run the FastAPI application with uvicorn"""
    try:
        print("🚀 Starting Guns Object Detection Web Application...")
        print("📡 Server will be available at: http://localhost:8000")
        print("🔧 Press Ctrl+C to stop the server\n")
        
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            reload_dirs=[".", "src", "static"],
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()