#!/usr/bin/env python
"""
Crotalus - Minecraft Server 1.21.11
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import asyncio
from loguru import logger

from core.config import config
from core.server import run


BANNER = r"""
   ______           __        __          
  / ____/________  / /_____ _/ /_  _______
 / /   / ___/ __ \/ __/ __ `/ / / / / ___/
/ /___/ /  / /_/ / /_/ /_/ / / /_/ (__  ) 
\____/_/   \____/\__/\__,_/_/\__,_/____/  
       Minecraft 1.21.11 Server
"""


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Crotalus - A Python Minecraft 1.21.11 Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=None,
        help="Server port (default: from config)",
    )
    
    parser.add_argument(
        "-m", "--max-players",
        type=int,
        default=None,
        help="Maximum players (default: from config)",
    )
    
    parser.add_argument(
        "--ip",
        type=str,
        default=None,
        help="Bind IP address (default: from config)",
    )
    
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )
    
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.version:
        print("Crotalus 1.21.11")
        print("Protocol: 774")
        return
    
    # Override config with command line args
    if args.port:
        config.set("server.server-port", args.port)
    if args.max_players:
        config.set("server.max-players", args.max_players)
    if args.ip:
        config.set("server.server-ip", args.ip)
    
    # Print banner
    print(BANNER)
    print(f"Starting server on port {config.server_port}...")
    print("-" * 40)
    
    # Run server
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
