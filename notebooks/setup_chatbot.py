#!/usr/bin/env python3
"""
Setup script for HealthWave Chatbot RAG System
This script sets up the virtual environment and runs chatbot_final.py
"""

import os
import sys
import subprocess

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def main():
    print("🚀 HealthWave Chatbot Setup Script")
    print("=" * 50)

    # Check if we're in the notebooks directory
    current_dir = os.getcwd()
    if not current_dir.endswith("notebooks"):
        notebooks_path = os.path.join(current_dir, "notebooks")
        if os.path.exists(notebooks_path):
            os.chdir(notebooks_path)
            print(f"📁 Changed directory to: {notebooks_path}")
        else:
            print("❌ Please run this script from the project root or notebooks directory")
            return False

    print(f"📁 Working directory: {os.getcwd()}")

    # Create virtual environment
    if not run_command("python -m venv chatbot_env", "Creating virtual environment"):
        return False

    # Activate virtual environment and install requirements
    if os.name == 'nt':  # Windows
        pip_path = "chatbot_env\\Scripts\\pip"
        python_path = "chatbot_env\\Scripts\\python"
    else:  # Unix/Linux/MacOS
        pip_path = "chatbot_env/bin/pip"
        python_path = "chatbot_env/bin/python"

    # Upgrade pip
    if not run_command(f"{pip_path} install --upgrade pip", "Upgrading pip"):
        return False

    # Install requirements
    if not run_command(f"{pip_path} install -r requirements.txt", "Installing requirements"):
        return False

    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Activate the virtual environment:")
    if os.name == 'nt':
        print("   chatbot_env\\Scripts\\activate")
    else:
        print("   source chatbot_env/bin/activate")
    print("2. Run the chatbot setup:")
    print("   python chatbot_final.py")
    print("\n💡 Or run directly with:")
    print(f"   {python_path} chatbot_final.py")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)