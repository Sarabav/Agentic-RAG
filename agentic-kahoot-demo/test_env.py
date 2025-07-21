import os
from dotenv import load_dotenv

load_dotenv()

print("Environment Variables Test:")
print(f"KAHOOT_PIN: {os.getenv('KAHOOT_PIN')}")
print(f"KAHOOT_NICKNAME: {os.getenv('KAHOOT_NICKNAME')}")

# Check if .env file exists
if os.path.exists('.env'):
    print("\n.env file found:")
    with open('.env', 'r') as f:
        print(f.read())
else:
    print("\nNo .env file found") 