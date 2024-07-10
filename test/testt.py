import os
from dotenv import load_dotenv
load_dotenv('../conf/.env')

PROJECT_ROOT = os.getenv("PROJECT_ROOT")

print(PROJECT_ROOT)