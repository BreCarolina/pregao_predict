from dotenv import load_dotenv
import os

load_dotenv()  # Carrega as variáveis do arquivo .env na raiz do projeto

# Agora expomos as variáveis importantes para uso em todo projeto
PROJECT_NAME = os.getenv('PROJECT_NAME')
GITHUB_USERNAME = os.getenv('GITHUB_USERNAME')
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
EMAIL = os.getenv('EMAIL')

SMTP_EMAIL = os.getenv('SMTP_EMAIL')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
SMTP_TO = os.getenv('SMTP_TO')
