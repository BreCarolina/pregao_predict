#@title Script operacional para configurar Git e sincronizar com GitHub
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='/content/.env')

def git_config():
    """Configura o Git localmente e sincroniza com o repositório remoto no GitHub."""

    # Carregar variáveis do ambiente
    load_dotenv(dotenv_path='/content/.env')

    GITHUB_USERNAME = os.getenv('GITHUB_USERNAME')
    EMAIL = os.getenv('EMAIL')
    GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
    PROJECT_NAME = os.getenv('PROJECT_NAME')

    if not all([GITHUB_USERNAME, EMAIL, GITHUB_TOKEN, PROJECT_NAME]):
        raise ValueError("Variáveis de ambiente faltando. Verifique o arquivo .env.")

    REPO_URL = f"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{PROJECT_NAME}.git"

    # Configurações globais do Git
    os.system(f'git config --global user.name "{GITHUB_USERNAME}"')
    os.system(f'git config --global user.email "{EMAIL}"')

    if os.path.isdir(PROJECT_NAME):
        print(f"[INFO] Diretório '{PROJECT_NAME}' já existe. Sincronizando...")
        os.chdir(PROJECT_NAME)

        os.system("git init")
        os.system("git remote remove origin || true")
        os.system(f"git remote add origin {REPO_URL}")
        os.system("git fetch origin")
        os.system("git checkout -B main")
        os.system("git pull origin main --allow-unrelated-histories --no-rebase")
    else:
        print(f"[INFO] Clonando o repositório '{PROJECT_NAME}'...")
        os.system(f"git clone {REPO_URL}")
        os.chdir(PROJECT_NAME)

    print(f"[SUCESSO] Git configurado e sincronizado com: {REPO_URL}")

if __name__ == "__main__":
    git_config()
