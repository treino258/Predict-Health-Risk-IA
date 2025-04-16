import subprocess

arquivo_streamlit = "Documents/app.py"  # Substitua pelo nome do seu arquivo

try:
    subprocess.run(["streamlit", "run", arquivo_streamlit], check=True)
except subprocess.CalledProcessError as e:
    print(f"Erro ao executar o Streamlit: {e}")
except FileNotFoundError:
    print("Erro: Comando 'streamlit' não encontrado. Certifique-se de que o Streamlit está instalado e no seu PATH.")