import subprocess
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
streamlit_path = os.path.join(base_dir, 'Documents', 'app.py')
arquivo_streamlit = streamlit_path 

try:
    subprocess.run(["streamlit", "run", arquivo_streamlit], check=True)
except subprocess.CalledProcessError as e:
    print(f"Erro ao executar o Streamlit: {e}")
except FileNotFoundError:
    print("Erro: Comando 'streamlit' não encontrado. Certifique-se de que o Streamlit está instalado e no seu PATH.")