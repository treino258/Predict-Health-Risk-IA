
import streamlit as st
from llm.client import create_chat, send_message


def render_chat():
    st.title("🤖 Assistente de Risco Cardíaco")
    st.caption("Converse com a IA sobre o modelo, fatores de risco e predições de pacientes.")

    # Inicializa sessão
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = create_chat()

    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Mensagem inicial
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                "Olá! Sou o assistente de análise de risco cardíaco. "
                "Posso te ajudar com:\n\n"
                "• 📊 **Quais variáveis mais influenciam o risco cardíaco**\n"
                "• 🔍 **Predição de risco para um paciente específico**\n"
                "• 📈 **Métricas e confiabilidade do modelo**\n\n"
                "Como posso ajudar?"
            )
        })

    # Exemplos de perguntas
    st.markdown("**💡 Exemplos de perguntas:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📊 Fatores de risco", use_container_width=True):
            st.session_state.pending_message = "Quais são os fatores clínicos que mais influenciam o risco cardíaco segundo o modelo?"
    with col2:
        if st.button("🔍 Predição exemplo", use_container_width=True):
            st.session_state.pending_message = "Analise o risco de um paciente masculino de 60 anos com dor no peito assintomática (ASY), pressão 140, colesterol 280, glicemia alta (1), ECG normal, frequência máxima 130, com angina (Y), oldpeak 2.0, ST slope Flat."
    with col3:
        if st.button("📈 Métricas do modelo", use_container_width=True):
            st.session_state.pending_message = "Qual é a precisão e confiabilidade do modelo?"

    st.divider()

    # Exibe histórico de mensagens
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Processa mensagem pendente (dos botões de exemplo)
    if "pending_message" in st.session_state:
        user_input = st.session_state.pop("pending_message")
        _process_message(user_input)
        st.rerun()

    # Input do usuário
    if user_input := st.chat_input("Digite sua pergunta..."):
        _process_message(user_input)
        st.rerun()

    # Botão de limpar conversa
    if len(st.session_state.messages) > 1:
        if st.button("🗑️ Limpar conversa"):
            st.session_state.chat_session = create_chat()
            st.session_state.messages = []
            st.rerun()


def _process_message(user_input: str):
    """Processa uma mensagem do usuário e adiciona ao histórico."""
    # Adiciona mensagem do usuário
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Obtém resposta do Gemini
    with st.spinner("Consultando o modelo..."):
        response = send_message(st.session_state.chat_session, user_input)

    # Adiciona resposta do assistente
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })