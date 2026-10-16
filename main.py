from click import prompt
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import streamlit as st
import time
import re
import math
from pathlib import Path


# Carrega o Modelo e o Tokenizer do Hugging Face
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct")
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-4-mini-instruct")

# Cria um Pipeline de Geração de Texto
generator = pipeline("text-generation", model="microsoft/Phi-4-mini-instruct", tokenizer=tokenizer)


# ETAPA I - Estrutura do System Prompt
SYSTEM_PROMPT = (
    "Você é o CogniTrace, um assistente de IA especializado em psicologia clínica "
    "que apoia psicólogos na análise de documentos de pacientes.\n\n"
    "PAPEL: agir como um assistente humano de psicologia, com postura ética, "
    "cuidadosa e baseada em evidências (abordagem cognitivo-comportamental).\n\n"
    "REGRAS:\n"
    "1. Baseie-se EXCLUSIVAMENTE no documento fornecido pelo usuário.\n"
    "2. Não invente informações que não estejam no texto.\n"
    "3. Não faça diagnósticos definitivos; apresente hipóteses para o psicólogo revisar.\n"
    "4. Use linguagem técnica, objetiva e respeitosa.\n\n"
    "4. Formate em Markdown: use ## para títulos, - para listas e **negrito** para termos-chave."
    "FORMATO DA RESPOSTA:\n"
    "- Comece sempre com a palavra 'CogniTrace'.\n"
    "- Resumo do caso\n"
    "- Padrões e distorções cognitivas observadas\n"
    "- Hipóteses de crenças centrais\n"
    "- Sugestões de intervenção e pontos para o psicólogo revisar."
)

# ETAPA II - Função para Gerar Texto com Base em um Prompt
def gerar_texto(prompt, max_length=4096, system_prompt=SYSTEM_PROMPT):
    mensagens = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    texto_formatado = tokenizer.apply_chat_template(
        mensagens, tokenize=False, add_generation_prompt=True
    )
    resultado = generator(texto_formatado, max_new_tokens=4096, max_length=max_length, num_return_sequences=1,return_full_text=False)
    return resultado[0]['generated_text']

# ETAPA III - Função de Efeito Máquina de Escrever
def typewriter_effect(text: str, speed: float = 0.03):
    """Efeito máquina de escrever."""
    container = st.empty()
    displayed = ""
    for ch in text:
        displayed += ch
        container.markdown(f"\n{displayed}█\n")
        time.sleep(speed)
    container.markdown(f"\n{displayed}\n")



# ETAPA IV - Leitura do Documento e Divisão de Chunks
CHUNK_SIZE = 120      # palavras por trecho
CHUNK_OVERLAP = 30    # sobreposição entre trechos
TOP_K = 4             # trechos recuperados por pergunta

# ETAPA V - Função de Ler o Documento/Texto Anexado
def ler_documento(arquivo):
    """Lê o conteúdo de um documento .txt/.md.

    Aceita tanto um arquivo enviado pelo Streamlit (UploadedFile, com .read())
    quanto um caminho/str. Retorna o texto como string.
    """
    if hasattr(arquivo, "read"):          # arquivo anexado na interface
        dados = arquivo.read()
    else:                                 # caminho no disco
        dados = Path(str(arquivo)).read_bytes()
    if isinstance(dados, bytes):
        try:
            return dados.decode("utf-8")
        except UnicodeDecodeError:
            return dados.decode("latin-1")
    return dados

# ETAPA VI - Função de Dividir Trechos
def dividir_em_trechos(texto, tamanho=CHUNK_SIZE, sobreposicao=CHUNK_OVERLAP):
    """Divide o texto em trechos de N palavras com sobreposição."""
    palavras = texto.split()
    if not palavras:
        return []
    passo = max(1, tamanho - sobreposicao)
    trechos = []
    for i in range(0, len(palavras), passo):
        bloco = palavras[i:i + tamanho]
        if not bloco:
            break
        trechos.append(" ".join(bloco))
        if i + tamanho >= len(palavras):
            break
    return trechos



# ETAPA VII - RECUPERAÇÃO (RAG minimalista): TF-IDF puro, sem frameworks pesados
def _tokens(texto):
    return re.findall(r"\b[a-z0-9áàâãéèêíïóôõöúüçñ]+\b", texto.lower())


def _vetor_tfidf(tokens, idf):
    if not tokens:
        return {}
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    vetor = {t: (c / len(tokens)) * idf.get(t, 1.0) for t, c in tf.items()}
    norma = math.sqrt(sum(v * v for v in vetor.values())) or 1.0
    return {t: v / norma for t, v in vetor.items()}


def recuperar_trechos(consulta, trechos, top_k=TOP_K):
    """Retorna os trechos mais relevantes para a consulta (similaridade de cosseno)."""
    if not trechos or not consulta.strip():
        return []

    tokens_trechos = [_tokens(t) for t in trechos]
    df = {}
    for toks in tokens_trechos:
        for termo in set(toks):
            df[termo] = df.get(termo, 0) + 1
    total = max(1, len(trechos))
    idf = {termo: math.log((1 + total) / (1 + n)) + 1 for termo, n in df.items()}

    vetor_q = _vetor_tfidf(_tokens(consulta), idf)
    if not vetor_q:
        return []

    pontuados = []
    for trecho, toks in zip(trechos, tokens_trechos):
        vetor = _vetor_tfidf(toks, idf)
        score = sum(vetor_q.get(t, 0.0) * v for t, v in vetor.items())
        if score > 0:
            pontuados.append((score, trecho))
    pontuados.sort(key=lambda item: item[0], reverse=True)
    return [trecho for _, trecho in pontuados[:top_k]]


def montar_contexto(trechos):
    """Junta os trechos recuperados em um bloco de texto para o prompt."""
    return "\n\n".join(f"[Trecho {i}]\n{t}" for i, t in enumerate(trechos, 1))


# ---------------------------------------------------------------------
# ETAPA VIV - Interface Principal + RAG
def main():
    st.title("CogniTrace AI")

    # 2) Lê o texto/documento anexado
    arquivo = st.file_uploader(
        "Anexe o documento do paciente (.txt ou .md)", type=["txt", "md"]
    )
    # Pergunta opcional do psicólogo sobre o documento
    pergunta = st.chat_input("Pergunte algo sobre o documento (opcional):")

    if arquivo is None:
        st.warning("Por favor, anexe um documento .txt ou .md para análise.")
        return

    texto = ler_documento(arquivo)
    if not texto.strip():
        st.warning("O documento está vazio.")
        return

    trechos = dividir_em_trechos(texto)

    # 3) Gera com a IA uma análise do documento como assistente de psicologia
    if pergunta:
        # RAG: recupera só os trechos relevantes à pergunta do psicólogo
        relevantes = recuperar_trechos(pergunta, trechos, top_k=TOP_K)
        contexto = montar_contexto(relevantes) if relevantes else texto[:8000]
        entrada = (
            f"CONTEXTO (trechos do documento do paciente):\n{contexto}\n\n"
            f"PERGUNTA DO PSICÓLOGO: {pergunta}"
        )
    elif st.button("Gerar análise do documento"):
        # Análise completa do documento anexado
        entrada = (
            "Analise o DOCUMENTO abaixo e produza o relatório clínico solicitado.\n\n"
            f"DOCUMENTO:\n{texto[:12000]}"
        )
    else:
        st.info(f"Documento carregado ({len(trechos)} trechos). "
                "Clique em 'Gerar análise do documento' ou faça uma pergunta.")
        return

    with st.spinner("Gerando..."):
        with st.chat_message('ai'):
            output = gerar_texto(entrada)
            typewriter_effect(output)


if __name__ == "__main__":
    main()
