import os
import json
import hashlib

import numpy as np
import streamlit as st
from llama_cpp import Llama

# ----------------------------------------------------------------------
# CONFIGURAÇÕES
# ----------------------------------------------------------------------
PASTA_SCRIPT = os.path.dirname(os.path.abspath(__file__))

MODELO_CHAT = os.path.join(PASTA_SCRIPT, "gemma-3-4b-it-Q4_K_M.gguf")
MODELO_EMBEDDING = os.path.join(PASTA_SCRIPT, "bge-m3-Q8_0.gguf")

# Se usar o nomic-embed-text-v1.5, troque por:
# PREFIXO_DOC = "search_document: "
# PREFIXO_QUERY = "search_query: "
PREFIXO_DOC = ""
PREFIXO_QUERY = ""

PASTA_DOCS = os.path.join(PASTA_SCRIPT, "documentos")
PASTA_INDICE = os.path.join(PASTA_SCRIPT, "indice")
os.makedirs(PASTA_DOCS, exist_ok=True)
os.makedirs(PASTA_INDICE, exist_ok=True)

CHUNK_SIZE = 120      # palavras por trecho
CHUNK_OVERLAP = 30    # sobreposição entre trechos
TOP_K = 4             # quantos trechos enviar ao modelo
MIN_SCORE = 0.25      # similaridade mínima (ajuste olhando as fontes na tela)

N_CTX = 4096
N_THREADS = 3
HISTORICO_MAX = 6     # últimas mensagens enviadas ao modelo

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


# ----------------------------------------------------------------------
# MODELOS (carregados uma única vez)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Carregando modelo de chat...")
def carregar_llm():
    return Llama(
        model_path=MODELO_CHAT,
        n_ctx=4096,
        n_threads=4,
        verbose=False,
    )


@st.cache_resource(show_spinner="Carregando modelo de embeddings...")
def carregar_embedder():
    return Llama(
        model_path=MODELO_EMBEDDING,
        embedding=True,
        n_ctx=4096,
        n_batch=2048,
        n_threads=4,
        verbose=False,
    )


# ----------------------------------------------------------------------
# LEITURA E CHUNKING DOS DOCUMENTOS
# ----------------------------------------------------------------------
def listar_arquivos():
    exts = (".txt", ".md", ".pdf")
    return sorted(
        f for f in os.listdir(PASTA_DOCS) if f.lower().endswith(exts)
    )


def ler_arquivo(caminho: str) -> str:
    if caminho.lower().endswith(".pdf"):
        from pypdf import PdfReader
        leitor = PdfReader(caminho)
        return "\n".join((p.extract_text() or "") for p in leitor.pages)
    with open(caminho, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def dividir_em_chunks(texto: str):
    palavras = texto.split()
    passo = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    chunks = []
    for inicio in range(0, len(palavras), passo):
        trecho = palavras[inicio:inicio + CHUNK_SIZE]
        if trecho:
            chunks.append(" ".join(trecho))
        if inicio + CHUNK_SIZE >= len(palavras):
            break
    return chunks


# ----------------------------------------------------------------------
# EMBEDDINGS E ÍNDICE VETORIAL
# ----------------------------------------------------------------------
def gerar_embedding(embedder: Llama, texto: str) -> np.ndarray:
    resp = embedder.create_embedding(texto)
    vetor = np.array(resp["data"][0]["embedding"], dtype=np.float32)
    if vetor.ndim == 2:            # alguns modelos devolvem 1 vetor por token
        vetor = vetor.mean(axis=0)
    vetor /= (np.linalg.norm(vetor) + 1e-9)   # normaliza -> cosseno = produto escalar
    return vetor


def assinatura_docs() -> str:
    """Muda sempre que um documento é adicionado/alterado/removido."""
    partes = [f"{CHUNK_SIZE}-{CHUNK_OVERLAP}-{os.path.basename(MODELO_EMBEDDING)}"]
    for nome in listar_arquivos():
        c = os.path.join(PASTA_DOCS, nome)
        partes.append(f"{nome}:{os.path.getsize(c)}:{os.path.getmtime(c)}")
    return hashlib.md5("|".join(partes).encode()).hexdigest()


@st.cache_resource(show_spinner=False)
def carregar_indice(assinatura: str):
    caminho_meta = os.path.join(PASTA_INDICE, "meta.json")
    caminho_vet = os.path.join(PASTA_INDICE, "vetores.npy")

    # 1) tenta reutilizar o índice salvo em disco
    if os.path.exists(caminho_meta) and os.path.exists(caminho_vet):
        with open(caminho_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("assinatura") == assinatura:
            return meta["chunks"], np.load(caminho_vet)

    # 2) senão, reconstrói
    embedder = carregar_embedder()
    chunks = []
    for nome in listar_arquivos():
        texto = ler_arquivo(os.path.join(PASTA_DOCS, nome))
        for trecho in dividir_em_chunks(texto):
            chunks.append({"texto": trecho, "fonte": nome})

    if not chunks:
        return [], np.zeros((0, 1), dtype=np.float32)

    barra = st.progress(0.0, text="Indexando documentos...")
    vetores = []
    for i, c in enumerate(chunks):
        vetores.append(gerar_embedding(embedder, PREFIXO_DOC + c["texto"]))
        barra.progress((i + 1) / len(chunks), text=f"Indexando {i + 1}/{len(chunks)}")
    barra.empty()

    matriz = np.vstack(vetores)
    np.save(caminho_vet, matriz)
    with open(caminho_meta, "w", encoding="utf-8") as f:
        json.dump({"assinatura": assinatura, "chunks": chunks}, f, ensure_ascii=False)
    return chunks, matriz


def buscar_contexto(pergunta: str, chunks, matriz):
    if len(chunks) == 0:
        return []
    embedder = carregar_embedder()
    q = gerar_embedding(embedder, PREFIXO_QUERY + pergunta)
    scores = matriz @ q
    melhores = np.argsort(scores)[::-1][:TOP_K]
    return [
        {**chunks[i], "score": float(scores[i])}
        for i in melhores
        if scores[i] >= MIN_SCORE
    ]


# ----------------------------------------------------------------------
# GERAÇÃO
# ----------------------------------------------------------------------
def montar_mensagens(pergunta: str, trechos, historico):
    mensagens = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in historico[-HISTORICO_MAX:]:
        mensagens.append({"role": m["role"], "content": m["content"]})

    if trechos:
        contexto = "\n\n".join(
            f"[{i + 1}] (fonte: {t['fonte']})\n{t['texto']}"
            for i, t in enumerate(trechos)
        )
        conteudo = (
            f"CONTEXT:\n{contexto}\n\n"
            f"QUESTION:\n{pergunta}"
        )
    else:
        conteudo = pergunta

    mensagens.append({"role": "user", "content": conteudo})
    return mensagens


def gerar_stream(mensagens):
    llm = carregar_llm()
    for parte in llm.create_chat_completion(
        messages=mensagens, temperature=0.7, stream=True
    ):
        delta = parte["choices"][0]["delta"]
        if delta.get("content"):
            yield delta["content"]


# ----------------------------------------------------------------------
# INTERFACE
# ----------------------------------------------------------------------
def mostrar_fontes(fontes):
    if not fontes:
        return
    with st.expander("Fontes utilizadas"):
        for f in fontes:
            st.caption(f"**{f['fonte']}** (similaridade {f['score']:.2f})")
            st.write(f["texto"])


def main():
    st.title("CogniTrace")

    # --- Sidebar: gerenciamento de documentos ---
    with st.sidebar:
        st.header("Base de conhecimento")
        enviados = st.file_uploader(
            "Adicionar documentos",
            type=["txt", "md", "pdf"],
            accept_multiple_files=True,
        )
        if enviados:
            for arq in enviados:
                destino = os.path.join(PASTA_DOCS, arq.name)
                if not os.path.exists(destino):
                    with open(destino, "wb") as f:
                        f.write(arq.getbuffer())
        arquivos = listar_arquivos()
        st.write(f"{len(arquivos)} documento(s) indexado(s):")
        for a in arquivos:
            st.caption(f"• {a}")
        usar_rag = st.toggle("Usar RAG", value=True)

    chunks, matriz = carregar_indice(assinatura_docs())

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            mostrar_fontes(msg.get("fontes"))

    if prompt := st.chat_input("Mensagem:"):
        historico = list(st.session_state.messages)  # antes de adicionar a atual
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            trechos = []
            if usar_rag:
                with st.spinner("Buscando nos documentos..."):
                    trechos = buscar_contexto(prompt, chunks, matriz)

            mensagens = montar_mensagens(prompt, trechos, historico)
            resposta = st.write_stream(gerar_stream(mensagens))
            mostrar_fontes(trechos)

        st.session_state.messages.append(
            {"role": "assistant", "content": resposta, "fontes": trechos}
        )


if __name__ == "__main__":
    main()