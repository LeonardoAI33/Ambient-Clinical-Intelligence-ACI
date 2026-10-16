"""
Validação da camada RAG do CogniTrace (sem carregar o LLM de 7B).

Extrai as funções puras REAIS do main.py via AST e prova que:
  - o documento .txt/.md é lido;
  - o texto é dividido em trechos;
  - a recuperação (TF-IDF) injeta o trecho certo no prompt.
"""

import ast
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
FUNCOES_PURAS = {
    "ler_documento", "dividir_em_trechos", "_tokens",
    "_vetor_tfidf", "recuperar_trechos", "montar_contexto",
}
CONSTANTES = {"CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K"}


def carregar_modulo_rag():
    arvore = ast.parse((RAIZ / "main.py").read_text(encoding="utf-8"))
    selecionados = []
    for no in arvore.body:
        if isinstance(no, (ast.Import, ast.ImportFrom)):
            selecionados.append(no)
        elif isinstance(no, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in CONSTANTES for t in no.targets
        ):
            selecionados.append(no)
        elif isinstance(no, ast.FunctionDef) and no.name in FUNCOES_PURAS:
            selecionados.append(no)
    ns = {}
    exec(compile(ast.Module(body=selecionados, type_ignores=[]), "main.py[RAG]", "exec"), ns)
    return ns


def main():
    rag = carregar_modulo_rag()

    # 1) Lê o documento .md e o .txt do disco.
    md = rag["ler_documento"](RAIZ / "documentos" / "distorcoes_cognitivas.md")
    txt = rag["ler_documento"](RAIZ / "documentos" / "protocolo_tcc.txt")
    assert "Catastrofização" in md
    assert "50 minutos" in txt
    print("[OK] ler_documento() leu .md e .txt corretamente")

    # 2) Divide em trechos.
    trechos = rag["dividir_em_trechos"](txt)
    assert len(trechos) >= 1
    print(f"[OK] dividir_em_trechos() gerou {len(trechos)} trecho(s)")

    # 3) Recuperação responde com base no CONTEÚDO do documento.
    casos = [
        (md, "o que é catastrofização", "catastrof"),
        (txt, "quanto tempo dura a sessão de TCC", "50 minutos"),
        (txt, "crença central de desamparo", "desamparo"),
        (txt, "o que é o registro de pensamentos automáticos RPD", "rpd"),
    ]
    for texto, pergunta, termo in casos:
        ts = rag["dividir_em_trechos"](texto)
        achados = rag["recuperar_trechos"](pergunta, ts)
        assert achados, f"Nada recuperado para: {pergunta}"
        contexto = rag["montar_contexto"](achados)
        assert termo in contexto.lower(), f"Fato {termo!r} ausente para {pergunta!r}"
        print(f"[OK] {pergunta!r} -> fato '{termo}' presente em {len(achados)} trecho(s)")

    # 4) O prompt final leva o contexto do documento ao modelo.
    pergunta = "quanto tempo dura a sessão de TCC"
    achados = rag["recuperar_trechos"](pergunta, rag["dividir_em_trechos"](txt))
    prompt_final = (
        "CONTEXTO (trechos do documento do paciente):\n"
        f"{rag['montar_contexto'](achados)}\n\n"
        f"PERGUNTA DO PSICÓLOGO: {pergunta}"
    )
    assert "50 minutos" in prompt_final
    print("\n[OK] Prompt aumentado contém o fato do arquivo -----------------")
    print("    ", prompt_final.replace("\n", " ")[:150], "...")
    print("\nVALIDAÇÃO CONCLUÍDA.")


if __name__ == "__main__":
    main()
