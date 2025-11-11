"""
nlp_pipeline.py

Módulo responsável pelo pipeline de processamento de linguagem natural.
Inclui funções para:
- Limpeza e pré-processamento de texto
- Segmentação em parágrafos e frases (com spaCy)
- Chunking (divisão em blocos)
- Extração de entidades nomeadas (NER)
- Cálculo de estatísticas
- Seleção de chunks mais relevantes baseado em embeddings
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional

# Importação do spaCy (será carregado sob demanda)
try:
    import spacy
    SPACY_DISPONIVEL = True
except ImportError:
    SPACY_DISPONIVEL = False
    print("Aviso: spaCy não encontrado. Instale com: pip install spacy")
    print("       Depois execute: python -m spacy download pt_core_news_sm")

# Modelo spaCy (carregado sob demanda)
_nlp_spacy = None


def carregar_spacy() -> Optional['spacy.Language']:
    """
    Carrega o modelo spaCy em português sob demanda.
    
    Returns:
        Modelo spaCy carregado ou None se não disponível
    """
    global _nlp_spacy
    
    if _nlp_spacy is not None:
        return _nlp_spacy
    
    if not SPACY_DISPONIVEL:
        return None
    
    try:
        _nlp_spacy = spacy.load("pt_core_news_sm")
        return _nlp_spacy
    except OSError:
        print("Erro: Modelo spaCy 'pt_core_news_sm' não encontrado.")
        print("Execute: python -m spacy download pt_core_news_sm")
        return None


def extrair_entidades(texto: str) -> Dict:
    """
    Extrai entidades nomeadas do texto usando spaCy NER.
    
    Categorias de entidades em português:
    - PER: Pessoas
    - ORG: Organizações
    - LOC: Locais
    - MISC: Outros (datas, valores, etc.)
    
    Args:
        texto: Texto para análise
        
    Returns:
        Dicionário com entidades categorizadas e estatísticas
    """
    nlp = carregar_spacy()
    
    if nlp is None:
        return {
            'disponivel': False,
            'entidades': [],
            'por_categoria': {},
            'total': 0,
            'mensagem': 'spaCy não disponível'
        }
    
    # Processa o texto
    doc = nlp(texto)
    
    # Organiza entidades por categoria
    entidades_por_tipo = {
        'PER': [],   # Pessoas
        'ORG': [],   # Organizações  
        'LOC': [],   # Locais
        'MISC': []   # Outros
    }
    
    todas_entidades = []
    
    for ent in doc.ents:
        entidade_info = {
            'texto': ent.text,
            'tipo': ent.label_,
            'inicio': ent.start_char,
            'fim': ent.end_char
        }
        
        todas_entidades.append(entidade_info)
        
        # Categoriza (spaCy português usa PER, ORG, LOC, MISC)
        if ent.label_ in entidades_por_tipo:
            entidades_por_tipo[ent.label_].append(ent.text)
        else:
            entidades_por_tipo['MISC'].append(ent.text)
    
    # Remove duplicatas mantendo ordem
    for tipo in entidades_por_tipo:
        entidades_por_tipo[tipo] = list(dict.fromkeys(entidades_por_tipo[tipo]))
    
    return {
        'disponivel': True,
        'entidades': todas_entidades,
        'por_categoria': entidades_por_tipo,
        'total': len(todas_entidades),
        'total_unicas': sum(len(v) for v in entidades_por_tipo.values()),
        'mensagem': 'OK'
    }


def segmentar_em_frases_spacy(texto: str) -> List[str]:
    """
    Divide o texto em frases usando spaCy (mais preciso que regex).
    
    Args:
        texto: Texto a ser segmentado
        
    Returns:
        Lista de frases
    """
    nlp = carregar_spacy()
    
    if nlp is None:
        # Fallback para regex se spaCy não disponível
        return segmentar_em_frases(texto)
    
    doc = nlp(texto)
    frases = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    return frases


def limpar_texto(texto: str) -> str:
    """
    Realiza limpeza básica do texto de entrada.
    
    Remove:
    - Espaços em branco duplicados
    - Quebras de linha excessivas
    - Caracteres especiais problemáticos
    
    Args:
        texto: Texto original a ser limpo
        
    Returns:
        Texto limpo e normalizado
    """
    # Remove quebras de linha duplicadas
    texto = re.sub(r'\n\s*\n', '\n\n', texto)
    
    # Remove espaços duplicados
    texto = re.sub(r' +', ' ', texto)
    
    # Remove espaços no início e fim de linhas
    linhas = [linha.strip() for linha in texto.split('\n')]
    texto = '\n'.join(linhas)
    
    # Remove caracteres de controle (exceto quebras de linha e tabs)
    texto = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', texto)
    
    return texto.strip()


def segmentar_em_paragrafos(texto: str) -> List[str]:
    """
    Divide o texto em parágrafos.
    
    Um parágrafo é definido como texto separado por uma ou mais quebras de linha.
    
    Args:
        texto: Texto a ser segmentado
        
    Returns:
        Lista de parágrafos
    """
    # Divide por quebras de linha duplas ou mais
    paragrafos = re.split(r'\n\s*\n', texto)
    
    # Remove parágrafos vazios e faz strip
    paragrafos = [p.strip() for p in paragrafos if p.strip()]
    
    return paragrafos


def segmentar_em_frases(texto: str) -> List[str]:
    """
    Divide o texto em frases individuais.
    
    Usa regex simples para detectar finais de frase (. ! ?).
    Para segmentação mais sofisticada, pode-se usar spaCy.
    
    Args:
        texto: Texto a ser segmentado
        
    Returns:
        Lista de frases
    """
    # Padrão para detectar fim de frase
    # Considera ponto, exclamação ou interrogação seguido de espaço e maiúscula
    padrao = r'(?<=[.!?])\s+(?=[A-ZÁÀÂÃÉÈÊÍÏÓÔÕÖÚÇÑ])'
    
    frases = re.split(padrao, texto)
    
    # Remove frases vazias
    frases = [f.strip() for f in frases if f.strip()]
    
    return frases


def contar_palavras(texto: str) -> int:
    """
    Conta o número de palavras em um texto.
    
    Args:
        texto: Texto para contagem
        
    Returns:
        Número de palavras
    """
    palavras = texto.split()
    return len(palavras)


def criar_chunks(texto: str, tamanho_chunk: int = 1000, overlap: int = 200, usar_spacy: bool = True) -> List[Dict]:
    """
    Divide o texto em chunks (blocos) de tamanho aproximado.
    
    Estratégia:
    - Divide primeiro em sentenças (usando spaCy se disponível)
    - Agrupa sentenças até atingir o tamanho desejado
    - Mantém overlap entre chunks para preservar contexto
    - Chunks respeitam limites de sentença para melhor coerência
    
    Args:
        texto: Texto a ser dividido
        tamanho_chunk: Tamanho aproximado de cada chunk em caracteres
        overlap: Número de caracteres de sobreposição entre chunks
        usar_spacy: Se True, tenta usar spaCy para segmentação de sentenças
        
    Returns:
        Lista de dicionários com informações de cada chunk
    """
    # Tenta usar spaCy para segmentação melhor
    if usar_spacy and carregar_spacy() is not None:
        sentencas = segmentar_em_frases_spacy(texto)
    else:
        # Fallback: usa parágrafos se spaCy não disponível
        sentencas = segmentar_em_paragrafos(texto)
    
    chunks = []
    chunk_atual = ""
    sentencas_no_chunk = []
    indice_chunk = 0
    
    for sentenca in sentencas:
        # Se adicionar esta sentença não ultrapassar muito o limite
        if len(chunk_atual) + len(sentenca) <= tamanho_chunk:
            chunk_atual += sentenca + " "
            sentencas_no_chunk.append(sentenca)
        else:
            # Salva o chunk atual se não estiver vazio
            if chunk_atual.strip():
                chunks.append({
                    'indice': indice_chunk,
                    'texto': chunk_atual.strip(),
                    'tamanho': len(chunk_atual.strip()),
                    'num_palavras': contar_palavras(chunk_atual.strip()),
                    'num_sentencas': len(sentencas_no_chunk)
                })
                indice_chunk += 1
            
            # Inicia novo chunk com overlap
            if overlap > 0 and sentencas_no_chunk:
                # Pega as últimas sentenças que caibam no overlap
                overlap_sentencas = []
                overlap_chars = 0
                
                for sent in reversed(sentencas_no_chunk):
                    if overlap_chars + len(sent) <= overlap:
                        overlap_sentencas.insert(0, sent)
                        overlap_chars += len(sent)
                    else:
                        break
                
                chunk_atual = " ".join(overlap_sentencas) + " " + sentenca + " "
                sentencas_no_chunk = overlap_sentencas + [sentenca]
            else:
                chunk_atual = sentenca + " "
                sentencas_no_chunk = [sentenca]
    
    # Adiciona o último chunk se houver
    if chunk_atual.strip():
        chunks.append({
            'indice': indice_chunk,
            'texto': chunk_atual.strip(),
            'tamanho': len(chunk_atual.strip()),
            'num_palavras': contar_palavras(chunk_atual.strip()),
            'num_sentencas': len(sentencas_no_chunk)
        })
    
    return chunks


def calcular_similaridade_coseno(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calcula a similaridade coseno entre dois vetores de embedding.
    
    A similaridade coseno mede o cosseno do ângulo entre dois vetores,
    variando de -1 (completamente opostos) a 1 (idênticos).
    
    Args:
        embedding1: Primeiro vetor de embedding
        embedding2: Segundo vetor de embedding
        
    Returns:
        Valor de similaridade entre 0 e 1
    """
    # Normaliza os vetores
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Calcula produto escalar dividido pelo produto das normas
    similaridade = np.dot(embedding1, embedding2) / (norm1 * norm2)
    
    return float(similaridade)


def selecionar_chunks_relevantes(
    chunks: List[Dict],
    embeddings_chunks: List[np.ndarray],
    embedding_global: np.ndarray,
    percentual_selecao: float = 0.7
) -> List[Dict]:
    """
    Seleciona os chunks mais relevantes baseado na similaridade com o embedding global.
    
    Estratégia:
    - Calcula a similaridade de cada chunk com o embedding do documento completo
    - Ordena os chunks por similaridade
    - Retorna os top N chunks (definido por percentual_selecao)
    
    Args:
        chunks: Lista de chunks com metadados
        embeddings_chunks: Lista de embeddings correspondentes aos chunks
        embedding_global: Embedding do documento completo
        percentual_selecao: Percentual de chunks a selecionar (0.0 a 1.0)
        
    Returns:
        Lista de chunks selecionados, ordenados pela ordem original
    """
    if not chunks or not embeddings_chunks:
        return chunks
    
    # Calcula similaridade de cada chunk com o embedding global
    similaridades = []
    for i, embedding_chunk in enumerate(embeddings_chunks):
        sim = calcular_similaridade_coseno(embedding_chunk, embedding_global)
        similaridades.append({
            'indice_original': i,
            'chunk': chunks[i],
            'similaridade': sim
        })
    
    # Ordena por similaridade (maior para menor)
    similaridades_ordenadas = sorted(similaridades, key=lambda x: x['similaridade'], reverse=True)
    
    # Seleciona top N chunks
    num_selecionar = max(1, int(len(chunks) * percentual_selecao))
    chunks_selecionados = similaridades_ordenadas[:num_selecionar]
    
    # Reordena pela ordem original para manter a coerência do texto
    chunks_selecionados = sorted(chunks_selecionados, key=lambda x: x['indice_original'])
    
    # Adiciona informação de similaridade aos chunks
    resultado = []
    for item in chunks_selecionados:
        chunk_com_sim = item['chunk'].copy()
        chunk_com_sim['similaridade'] = item['similaridade']
        resultado.append(chunk_com_sim)
    
    return resultado


def calcular_estatisticas(texto: str) -> Dict:
    """
    Calcula estatísticas descritivas sobre o texto.
    
    Args:
        texto: Texto para análise
        
    Returns:
        Dicionário com estatísticas (palavras, frases, caracteres, parágrafos)
    """
    num_caracteres = len(texto)
    num_palavras = contar_palavras(texto)
    frases = segmentar_em_frases(texto)
    num_frases = len(frases)
    paragrafos = segmentar_em_paragrafos(texto)
    num_paragrafos = len(paragrafos)
    
    return {
        'num_caracteres': num_caracteres,
        'num_palavras': num_palavras,
        'num_frases': num_frases,
        'num_paragrafos': num_paragrafos
    }


def processar_texto_completo(
    texto: str,
    percentual_selecao: float = 0.7,
    tamanho_chunk: int = 1000
) -> Tuple[str, List[Dict], Dict]:
    """
    Executa o pipeline completo de pré-processamento de texto.
    
    Pipeline:
    1. Limpa o texto
    2. Calcula estatísticas
    3. Divide em chunks
    
    Nota: Os embeddings e seleção de chunks são feitos separadamente
    pois dependem da chamada à API da OpenAI.
    
    Args:
        texto: Texto original
        percentual_selecao: Percentual de chunks a selecionar depois
        tamanho_chunk: Tamanho desejado para cada chunk
        
    Returns:
        Tupla contendo:
        - Texto limpo
        - Lista de chunks
        - Estatísticas do texto original
    """
    # Etapa 1: Limpeza
    texto_limpo = limpar_texto(texto)
    
    # Etapa 2: Estatísticas
    estatisticas = calcular_estatisticas(texto_limpo)
    
    # Etapa 3: Chunking (usando spaCy se disponível)
    chunks = criar_chunks(texto_limpo, tamanho_chunk=tamanho_chunk, overlap=200, usar_spacy=True)
    
    return texto_limpo, chunks, estatisticas


def analisar_com_spacy(texto: str, chunks: List[Dict] = None) -> Dict:
    """
    Realiza análise linguística completa do texto usando spaCy.
    
    Inclui:
    - Extração de entidades nomeadas (NER)
    - Segmentação de sentenças com spaCy vs regex
    - Análise de chunks (se fornecidos)
    - Estatísticas comparativas
    
    Args:
        texto: Texto para análise
        chunks: Lista opcional de chunks para análise adicional
        
    Returns:
        Dicionário com resultados da análise linguística
    """
    resultado = {
        'spacy_disponivel': carregar_spacy() is not None,
        'entidades': {},
        'segmentacao': {},
        'chunks_analise': [],
        'estatisticas': {}
    }
    
    if not resultado['spacy_disponivel']:
        resultado['mensagem'] = 'spaCy não está disponível. Instale com: pip install spacy && python -m spacy download pt_core_news_sm'
        return resultado
    
    # Extração de entidades
    resultado['entidades'] = extrair_entidades(texto)
    
    # Comparação de segmentação
    frases_spacy = segmentar_em_frases_spacy(texto)
    frases_regex = segmentar_em_frases(texto)
    
    resultado['segmentacao'] = {
        'num_frases_spacy': len(frases_spacy),
        'num_frases_regex': len(frases_regex),
        'diferenca': abs(len(frases_spacy) - len(frases_regex)),
        'melhoria': len(frases_spacy) < len(frases_regex)  # spaCy geralmente identifica menos mas corretamente
    }
    
    # Análise de chunks (se fornecidos)
    if chunks:
        for chunk in chunks:
            entidades_chunk = extrair_entidades(chunk['texto'])
            
            resultado['chunks_analise'].append({
                'indice': chunk['indice'],
                'num_sentencas': chunk.get('num_sentencas', 0),
                'num_entidades': entidades_chunk['total'],
                'entidades_por_tipo': {
                    k: len(v) for k, v in entidades_chunk['por_categoria'].items() if v
                },
                'similaridade': chunk.get('similaridade', 0)
            })
    
    # Estatísticas gerais
    resultado['estatisticas'] = {
        'total_entidades': resultado['entidades']['total'],
        'total_entidades_unicas': resultado['entidades']['total_unicas'],
        'categorias_encontradas': [k for k, v in resultado['entidades']['por_categoria'].items() if v]
    }
    
    return resultado
