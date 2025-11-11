"""
llm_client.py

Módulo responsável pela comunicação com a API da OpenAI.
Gerencia chamadas para:
- Modelo o4-mini: geração de resumos e simplificações
- Modelo text-embedding-3-small: cálculo de embeddings
"""

import os
from typing import List, Dict
import numpy as np


# Configuração da API da OpenAI
# IMPORTANTE: Defina a variável de ambiente OPENAI_API_KEY com sua chave
# Exemplo no Windows PowerShell:
# $env:OPENAI_API_KEY = "sua-chave-aqui"
# 
# Ou crie um arquivo .env na raiz do projeto com:
# OPENAI_API_KEY=sua-chave-aqui

try:
    from openai import OpenAI
    OPENAI_DISPONIVEL = True
except ImportError:
    OPENAI_DISPONIVEL = False
    print("Aviso: Biblioteca openai não encontrada. Instale com: pip install openai")


def inicializar_cliente() -> 'OpenAI':
    """
    Inicializa o cliente da OpenAI com a chave da API.
    
    Returns:
        Cliente OpenAI configurado
        
    Raises:
        ValueError: Se a chave da API não estiver configurada
    """
    if not OPENAI_DISPONIVEL:
        raise ImportError("Biblioteca openai não está instalada")
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "Chave da API da OpenAI não encontrada. "
            "Configure a variável de ambiente OPENAI_API_KEY"
        )
    
    return OpenAI(api_key=api_key)


def truncar_texto_para_embedding(texto: str, max_tokens: int = 8000) -> str:
    """
    Trunca o texto para caber no limite de tokens do modelo de embedding.
    
    O text-embedding-3-small tem limite de 8192 tokens.
    Usamos aproximação de 1 token = 4 caracteres.
    
    Args:
        texto: Texto original
        max_tokens: Número máximo de tokens (padrão: 8000 para margem de segurança)
        
    Returns:
        Texto truncado
    """
    # Aproximação: 1 token ≈ 4 caracteres em português
    max_chars = max_tokens * 4
    
    if len(texto) <= max_chars:
        return texto
    
    # Trunca mantendo palavras completas
    texto_truncado = texto[:max_chars]
    ultimo_espaco = texto_truncado.rfind(' ')
    
    if ultimo_espaco > 0:
        texto_truncado = texto_truncado[:ultimo_espaco]
    
    print(f"Aviso: Texto truncado de {len(texto)} para {len(texto_truncado)} caracteres para caber no limite de tokens")
    
    return texto_truncado


def gerar_embedding(texto: str, modelo: str = "text-embedding-3-small") -> np.ndarray:
    """
    Gera um embedding vetorial para um texto usando a API da OpenAI.
    
    O modelo text-embedding-3-small retorna vetores de 1536 dimensões
    que capturam o significado semântico do texto.
    Limite: 8192 tokens por requisição.
    
    Args:
        texto: Texto para gerar embedding
        modelo: Nome do modelo de embedding (padrão: text-embedding-3-small)
        
    Returns:
        Array numpy com o vetor de embedding
    """
    if not OPENAI_DISPONIVEL:
        # Retorna um embedding fictício para testes
        print(f"Aviso: Retornando embedding fictício (biblioteca openai não disponível)")
        return np.random.rand(1536)
    
    try:
        cliente = inicializar_cliente()
        
        # Remove espaços excessivos
        texto_limpo = texto.strip()
        if not texto_limpo:
            return np.zeros(1536)
        
        # Trunca o texto se necessário para evitar erro de limite de tokens
        texto_limpo = truncar_texto_para_embedding(texto_limpo, max_tokens=8000)
        
        # Chamada à API
        resposta = cliente.embeddings.create(
            input=texto_limpo,
            model=modelo
        )
        
        # Extrai o vetor de embedding
        embedding = np.array(resposta.data[0].embedding)
        
        return embedding
        
    except Exception as e:
        print(f"Erro ao gerar embedding: {e}")
        # Retorna vetor zero em caso de erro
        return np.zeros(1536)


def gerar_embeddings_batch(textos: List[str], modelo: str = "text-embedding-3-small") -> List[np.ndarray]:
    """
    Gera embeddings para múltiplos textos de uma vez (MUITO MAIS RÁPIDO).
    
    A API da OpenAI aceita até 2048 textos por requisição.
    Esta abordagem é 10-20x mais rápida que chamar gerar_embedding() individualmente.
    
    Args:
        textos: Lista de textos para gerar embeddings
        modelo: Nome do modelo de embedding
        
    Returns:
        Lista de arrays numpy com os embeddings
    """
    if not textos:
        return []
    
    if not OPENAI_DISPONIVEL:
        print(f"Aviso: Retornando embeddings fictícios (biblioteca openai não disponível)")
        return [np.random.rand(1536) for _ in textos]
    
    try:
        cliente = inicializar_cliente()
        
        # Limpa e trunca todos os textos
        textos_limpos = []
        for texto in textos:
            texto_limpo = texto.strip()
            if texto_limpo:
                # Trunca cada texto individualmente
                texto_limpo = truncar_texto_para_embedding(texto_limpo, max_tokens=8000)
                textos_limpos.append(texto_limpo)
            else:
                textos_limpos.append("texto vazio")  # Placeholder para textos vazios
        
        if not textos_limpos:
            return [np.zeros(1536) for _ in textos]
        
        # Chamada ÚNICA à API com todos os textos (batch processing)
        print(f"Gerando {len(textos_limpos)} embeddings em batch...")
        resposta = cliente.embeddings.create(
            input=textos_limpos,
            model=modelo
        )
        
        # Extrai todos os vetores de embedding
        embeddings = [np.array(item.embedding) for item in resposta.data]
        
        print(f"✓ {len(embeddings)} embeddings gerados com sucesso!")
        
        return embeddings
        
    except Exception as e:
        print(f"Erro ao gerar embeddings em batch: {e}")
        # Retorna vetores zero em caso de erro
        return [np.zeros(1536) for _ in textos]


def gerar_embeddings_para_chunks(chunks: List[Dict], modelo: str = "text-embedding-3-small") -> List[np.ndarray]:
    """
    Gera embeddings para uma lista de chunks de texto.
    
    OTIMIZADO: Usa batch processing para ser muito mais rápido.
    
    Args:
        chunks: Lista de dicionários contendo os chunks com campo 'texto'
        modelo: Nome do modelo de embedding
        
    Returns:
        Lista de arrays numpy com os embeddings
    """
    if not chunks:
        return []
    
    # Extrai textos dos chunks
    textos = [chunk['texto'] for chunk in chunks]
    
    # Usa batch processing (muito mais rápido)
    embeddings = gerar_embeddings_batch(textos, modelo=modelo)
    
    return embeddings


def construir_prompt_resumo(
    texto_original: str,
    chunks_selecionados: List[Dict],
    tamanho_resumo: str
) -> str:
    """
    Constrói o prompt para geração de resumo.
    
    IMPORTANTE: Limita o tamanho dos chunks para evitar exceder limite de tokens do modelo.
    
    Args:
        texto_original: Texto completo original (usado para contexto)
        chunks_selecionados: Lista de chunks mais relevantes
        tamanho_resumo: "Curto", "Medio" ou "Longo"
        
    Returns:
        Prompt formatado para o modelo
    """
    # Define instruções de tamanho
    instrucoes_tamanho = {
        "Curto": "Produza um resumo com 3 a 5 tópicos principais.",
        "Medio": "Produza um resumo com 6 a 10 tópicos, cobrindo os pontos principais e alguns detalhes.",
        "Longo": "Produza um resumo com 12 a 15 tópicos, incluindo contexto e detalhes importantes."
    }
    
    instrucao_tamanho = instrucoes_tamanho.get(tamanho_resumo, instrucoes_tamanho["Medio"])
    
    # Monta o texto dos chunks, limitando tamanho total
    # Aproximadamente 1 token = 4 caracteres, limitamos a ~25000 caracteres (6250 tokens)
    # para deixar espaço para o resto do prompt
    max_chars_chunks = 25000
    texto_chunks_parts = []
    total_chars = 0
    
    for chunk in chunks_selecionados:
        chunk_texto = chunk['texto']
        if total_chars + len(chunk_texto) > max_chars_chunks:
            # Se adicionar este chunk exceder o limite, trunca
            chars_restantes = max_chars_chunks - total_chars
            if chars_restantes > 500:  # Só adiciona se ainda houver espaço razoável
                texto_chunks_parts.append(chunk_texto[:chars_restantes] + "...")
            break
        texto_chunks_parts.append(chunk_texto)
        total_chars += len(chunk_texto) + 10  # +10 para o separador
    
    texto_chunks = "\n\n---\n\n".join(texto_chunks_parts)
    
    prompt = f"""Você é um assistente especializado em resumir textos de forma clara e acessível em português do Brasil.

Sua tarefa é criar um resumo do texto fornecido, seguindo estas REGRAS OBRIGATÓRIAS:

FORMATO DOS TÓPICOS:
- Organize em tópicos usando marcadores (bullet points)
- Cada tópico começa com um **título curto em negrito** (2-5 palavras)
- Depois do título, escreva 1 ou 2 frases explicando a ideia de forma simples
- NÃO use prefixos como "O que é:" ou "Na prática:" nos tópicos
- Deixe uma linha em branco entre cada tópico

ESTILO DE ESCRITA:
- VARIE a forma de escrever - não repita sempre a mesma estrutura de frase
- Use frases curtas e diretas (máximo 2 linhas por frase)
- Foque em explicar o que isso significa para a vida da pessoa
- Escreva de forma natural, como se estivesse conversando

TERMOS TÉCNICOS:
- EVITE números de leis (Lei nº 13.146/2015, Decreto 3.298, etc.) - só mencione se for absolutamente essencial
- Se usar uma palavra difícil, explique logo depois com palavras simples
- Exemplos de explicações:
  * "desenho universal" → explique como "produtos feitos para serem usados por qualquer pessoa"
  * "tecnologia assistiva" → explique como "equipamentos que ajudam nas atividades do dia a dia"
  * "avaliação biopsicossocial" → explique como "análise que considera saúde física, mental e situação social"

QUANTIDADE:
{instrucao_tamanho}

IMPORTANTE: Seja fiel ao conteúdo original. Não invente direitos ou informações novas.

EXEMPLO DO FORMATO ESPERADO:

• **Direito à igualdade garantido**
Existe uma lei brasileira que garante os mesmos direitos e oportunidades para pessoas com deficiência. Isso significa acesso igual a trabalho, educação e serviços públicos.

• **Acesso facilitado aos lugares**
Prédios, transportes e serviços precisam ser acessíveis para todos. Você encontra rampas, elevadores e informações em formatos que consiga usar.

• **Equipamentos que ajudam no dia a dia**
Você tem direito a usar recursos e tecnologias que facilitam sua rotina, como próteses, aplicativos especiais e outros dispositivos que aumentam sua independência.

---

Os trechos a seguir foram selecionados como mais centrais para o entendimento do documento:

{texto_chunks}

Produza o resumo em tópicos agora (siga exatamente o formato do exemplo):"""
    
    return prompt


def construir_prompt_simplificacao(
    texto_original: str,
    chunks_selecionados: List[Dict]
) -> str:
    """
    Constrói o prompt para simplificação de texto.
    
    IMPORTANTE: Limita o tamanho dos chunks para evitar exceder limite de tokens do modelo.
    
    Args:
        texto_original: Texto completo original
        chunks_selecionados: Lista de chunks mais relevantes
        
    Returns:
        Prompt formatado para o modelo
    """
    # Monta o texto dos chunks, limitando tamanho total
    # Aproximadamente 1 token = 4 caracteres, limitamos a ~25000 caracteres (6250 tokens)
    max_chars_chunks = 25000
    texto_chunks_parts = []
    total_chars = 0
    
    for chunk in chunks_selecionados:
        chunk_texto = chunk['texto']
        if total_chars + len(chunk_texto) > max_chars_chunks:
            chars_restantes = max_chars_chunks - total_chars
            if chars_restantes > 500:
                texto_chunks_parts.append(chunk_texto[:chars_restantes] + "...")
            break
        texto_chunks_parts.append(chunk_texto)
        total_chars += len(chunk_texto) + 10
    
    texto_chunks = "\n\n---\n\n".join(texto_chunks_parts)
    
    prompt = f"""Você é um assistente especializado em simplificar textos complexos em português do Brasil, tornando-os acessíveis para qualquer pessoa.

Sua tarefa é reescrever o texto fornecido, seguindo estas REGRAS OBRIGATÓRIAS:

FORMATO DOS TÓPICOS:
- Organize em tópicos usando marcadores (bullet points)
- Cada tópico começa com um **título curto em negrito** (2-5 palavras)
- Depois do título, escreva 1 ou 2 frases explicando a ideia de forma simples
- NÃO use prefixos como "O que é:" ou "Na prática:" nos tópicos
- Deixe uma linha em branco entre cada tópico

ESTILO DE ESCRITA:
- VARIE a forma de escrever - não repita sempre a mesma estrutura de frase
- Use frases curtas e diretas (máximo 2 linhas por frase)
- Foque em explicar o que isso significa para a vida da pessoa
- Escreva de forma natural, como se estivesse conversando
- Use linguagem bem simples, como se explicasse para alguém sem conhecimento técnico

SIMPLIFICAÇÃO DE PALAVRAS:
- Substitua palavras difíceis por simples:
  * "institui" → "cria"
  * "assegurar" → "garante"
  * "exercício dos direitos" → "usar seus direitos"
  * "impedimento" → "dificuldade"

TERMOS TÉCNICOS:
- EVITE números de leis (Lei nº 13.146/2015, Decreto 3.298) - NÃO mencione, a menos que seja absolutamente essencial
- Se precisar usar uma palavra técnica, explique imediatamente em seguida com linguagem simples
- Exemplos de explicações naturais:
  * "desenho universal" → "lugares e produtos feitos para qualquer pessoa usar"
  * "tecnologia assistiva" → "equipamentos como cadeiras de rodas e aplicativos que facilitam o dia a dia"
  * "avaliação biopsicossocial" → "análise que olha para a saúde física, mental e situação social da pessoa"
  * "atendimento prioritário" → "você é atendido antes na fila"

IMPORTANTE: 
- Seja fiel ao conteúdo original - não invente informações
- Não omita informações importantes, apenas simplifique como são ditas
- Mantenha todos os direitos e garantias mencionados no texto original

EXEMPLO DO FORMATO ESPERADO:

• **Igualdade de direitos garantida**
Você tem os mesmos direitos que qualquer outra pessoa no Brasil. Isso inclui acesso a trabalho, educação e todos os serviços públicos.

• **Lugares acessíveis para todos**
Prédios, transportes e serviços devem ter rampas, elevadores e outras adaptações. Você também pode pedir informações em formatos que consiga entender facilmente.

• **Ajuda de equipamentos e tecnologias**
Você pode usar recursos que facilitam sua vida, como próteses, bengalas, aplicativos de celular e outros aparelhos que aumentam sua independência.

• **Prioridade no atendimento**
Em bancos, hospitais e outros lugares, você tem direito a ser atendido antes na fila. Nos transportes, você também tem garantia de embarque seguro.

---

Os trechos a seguir foram selecionados como mais centrais do documento:

{texto_chunks}

Reescreva o texto de forma simplificada em tópicos (siga exatamente o formato do exemplo):"""
    
    return prompt


def gerar_texto_o4_mini(
    texto_original: str,
    chunks_selecionados: List[Dict],
    tipo_saida: str,
    tamanho_resumo: str = "Medio",
    modelo: str = "o4-mini"
) -> Dict:
    """
    Gera texto processado usando o modelo o4-mini da OpenAI.
    
    Args:
        texto_original: Texto completo original
        chunks_selecionados: Lista de chunks selecionados pelo pipeline
        tipo_saida: "Resumo" ou "Versao simplificada"
        tamanho_resumo: "Curto", "Medio" ou "Longo" (usado apenas para resumos)
        modelo: Nome do modelo (padrão: o4-mini)
        
    Returns:
        Dicionário contendo:
        - texto_gerado: Texto produzido pelo modelo
        - sucesso: Boolean indicando se foi bem-sucedido
        - erro: Mensagem de erro (se houver)
        - tokens_usados: Número aproximado de tokens (se disponível)
    """
    if not OPENAI_DISPONIVEL:
        return {
            'texto_gerado': "[MODO DE TESTE] Biblioteca openai não está disponível. "
                           "Este é um texto de exemplo que seria gerado pelo modelo o4-mini. "
                           "Instale a biblioteca openai e configure sua chave de API para usar o modelo real.",
            'sucesso': False,
            'erro': "Biblioteca openai não disponível",
            'tokens_usados': 0
        }
    
    try:
        cliente = inicializar_cliente()
        
        # Constrói o prompt apropriado
        if tipo_saida == "Resumo":
            prompt = construir_prompt_resumo(texto_original, chunks_selecionados, tamanho_resumo)
        else:  # Versao simplificada
            prompt = construir_prompt_simplificacao(texto_original, chunks_selecionados)
        
        # Chamada à API do modelo o4-mini
        # NOTA: O modelo o4-mini não suporta o parâmetro temperature
        # Apenas o valor padrão (1) é permitido
        resposta = cliente.chat.completions.create(
            model=modelo,
            messages=[
                {
                    "role": "system",
                    "content": "Você é um assistente útil especializado em processar textos."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_completion_tokens=2000   # Limite de tokens na resposta
        )
        
        # Extrai o texto gerado
        texto_gerado = resposta.choices[0].message.content.strip()
        
        # Informações de uso
        tokens_usados = resposta.usage.total_tokens if hasattr(resposta, 'usage') else 0
        
        return {
            'texto_gerado': texto_gerado,
            'sucesso': True,
            'erro': None,
            'tokens_usados': tokens_usados
        }
        
    except Exception as e:
        erro_msg = f"Erro ao gerar texto com o4-mini: {str(e)}"
        print(erro_msg)
        
        return {
            'texto_gerado': "",
            'sucesso': False,
            'erro': erro_msg,
            'tokens_usados': 0
        }


def verificar_disponibilidade_api() -> Dict:
    """
    Verifica se a API da OpenAI está configurada e acessível.
    
    Returns:
        Dicionário com status da disponibilidade
    """
    resultado = {
        'biblioteca_instalada': OPENAI_DISPONIVEL,
        'chave_configurada': False,
        'api_acessivel': False,
        'mensagem': ""
    }
    
    if not OPENAI_DISPONIVEL:
        resultado['mensagem'] = "Biblioteca openai não está instalada. Execute: pip install openai"
        return resultado
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        resultado['mensagem'] = "Chave da API não configurada. Defina a variável de ambiente OPENAI_API_KEY"
        return resultado
    
    resultado['chave_configurada'] = True
    
    # Tenta fazer uma chamada simples para verificar conectividade
    try:
        cliente = inicializar_cliente()
        # Testa com um embedding pequeno
        resposta = cliente.embeddings.create(
            input="teste",
            model="text-embedding-3-small"
        )
        resultado['api_acessivel'] = True
        resultado['mensagem'] = "API da OpenAI configurada e acessível"
    except Exception as e:
        resultado['mensagem'] = f"Erro ao acessar API: {str(e)}"
    
    return resultado
