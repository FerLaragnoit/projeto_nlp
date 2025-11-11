# Sistema de Resumo e Simplificação de Textos

Sistema completo de processamento de linguagem natural (NLP) para geração de resumos e simplificação de textos usando embeddings e modelos de linguagem da OpenAI.

## Características

- Interface web interativa com Streamlit
- Pipeline completo de NLP: limpeza, segmentação, chunking
- Uso de embeddings (text-embedding-3-small) para seleção inteligente de conteúdo
- Geração com LLM o4-mini da OpenAI
- Suporte a entrada de texto manual e upload de arquivos (.txt, .pdf)
- Dois modos de operação: Resumo e Simplificação

## Estrutura do Projeto

```
projeto_ac/
│
├── app.py                  # Aplicação Streamlit (interface web)
├── nlp_pipeline.py         # Pipeline de processamento de NLP
├── llm_client.py           # Cliente para API da OpenAI
├── requirements.txt        # Dependências do projeto
└── README.md              # Este arquivo
```

## Instalação

### 1. Instalar dependências

```powershell
pip install -r requirements.txt
```

### 2. Configurar chave da API da OpenAI

Você precisa de uma chave válida da API da OpenAI. Configure-a como variável de ambiente:

**No Windows PowerShell:**
```powershell
$env:OPENAI_API_KEY = "sua-chave-aqui"
```

**Ou crie um arquivo .env na raiz do projeto:**
```
OPENAI_API_KEY=sua-chave-aqui
```

Se usar .env, instale também:
```powershell
pip install python-dotenv
```

E adicione no início dos arquivos Python:
```python
from dotenv import load_dotenv
load_dotenv()
```

## Como Usar

### Executar a aplicação

```powershell
streamlit run app.py
```

A aplicação abrirá automaticamente no navegador (geralmente em `http://localhost:8501`).

### Fluxo de uso

1. **Entrada de texto:**
   - Cole o texto diretamente na área de texto, ou
   - Faça upload de um arquivo .txt ou .pdf

2. **Configurar opções (na barra lateral):**
   - Escolha o tipo de saída: Resumo ou Versão simplificada
   - Se escolheu Resumo, selecione o tamanho: Curto, Médio ou Longo
   - Opcionalmente, marque para exibir detalhes técnicos do processamento

3. **Processar:**
   - Clique em "Processar texto"
   - Aguarde o processamento (pode levar alguns segundos)
   - Visualize o resultado e as estatísticas

4. **Análise:**
   - Compare estatísticas de entrada e saída
   - Se ativou detalhes técnicos, explore informações sobre chunks e embeddings

## Arquitetura

### Pipeline de NLP (nlp_pipeline.py)

1. **Limpeza:** Remove espaços duplicados, quebras de linha excessivas
2. **Segmentação:** Divide em parágrafos e frases
3. **Chunking:** Separa texto em blocos de tamanho gerenciável com overlap
4. **Estatísticas:** Calcula métricas (palavras, frases, caracteres)

### Embeddings e Seleção (llm_client.py + nlp_pipeline.py)

1. Calcula embedding do documento completo usando text-embedding-3-small
2. Calcula embeddings individuais de cada chunk
3. Usa similaridade coseno para identificar chunks mais representativos
4. Seleciona top 70% dos chunks mais relevantes

**Por que não usamos RAG completo?**

Este MVP usa uma abordagem simplificada de seleção de chunks baseada em embeddings, que é eficiente para textos de tamanho médio. A diferença para um RAG completo:

- **Abordagem atual:** Calcula embeddings de todos os chunks, compara com embedding global, e envia os mais relevantes ao LLM em uma única chamada
- **RAG completo:** Usaria um vector database, faria busca semântica baseada em queries, e potencialmente múltiplas iterações

**Quando considerar RAG:**
- Documentos muito longos (>100k tokens)
- Múltiplos documentos para consulta
- Necessidade de citações precisas
- Sistema de perguntas e respostas

Para este caso de uso (resumo/simplificação de documentos únicos), a abordagem de chunking com seleção por embeddings é mais eficiente e direta.

### Geração com LLM (llm_client.py)

1. Monta prompt específico para resumo ou simplificação
2. Inclui apenas chunks selecionados para otimizar contexto
3. Limita o tamanho total do prompt a ~25k caracteres para evitar erros de token
4. Chama modelo o4-mini (sem parâmetro temperature, pois o modelo não suporta)
5. Retorna texto gerado com metadados

### Limitações de Tokens

- **text-embedding-3-small:** Máximo 8192 tokens por requisição (truncamos automaticamente)
- **o4-mini:** Limitamos os chunks enviados a ~25k caracteres total (~6250 tokens) para deixar espaço para instruções
- **Chunks:** Reduzidos para 500 caracteres cada para melhor granularidade

## Modelos Utilizados

- **text-embedding-3-small:** Embeddings de 1536 dimensões para análise semântica
- **o4-mini:** Modelo de linguagem para geração de texto (resumos e simplificações)

## Personalização

### Ajustar tamanho dos chunks

Em `nlp_pipeline.py`, função `criar_chunks()`:
```python
chunks = criar_chunks(texto_limpo, tamanho_chunk=1000, overlap=200)
```

### Alterar percentual de seleção

Em `nlp_pipeline.py`, função `selecionar_chunks_relevantes()`:
```python
percentual_selecao=0.7  # 70% dos chunks
```

### Modificar prompts

Em `llm_client.py`, funções:
- `construir_prompt_resumo()`
- `construir_prompt_simplificacao()`

### Ajustar parâmetros do modelo

Em `llm_client.py`, função `gerar_texto_o4_mini()`:
```python
temperature=0.3,  # Criatividade (0.0-1.0)
max_completion_tokens=2000   # Tamanho máximo da resposta
```

## Troubleshooting

### Erro: "Biblioteca openai não encontrada"
```powershell
pip install openai
```

### Erro: "Chave da API não configurada"
Verifique se definiu a variável de ambiente OPENAI_API_KEY corretamente.

### Erro ao ler PDF
```powershell
pip install PyPDF2
```

### Timeout ou erros de rede
- Verifique sua conexão com internet
- Confirme que a API da OpenAI está acessível
- Tente com um texto menor

## Limitações

- Textos muito longos podem exceder limites de contexto
- Qualidade depende dos créditos/limites da sua conta OpenAI
- PDFs com imagens ou formatação complexa podem não extrair texto corretamente

## Requisitos

- Python 3.8+
- Conta ativa na OpenAI com créditos disponíveis
- Conexão com internet

## Licença

Este projeto é um MVP educacional.

