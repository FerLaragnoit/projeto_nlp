"""
app.py

Aplicação Streamlit para resumo e simplificação de textos.
Interface web interativa que permite processar textos usando NLP e modelos de linguagem.
"""

import streamlit as st
import numpy as np
from typing import Dict, List
import io

# Importa os módulos personalizados
import nlp_pipeline
import llm_client


# Configuração da página
st.set_page_config(
    page_title="Vozes acessíveis",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Estilos CSS personalizados (paleta verde claro e cinza)
st.markdown("""
<style>
    /* Cabeçalhos principais em verde */
    h1, h2, h3 {
        color: #2d6a4f;
    }
    
    /* Botões em verde */
    .stButton>button {
        background-color: #52b788;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    .stButton>button:hover {
        background-color: #40916c;
    }
    
    /* Caixas de destaque */
    .stAlert {
        background-color: #3D8737;
        border-left: 4px solid #52b788;
    }
    
    /* Sidebar em tons de cinza claro */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Métricas */
    .stMetric {
        background-color: #e9ecef;
        padding: 1rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


def extrair_texto_de_arquivo(arquivo) -> str:
    """
    Extrai texto de arquivo enviado pelo usuário (.txt ou .pdf).
    
    Args:
        arquivo: Objeto de arquivo do Streamlit
        
    Returns:
        Texto extraído do arquivo
    """
    nome_arquivo = arquivo.name.lower()
    
    try:
        if nome_arquivo.endswith('.txt'):
            # Lê arquivo de texto simples
            texto = arquivo.read().decode('utf-8', errors='ignore')
            return texto
            
        elif nome_arquivo.endswith('.pdf'):
            # Tenta importar biblioteca para PDF
            try:
                import PyPDF2
                
                # Lê o PDF
                leitor_pdf = PyPDF2.PdfReader(io.BytesIO(arquivo.read()))
                texto_completo = ""
                
                for pagina in leitor_pdf.pages:
                    texto_completo += pagina.extract_text() + "\n"
                
                return texto_completo
                
            except ImportError:
                st.error("Biblioteca PyPDF2 não está instalada. Execute: pip install PyPDF2")
                return ""
                
        else:
            st.error("Formato de arquivo não suportado. Use .txt ou .pdf")
            return ""
            
    except Exception as e:
        st.error(f"Erro ao ler arquivo: {str(e)}")
        return ""


def exibir_detalhes_nlp(chunks: List[Dict], chunks_selecionados: List[Dict], 
                        embeddings_info: Dict):
    """
    Exibe detalhes técnicos do processamento de NLP.
    
    Args:
        chunks: Lista completa de chunks
        chunks_selecionados: Lista de chunks selecionados
        embeddings_info: Informações sobre os embeddings calculados
    """
    st.subheader("Detalhes do Processamento de NLP")
    
    # Informações sobre chunks
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total de chunks gerados", len(chunks))
        st.metric("Chunks selecionados", len(chunks_selecionados))
    
    with col2:
        percentual = (len(chunks_selecionados) / len(chunks) * 100) if chunks else 0
        st.metric("Percentual selecionado", f"{percentual:.1f}%")
        st.metric("Dimensão dos embeddings", embeddings_info.get('dimensao', 'N/A'))
    
    # Informações sobre embeddings
    st.write("**Informações de Embeddings:**")
    st.write(f"- Modelo usado: {embeddings_info.get('modelo', 'N/A')}")
    st.write(f"- Número de embeddings calculados: {embeddings_info.get('num_embeddings', 0)}")
    
    # Mostra trechos dos chunks selecionados
    st.write("**Chunks selecionados (prévia):**")
    
    for chunk in chunks_selecionados[:5]:  # Mostra no máximo 5
        with st.expander(f"Chunk {chunk['indice']} - Similaridade: {chunk.get('similaridade', 0):.3f}"):
            # Mostra apenas as primeiras 300 caracteres
            trecho = chunk['texto'][:300]
            if len(chunk['texto']) > 300:
                trecho += "..."
            st.text(trecho)
            st.caption(f"Tamanho: {chunk['tamanho']} caracteres | Palavras: {chunk['num_palavras']}")


def processar_texto(texto_entrada: str, tipo_saida: str, tamanho_resumo: str,
                   mostrar_detalhes: bool) -> Dict:
    """
    Executa o pipeline completo de processamento do texto.
    
    Args:
        texto_entrada: Texto fornecido pelo usuário
        tipo_saida: "Resumo" ou "Versao simplificada"
        tamanho_resumo: "Curto", "Medio" ou "Longo"
        mostrar_detalhes: Se deve retornar detalhes técnicos
        
    Returns:
        Dicionário com resultados do processamento
    """
    resultado = {
        'sucesso': False,
        'texto_gerado': "",
        'estatisticas_entrada': {},
        'estatisticas_saida': {},
        'chunks': [],
        'chunks_selecionados': [],
        'embeddings_info': {},
        'erro': None
    }
    
    try:
        # Etapa 1: Pré-processamento
        with st.spinner("Processando texto (limpeza e segmentação)..."):
            # Usa chunks menores (500 chars) para evitar problemas com limite de tokens
            texto_limpo, chunks, estatisticas = nlp_pipeline.processar_texto_completo(
                texto_entrada,
                percentual_selecao=0.7,
                tamanho_chunk=500  # Reduzido para 500 caracteres por chunk
            )
            
            resultado['estatisticas_entrada'] = estatisticas
            resultado['chunks'] = chunks
        
        # Etapa 2: Embeddings
        with st.spinner("Calculando embeddings..."):
            # Gera embedding do documento completo
            embedding_global = llm_client.gerar_embedding(texto_limpo)
            
            # Gera embeddings de cada chunk
            embeddings_chunks = llm_client.gerar_embeddings_para_chunks(chunks)
            
            resultado['embeddings_info'] = {
                'modelo': 'text-embedding-3-small',
                'dimensao': len(embedding_global),
                'num_embeddings': len(embeddings_chunks) + 1  # chunks + global
            }
        
        # Etapa 3: Seleção de chunks relevantes
        with st.spinner("Selecionando trechos mais relevantes..."):
            chunks_selecionados = nlp_pipeline.selecionar_chunks_relevantes(
                chunks,
                embeddings_chunks,
                embedding_global,
                percentual_selecao=0.7
            )
            
            resultado['chunks_selecionados'] = chunks_selecionados
        
        # Etapa 4: Geração com LLM
        with st.spinner(f"Gerando {tipo_saida.lower()} com o4-mini..."):
            resposta_llm = llm_client.gerar_texto_o4_mini(
                texto_limpo,
                chunks_selecionados,
                tipo_saida,
                tamanho_resumo
            )
            
            if resposta_llm['sucesso']:
                resultado['texto_gerado'] = resposta_llm['texto_gerado']
                resultado['sucesso'] = True
                
                # Calcula estatísticas do texto gerado
                resultado['estatisticas_saida'] = nlp_pipeline.calcular_estatisticas(
                    resposta_llm['texto_gerado']
                )
            else:
                resultado['erro'] = resposta_llm.get('erro', 'Erro desconhecido')
        
    except Exception as e:
        resultado['erro'] = f"Erro durante processamento: {str(e)}"
    
    return resultado




def main():
    """
    Função principal da aplicação Streamlit.
    """
    # Cabeçalho
    st.title("Vozes acessíveis")
    st.markdown("Ferramenta de processamento de linguagem natural para tornar textos mais acessíveis")
    
    # Sidebar - Configurações
    with st.sidebar:
        st.header("Configurações")
        
        # Tipo de saída
        tipo_saida = st.radio(
            "Tipo de saída:",
            options=["Resumo", "Versao simplificada"],
            help="Escolha entre gerar um resumo objetivo ou uma versão simplificada do texto"
        )
        
        # Tamanho do resumo (apenas para resumos)
        tamanho_resumo = "Medio"
        if tipo_saida == "Resumo":
            tamanho_resumo = st.select_slider(
                "Tamanho do resumo:",
                options=["Curto", "Medio", "Longo"],
                value="Medio",
                help="Define o nível de detalhe do resumo gerado"
            )
        
        # Opção de mostrar detalhes técnicos
        mostrar_detalhes = st.checkbox(
            "Exibir detalhes do processamento de NLP",
            value=False,
            help="Mostra informações técnicas sobre chunks, embeddings e seleção"
        )
        
        st.divider()
        
        # Verificação de disponibilidade da API
        st.subheader("Status da API")
        status_api = llm_client.verificar_disponibilidade_api()
        
        if status_api['api_acessivel']:
            st.success("API OpenAI: Configurada")
        elif status_api['chave_configurada']:
            st.warning("API OpenAI: Chave configurada, mas não testada")
        else:
            st.error("API OpenAI: Não configurada")
            st.caption(status_api['mensagem'])
    
    # Área principal
    st.header("Entrada de Texto")
    
    # Abas para diferentes formas de entrada
    aba_texto, aba_arquivo = st.tabs(["Colar texto", "Enviar arquivo"])
    
    texto_entrada = ""
    
    with aba_texto:
        texto_entrada = st.text_area(
            "Cole seu texto aqui:",
            height=300,
            placeholder="Digite ou cole o texto que deseja processar...",
            help="Área para entrada manual de texto"
        )
    
    with aba_arquivo:
        arquivo = st.file_uploader(
            "Ou envie um arquivo de texto ou PDF:",
            type=['txt', 'pdf'],
            help="Formatos suportados: .txt e .pdf"
        )
        
        if arquivo is not None:
            st.info(f"Arquivo carregado: {arquivo.name}")
            texto_extraido = extrair_texto_de_arquivo(arquivo)
            
            if texto_extraido:
                texto_entrada = texto_extraido
                st.success(f"Texto extraído com sucesso ({len(texto_extraido)} caracteres)")
    
    # Mostra prévia do texto de entrada se houver
    if texto_entrada:
        with st.expander("Visualizar texto de entrada"):
            st.text_area("Texto que será processado:", texto_entrada, height=150, disabled=True)
    
    # Botão de processamento
    st.divider()
    
    col_botao1, col_botao2, col_botao3 = st.columns([1, 2, 1])
    
    with col_botao2:
        processar = st.button("Processar texto", use_container_width=True, type="primary")
    
    # Processamento
    if processar:
        if not texto_entrada or len(texto_entrada.strip()) < 50:
            st.error("Por favor, forneça um texto com pelo menos 50 caracteres.")
        else:
            # Executa o processamento
            resultado = processar_texto(
                texto_entrada,
                tipo_saida,
                tamanho_resumo,
                mostrar_detalhes
            )
            
            if resultado['sucesso']:
                st.success("Processamento concluído com sucesso!")
                
                # Exibe resultado
                st.header("Resultado")
                
                # Caixa com o texto gerado
                st.subheader(f"{tipo_saida} Gerado")
                st.markdown(f"""
                <div style="background-color: #3D8737; padding: 1.5rem; 
                            border-radius: 10px; border-left: 5px solid #52b788; color: white;">
                {resultado['texto_gerado']}
                </div>
                """, unsafe_allow_html=True)
                
                # Estatísticas comparativas
                st.subheader("Estatísticas")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Palavras (entrada)",
                        resultado['estatisticas_entrada'].get('num_palavras', 0)
                    )
                
                with col2:
                    st.metric(
                        "Palavras (saída)",
                        resultado['estatisticas_saida'].get('num_palavras', 0)
                    )
                
                with col3:
                    st.metric(
                        "Frases (entrada)",
                        resultado['estatisticas_entrada'].get('num_frases', 0)
                    )
                
                with col4:
                    st.metric(
                        "Frases (saída)",
                        resultado['estatisticas_saida'].get('num_frases', 0)
                    )
                
                # Exibe detalhes técnicos se solicitado
                if mostrar_detalhes:
                    st.divider()
                    exibir_detalhes_nlp(
                        resultado['chunks'],
                        resultado['chunks_selecionados'],
                        resultado['embeddings_info']
                    )
            
            else:
                st.error("Erro durante o processamento:")
                st.error(resultado['erro'])
                
                # Sugestões de solução
                st.info("""
                **Possíveis soluções:**
                - Verifique se a chave da API da OpenAI está configurada corretamente
                - Confirme que você tem créditos disponíveis na sua conta OpenAI
                - Tente com um texto menor
                - Verifique sua conexão com a internet
                """)
    
    # Rodapé
    st.divider()
    st.caption("Sistema de Resumo e Simplificação de Textos | Processamento com NLP e LLM")


if __name__ == "__main__":
    main()
