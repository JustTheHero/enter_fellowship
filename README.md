Enter_fellowship


# Enter Fellowship


## 📁 Estrutura de Pastas

* `/frontend`: A aplicação React (interface do usuário).
* `api.py`: O servidor backend (FastAPI).
* `doc_extraction.py`: A lógica principal de extração de dados.

## 📋 Pré-requisitos

* [Python](https://www.python.org/downloads/) (3.8+)
* [Node.js](https://nodejs.org/en) (18+)
* Uma chave de API da openAI

## ⚙️ Configuração (Setup)

Você precisa configurar o backend (Python) e o frontend (React) uma vez.

### 1. Backend (Python)

1.  **Ative o ambiente virtual (`venv`)**
    * macOS/Linux: `source venv/bin/activate`
    * Windows: `.\venv\Scripts\activate`

2.  **Instale as dependências**
    ```bash
    pip install fastapi uvicorn "python-multipart[all]" openai pymupdf
    ```

3.  **Defina a chave da OpenAI**
    * macOS/Linux: `export OPENAI_API_KEY="sk-chave"`
    * Windows: `$env:OPENAI_API_KEY="sk-chave"`

### 2. Frontend (React)

1.  **Entre na pasta `frontend`**
    ```bash
    cd frontend
    ```
2.  **Instale as dependências**
    ```bash
    npm install
    ```
3.  **Volte para a pasta raiz**
    ```bash
    cd ..
    ```

## 🚀 Como Executar

Você precisará de **dois terminais**.

### Terminal 1: Rodar o Backend

1.  Na pasta raiz, ative o `venv` (passo 1.1 da configuração).
2.  Defina sua chave de API (passo 1.3 da configuração).
3.  Inicie o servidor:
    ```bash
    python api.py
    ```
4.  O backend estará rodando em `http://localhost:8000`.

### Terminal 2: Rodar o Frontend

1.  Vá para a pasta `frontend`:
    ```bash
    cd frontend
    ```
2.  Inicie o servidor de desenvolvimento:
    ```bash
    npm run dev
    ```
3.  O frontend estará disponível em `http://localhost:5173`.

## Como Usar

1.  Abra `http://localhost:5173` no seu navegador.
2.  **Schema:** Defina um nome (Label) e os campos que deseja extrair.
3.  **Documentos:** Arraste seus arquivos PDF.
4.  Clique em **Processar**.
5.  Veja os resultados na parte inferior.
