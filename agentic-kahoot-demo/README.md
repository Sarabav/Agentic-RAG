# CrewAI Kahoot Bot

A persistent AI agent that automatically plays Kahoot quizzes using CrewAI Flows and Browser MCP.

## Features

- **Persistent Agent**: Stays connected throughout entire game sessions
- **Event-Driven Architecture**: Uses CrewAI Flow with `@start()` and `@listen()` decorators  
- **Browser Automation**: Leverages Browser MCP for DOM interaction
- **Fast Response**: Aims to answer within 2 seconds using cached knowledge + LLM reasoning
- **Multi-Agent Collaboration**: Navigator, Parser, Knowledge Guru, and Clicker agents work together

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CrewAI Flow   │    │   Browser MCP   │    │   OpenRouter    │
│   Orchestrator  │◄──►│     Server      │◄──►│      LLM        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│     Agents      │    │    Kahoot.it    │
│ • Navigator     │    │   (Persistent   │
│ • Parser        │    │    Session)     │
│ • Knowledge     │    └─────────────────┘
│ • Clicker       │
└─────────────────┘
```

## Python Environment Setup

Setting up a dedicated Python environment is highly recommended to avoid dependency conflicts. The recommended approach is to use **Miniconda** (lightweight Anaconda distribution). You may use Anaconda or `virtualenv` if preferred, but detailed steps below use Miniconda.

### 1. Install Miniconda

- Download the Miniconda installer for your OS: [Miniconda Download Page](https://docs.conda.io/en/latest/miniconda.html)
- Follow the installation instructions for your platform.

### 2. Create a New Environment (Python 3.12.3, conda-forge)

Open a terminal (Anaconda Prompt, Command Prompt, or terminal of your choice) and run:

```bash
conda create -n kahoot-bot python=3.12.3 -c conda-forge
```

### 3. Activate the Environment

```bash
conda activate kahoot-bot
```

### 4. Verify Python Version

```bash
python --version
```
- Output should be `Python 3.12.3`

> **Note:** You may use Anaconda or `virtualenv` if you prefer, but Miniconda is recommended for most users due to its lightweight footprint and ease of use.

---

### Alternative: Using Python Virtual Environment (`venv`)

If you prefer not to use Miniconda, you can set up a dedicated Python environment using Python's built-in `venv` module. This approach is fully supported and works on all platforms.

#### 1. Create a Virtual Environment

Open a terminal in the project directory and run:

```bash
python -m venv venv
```

This will create a new directory named `venv` containing the isolated Python environment.

#### 2. Activate the Virtual Environment

- **On Windows:**
  ```cmd
  venv\Scripts\activate
  ```
- **On macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```

#### 3. Verify Python Version

```bash
python --version
```
- Output should be `Python 3.12.3`

#### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** You can use either Miniconda or Python's built-in `venv` to manage your environment. Both approaches are supported. Miniconda is recommended for most users, but `venv` is a lightweight alternative that works with any standard Python installation.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Environment Variables

Create a `.env` file with the following:

```env
# OPENROUTER_API_KEY is required for LLM (question answering) via OpenRouter
OPENROUTER_API_KEY=your_openrouter_api_key_here

# OPENAI_API_KEY is required for embeddings (document search); also used if you want to use OpenAI LLMs directly
OPENAI_API_KEY=your_openai_api_key_here

# Kahoot Game Configuration (optional, can be set at runtime)
KAHOOT_PIN=435949
KAHOOT_NICKNAME=CrewAI_Bot
```
- `OPENROUTER_API_KEY` is required for LLM-powered question answering (via OpenRouter).
- `OPENAI_API_KEY` is required for document embeddings (vector search) and is also used if you select an OpenAI LLM model.

### 3. Node.js Requirement

- **Node.js v21.7.3 or later must be installed** (required for Browser MCP extension).
- [Download Node.js](https://nodejs.org/) if not already installed.

> **Note:** You do **not** need to manually install or start the Browser MCP server.
> The Python code will automatically launch the Browser MCP extension as needed.


### 4. Run the Kahoot Bot

Follow this step-by-step flow to start the Kahoot bot:

1. Run the bot:
   ```bash
   python kahoot_bot.py
   ```
2. When prompted, select the RAG collection to use for question answering.
3. When prompted to "press Enter to continue," **switch to your browser** and:
   - Open [kahoot.it](https://kahoot.it/) and enter the quiz PIN provided by the bot.
   - Join the game with your chosen nickname.
   - Connect the Browser MCP extension (follow the on-screen instructions if needed).
4. **Only press Enter in the terminal when the quiz question is visible on your screen.**
5. The bot will answer the question automatically, then wait for your input to continue to the next question.

Repeat steps 4–5 for each question in the quiz.


## Using `chromadb_manager.py`

The `chromadb_manager.py` script provides an **interactive, menu-driven interface** for managing ChromaDB collections and documents.

### Usage

1. Run the script:
   ```bash
   python chromadb_manager.py
   ```
2. Follow the on-screen menu prompts to:
   - List all collections
   - Create or select a collection
   - Add documents (with optional metadata)
   - Query a collection
   - View collection stats
   - Delete a collection

All actions are performed interactively—no CLI subcommands are required. Simply select the desired option from the menu and follow the prompts.


## How It Works

### Flow Architecture

The bot uses **CrewAI Flow** (not Sequential or Hierarchical) because:

- **Event-driven**: Reacts to DOM changes and timer updates
- **Parallel processing**: Can overlap LLM calls with DOM operations  
- **Real-time responsive**: Sub-second reaction times
- **State management**: Maintains game context between questions
- **Conditional routing**: Different paths for cache hits vs misses

### Agent Roles

1. **Navigator**: Joins game, manages browser session, handles reconnects
2. **Parser**: Extracts question text and answer choices from DOM
3. **Knowledge Guru**: Decides best answer (cache → LLM → fallback)
4. **Clicker**: Maps answer to DOM element and executes click

### Flow Execution

```python
@start()
def initialize_bot():
    # Get PIN, join game

@listen(initialize_bot) 
def join_game():
    # Navigate to kahoot.it, enter credentials

@listen(join_game)
def monitor_and_play():
    # Main game loop with question detection
```

## Performance Strategy

- **DOM-only approach**: No screenshots needed, pure text extraction
- **Speed targets**: Complete pipeline in < 2 seconds
- **Cache system**: Store Q→A pairs for instant recall
- **Model selection**: Fast models (gpt-4o-mini) for time pressure
- **Fallback logic**: Heuristic guessing if LLM times out

## Customization

### Different LLM Providers

```env
# Use Anthropic Claude
OPENROUTER_MODEL=anthropic/claude-3-haiku

# Use local Ollama
OPENROUTER_MODEL=ollama/llama3.2:latest
OPENROUTER_API_KEY=ollama
OPENROUTER_BASE_URL=http://localhost:11434/v1
```


## Troubleshooting

### Game Join Failures

1. Verify Kahoot PIN is active
2. Check network connectivity
3. Ensure browser can access kahoot.it

### Slow Response Times

1. Use faster LLM models (gpt-4o-mini vs gpt-4o)
2. Reduce max_tokens in LLM calls
3. Implement better caching strategy

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -m 'Add enhancement'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Open Pull Request

## License

MIT License - see LICENSE file for details.

## Disclaimer

This bot is for educational purposes only. Ensure you have permission to use automation tools in your Kahoot games. Respect the terms of service of all platforms used. 