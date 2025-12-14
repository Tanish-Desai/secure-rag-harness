# Variables with defaults
attack ?= none
defense ?= baseline
profile ?= P1
topology ?= sequential
seed ?= 42
limit ?= 10

.PHONY: setup setup-llm up down test run help

# 1. Setup: Ensures environment configuration exists
setup: setup-llm
	@if [ ! -f .env ]; then \
		echo "Creating .env from template..."; \
		echo "# Auto-generated .env" > .env; \
		echo "LLM_API_BASE=http://host.docker.internal:11434/v1" >> .env; \
		echo "LLM_MODEL_NAME=secure-rag-llama3" >> .env; \
		echo ""; \
		echo "⚠️  WARNING: If running on WSL2, update LLM_API_BASE in .env"; \
		echo "   to use your eth0 IP (run 'ip addr show eth0')."; \
	else \
		echo "✅ .env file already exists."; \
	fi

# 1b. LLM Setup: Generates absolute-path Modelfile
setup-llm:
	@echo "🔧 Configuring Local LLM..."
	@if [ ! -f services/llm/weights/llama3.gguf ]; then \
		echo "❌ GGUF weights not found at services/llm/weights/llama3.gguf"; \
		echo "   Please run the wget command from README."; \
		exit 1; \
	fi
	@echo "📝 Generating Modelfile with absolute paths..."
	@# Replaces __WEIGHTS_DIR__ with the output of $(PWD)/services/llm/weights
	@sed "s|__WEIGHTS_DIR__|$(PWD)/services/llm/weights|g" services/llm/Modelfile.template > services/llm/Modelfile
	@echo "🧠 Creating Ollama Model 'secure-rag-llama3'..."
	@ollama create secure-rag-llama3 -f services/llm/Modelfile
	@echo "✅ Model 'secure-rag-llama3' created successfully!"

# 2. Infrastructure Management
up: setup
	docker-compose up --build -d

down:
	docker-compose down

# 3. Smoke Test (Manual verification)
test:
	curl -X POST http://localhost:8000/chat \
	-H "Content-Type: application/json" \
	-d '{"query": "Hello RAG", "topology": "sequential"}'

# 4. The Experiment Runner (The "Harness")
run:
	python3 harness/main.py \
		--attack=$(attack) \
		--defense=$(defense) \
		--profile=$(profile) \
		--topology=$(topology) \
		--seed=$(seed) \
		--limit=$(limit)

# Help command
help:
	@echo "Secure RAG Harness - Usage:"
	@echo "  make setup       Create config files & build LLM"
	@echo "  make up          Start infrastructure"
	@echo "  make test        Run a single manual smoke test"
	@echo "  make run         Execute an experiment suite"