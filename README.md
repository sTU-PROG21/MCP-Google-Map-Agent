# Google Maps Agent with MCP & Ollama

A conversational AI agent that provides intelligent Google Maps search results using the Model Context Protocol (MCP), Ollama, and LangChain.

## 🚀 What it does

Ask natural language questions about places and get concise, well-formatted responses with ratings, locations, and recommendations.

**Example:**
- **You ask:** "Find me tea shops in HSR Layout Bangalore with good reviews"
- **Agent responds:** Organized list of top-rated tea shops with ratings, locations, and recommendations

## 🏗️ Architecture

- **Ollama**: Local LLM server running Mistral-Small model
- **MCP Google Maps**: Standardized interface to Google Maps API
- **LangChain**: Agent orchestration and tool integration
- **Python**: Main application logic
- **Node.js**: Powers the MCP Google Maps server

## 📋 Prerequisites

- **Python 3.12+**
- **Node.js 18+** 
- **Ollama** installed and running
- **Google Maps API Key**

## 🛠️ Setup Instructions

### 1. Clone and Setup Environment

```bash
git clone https://github.com/YOUR_USERNAME/MCP-Google-Map-Agent.git
cd MCP-Google-Map-Agent

# Create Python virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux  
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Install Ollama

**Windows:**
- Download installer from https://ollama.com/download/windows
- Run installer and add to PATH

**Mac/Linux:**
```bash
curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
tar -C /usr -xzf ollama-linux-amd64.tgz
```

### 3. Download the AI Model

```bash
ollama pull mistral-small:24b-instruct-2501-q4_K_M
```

### 4. Get Google Maps API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create/select a project
3. Enable "Maps JavaScript API"
4. Create credentials → API Key
5. Restrict key to Maps APIs (recommended)

### 5. Set Environment Variable

```bash
# Windows PowerShell
$env:GOOGLE_MAPS_API_KEY="your_api_key_here"

# Mac/Linux/Git Bash
export GOOGLE_MAPS_API_KEY="your_api_key_here"
```

## 🏃‍♂️ Running the Agent

### Local Testing

```python
from app import InferlessPythonModel

# Initialize the model
model = InferlessPythonModel()
model.initialize()

# Ask a question
request = RequestObjects(user_query="Find coffee shops near Times Square with good reviews")
response = model.infer(request)
print(response.generated_result)
```

### Deploy to Inferless

```bash
inferless deploy --gpu A100 --env GOOGLE_MAPS_API_KEY=your_key_here
```

## 💡 Example Queries

- "Find me pizza places in downtown Seattle with 4+ star ratings"
- "What are the best-reviewed sushi restaurants in Tokyo?"
- "Show me gyms near Central Park with good facilities"
- "Find family-friendly restaurants in Paris with outdoor seating"

## 🔧 Customization

### Modify the Response Format

Edit the `get_prompt()` method in `app.py` to change how results are formatted.

### Change the AI Model

Replace `mistral-small:24b-instruct-2501-q4_K_M` with any Ollama-compatible model:

```bash
ollama pull llama3.1:8b
# Update model_id in app.py
```

### Adjust Search Parameters

Modify the MCP Google Maps integration to filter by specific criteria (price range, distance, etc.)

## 📁 Project Structure

```
├── app.py                           # Main application logic
├── ollama_manager.py               # Ollama server management
├── inferless-runtime-config.yaml  # Deployment configuration
├── inferless.yaml                 # Inferless settings
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🎯 Next Steps / TODOs

- [ ] Add support for route planning
- [ ] Implement place details (hours, contact info)
- [ ] Add image search for places
- [ ] Create web interface
- [ ] Add caching for faster responses
- [ ] Support multiple languages

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built following the [Inferless MCP Google Maps tutorial](https://docs.inferless.com/cookbook/google-map-agent-using-mcp)
- Powered by [Ollama](https://ollama.com/) and [MCP](https://modelcontextprotocol.io/)
- Uses [LangChain](https://langchain.com/) for agent orchestration

---

**🚀 Ready to explore the world with AI? Get started by asking about your favorite places!**