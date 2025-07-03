# eChat - AI Chat Application

A modern chat application with AI capabilities using Ollama, MLflow, and monitoring.

## Prerequisites

1. Docker and Docker Compose
2. NVIDIA GPU with CUDA support (for optimal performance)
3. NVIDIA Container Toolkit

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/nandagopalan392/echat.git
cd echat
```

2. Launch the application:
```bash
# Start all services
docker compose up -d

# Wait for all services to initialize (about 1-2 minutes)
```

## Accessing Services


All services are configured to run on localhost:

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- Ollama: http://localhost:11434
- MinIO API: http://localhost:9100
- MinIO Console: http://localhost:9101

## Development

### Frontend Development

```bash
cd frontend
npm install
npm start
```

### Backend Development

```bash
cd backend
pip install -r requirements.txt
python main.py
```


## Troubleshooting

1. If services fail to start, check logs:
```bash
docker compose logs -f
```

2. Reset everything and start fresh:
```bash
docker compose down -v
docker compose up -d
```

## Contribution

We welcome contributions to eChat! Please follow these steps to contribute:

### How to Contribute

1. **Fork the repository**
   ```bash
   # Click "Fork" on GitHub, then clone your fork
   git clone https://github.com/your-username/echat.git
   cd echat
   ```

2. **Set up the development environment**
   ```bash
   # Add the original repository as upstream
   git remote add upstream https://github.com/nandagopalan392/echat.git
   
   # Create a virtual environment for backend development
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   
   # Install frontend dependencies
   cd ../frontend
   npm install
   ```

3. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make your changes**
   - Follow the existing code style and conventions
   - Add tests for new functionality
   - Update documentation as needed

5. **Test your changes**
   ```bash
   # Start the development environment
   docker compose up -d
   
   # Test backend changes
   cd backend
   python -m pytest  # If tests are available
   
   # Test frontend changes
   cd frontend
   npm test  # If tests are available
   ```

6. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

7. **Push to your fork and create a Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then go to GitHub and create a Pull Request from your fork.

### Contribution Guidelines

- **Code Style**: Follow PEP 8 for Python code and ESLint rules for JavaScript
- **Commit Messages**: Use conventional commit format (feat:, fix:, docs:, etc.)
- **Documentation**: Update README.md and inline comments for any new features
- **Testing**: Add appropriate tests for new functionality
- **Security**: Never commit sensitive information like API keys or passwords

### Areas for Contribution

- 🐛 Bug fixes and improvements
- ✨ New AI model integrations
- 🎨 UI/UX improvements
- 📚 Documentation enhancements
- 🧪 Test coverage improvements
- 🔧 DevOps and deployment optimizations

### Getting Help

- Open an issue for bug reports or feature requests
- Join discussions in existing issues
- Contact maintainers for questions about contributing

By contributing, you agree to the terms of the Contributor License Agreement.