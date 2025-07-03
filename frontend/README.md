# ChatPDF Frontend

## Setup

1. Install dependencies:
```bash
npm install
```

2. Create a `.env` file in the frontend directory with:
```bash
REACT_APP_API_URL=http://localhost:5000
```

## Running the Application

To start the development server:
```bash
npm start
```

The application will open in your browser at [http://localhost:3000](http://localhost:3000).

## Troubleshooting Setup

If you encounter ENOENT errors:

1. Verify you're in the correct directory:
```bash
cd frontend
```

2. Delete node_modules and package-lock.json if they exist:
```bash
rm -rf node_modules package-lock.json
```

3. Clear npm cache:
```bash
npm cache clean --force
```

4. Reinstall dependencies:
```bash
npm install
```
