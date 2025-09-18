import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, CssBaseline } from '@mui/material';
import { theme } from './theme';
import Login from './components/Login';
import Register from './components/Register';
import Chat from './components/Chat';
import KnowledgeHubPage from './pages/KnowledgeHubPage';
import DocumentChunksPage from './pages/DocumentChunksPage';
import ModelSettingsPage from './pages/ModelSettingsPage';
import ManageUserPage from './pages/ManageUserPage';
import EvaluationPage from './pages/EvaluationPage';
import FinetuningPage from './pages/FinetuningPage';

const PrivateRoute = ({ children }) => {
    const isAuthenticated = !!localStorage.getItem('token');
    return isAuthenticated ? children : <Navigate to="/login" />;
};

function App() {
    return (
        <ThemeProvider theme={theme}>
            <CssBaseline />
            <Router>
                <Routes>
                    <Route path="/login" element={<Login />} />
                    <Route path="/register" element={<Register />} />
                <Route
                    path="/chat"
                    element={
                        <PrivateRoute>
                            <Chat />
                        </PrivateRoute>
                    }
                />
                <Route
                    path="/knowledge-hub"
                    element={
                        <PrivateRoute>
                            <KnowledgeHubPage />
                        </PrivateRoute>
                    }
                />
                <Route
                    path="/documents/:docId/chunks"
                    element={
                        <PrivateRoute>
                            <DocumentChunksPage />
                        </PrivateRoute>
                    }
                />
                <Route
                    path="/model-settings"
                    element={
                        <PrivateRoute>
                            <ModelSettingsPage />
                        </PrivateRoute>
                    }
                />
                <Route
                    path="/manage-users"
                    element={
                        <PrivateRoute>
                            <ManageUserPage />
                        </PrivateRoute>
                    }
                />
                <Route
                    path="/evaluation"
                    element={
                        <PrivateRoute>
                            <EvaluationPage />
                        </PrivateRoute>
                    }
                />
                <Route
                    path="/finetuning"
                    element={
                        <PrivateRoute>
                            <FinetuningPage />
                        </PrivateRoute>
                    }
                />
                <Route path="/" element={<Navigate to="/login" />} />
            </Routes>
        </Router>
        </ThemeProvider>
    );
}

export default App;
