import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Login from './components/Login';
import Register from './components/Register';
import Chat from './components/Chat';
import KnowledgeHubPage from './pages/KnowledgeHubPage';
import DocumentChunksPage from './pages/DocumentChunksPage';
import ModelSettingsPage from './pages/ModelSettingsPage';
import ManageUserPage from './pages/ManageUserPage';

const PrivateRoute = ({ children }) => {
    const isAuthenticated = !!localStorage.getItem('token');
    return isAuthenticated ? children : <Navigate to="/login" />;
};

function App() {
    return (
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
                <Route path="/" element={<Navigate to="/login" />} />
            </Routes>
        </Router>
    );
}

export default App;
