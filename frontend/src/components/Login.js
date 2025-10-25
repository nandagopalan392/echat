import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
    Box, 
    Card, 
    CardContent, 
    TextField, 
    Button, 
    Typography, 
    Alert, 
    Container,
    Paper,
    InputAdornment,
    IconButton,
    ThemeProvider
} from '@mui/material';
import { Visibility, VisibilityOff, Person, Lock } from '@mui/icons-material';
import { authService } from '../services/authService';
import { theme } from '../theme';
import echatLogo from '../assets/echat_logo.svg';

const Login = () => {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [loading, setLoading] = useState(false);
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setLoading(true);
        
        try {
            await authService.login(username, password);
            console.log('✅ Login successful, tokens stored securely');
            navigate('/chat');
        } catch (error) {
            setError(error.message || 'Invalid username or password');
            console.error('❌ Login error:', error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <ThemeProvider theme={theme}>
            <Box
                sx={{
                    minHeight: '100vh',
                    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    padding: 2,
                    position: 'relative',
                    '&::before': {
                        content: '""',
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        right: 0,
                        bottom: 0,
                        background: 'url("data:image/svg+xml,%3Csvg width="60" height="60" viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg"%3E%3Cg fill="none" fill-rule="evenodd"%3E%3Cg fill="%23ffffff" fill-opacity="0.05"%3E%3Ccircle cx="7" cy="7" r="1"/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")',
                        animation: 'float 20s ease-in-out infinite',
                    },
                }}
            >
                <Container maxWidth="sm">
                    <Card
                        elevation={0}
                        sx={{
                            background: 'rgba(255, 255, 255, 0.95)',
                            backdropFilter: 'blur(20px)',
                            borderRadius: 4,
                            border: '1px solid rgba(255, 255, 255, 0.2)',
                            overflow: 'visible',
                            position: 'relative',
                        }}
                    >
                        <CardContent sx={{ p: 6 }}>
                            <Box sx={{ textAlign: 'center', mb: 4 }}>
                                <Box
                                    component="img"
                                    src={echatLogo}
                                    alt="eChat Logo"
                                    sx={{
                                        height: 64,
                                        width: 'auto',
                                        mb: 3,
                                        filter: 'drop-shadow(0 4px 8px rgba(0,0,0,0.1))',
                                    }}
                                />
                                <Typography
                                    variant="h4"
                                    component="h1"
                                    sx={{
                                        fontWeight: 700,
                                        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                                        backgroundClip: 'text',
                                        WebkitBackgroundClip: 'text',
                                        WebkitTextFillColor: 'transparent',
                                        mb: 1,
                                    }}
                                >
                                    Welcome Back
                                </Typography>
                                <Typography variant="body1" color="text.secondary">
                                    Sign in to continue to eChat
                                </Typography>
                            </Box>

                            {error && (
                                <Alert 
                                    severity="error" 
                                    sx={{ 
                                        mb: 3,
                                        borderRadius: 2,
                                        '& .MuiAlert-message': {
                                            fontWeight: 500,
                                        },
                                    }}
                                >
                                    {error}
                                </Alert>
                            )}

                            <Box component="form" onSubmit={handleSubmit} sx={{ mt: 2 }}>
                                <TextField
                                    fullWidth
                                    label="Username"
                                    variant="outlined"
                                    value={username}
                                    onChange={(e) => setUsername(e.target.value)}
                                    required
                                    sx={{ mb: 3 }}
                                    InputProps={{
                                        startAdornment: (
                                            <InputAdornment position="start">
                                                <Person color="action" />
                                            </InputAdornment>
                                        ),
                                        sx: {
                                            '& fieldset': {
                                                borderColor: 'rgba(0, 0, 0, 0.12)',
                                            },
                                            '&:hover fieldset': {
                                                borderColor: 'primary.main',
                                            },
                                        },
                                    }}
                                />
                                
                                <TextField
                                    fullWidth
                                    label="Password"
                                    type={showPassword ? 'text' : 'password'}
                                    variant="outlined"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    required
                                    sx={{ mb: 4 }}
                                    InputProps={{
                                        startAdornment: (
                                            <InputAdornment position="start">
                                                <Lock color="action" />
                                            </InputAdornment>
                                        ),
                                        endAdornment: (
                                            <InputAdornment position="end">
                                                <IconButton
                                                    onClick={() => setShowPassword(!showPassword)}
                                                    edge="end"
                                                >
                                                    {showPassword ? <VisibilityOff /> : <Visibility />}
                                                </IconButton>
                                            </InputAdornment>
                                        ),
                                        sx: {
                                            '& fieldset': {
                                                borderColor: 'rgba(0, 0, 0, 0.12)',
                                            },
                                            '&:hover fieldset': {
                                                borderColor: 'primary.main',
                                            },
                                        },
                                    }}
                                />

                                <Button
                                    type="submit"
                                    fullWidth
                                    variant="contained"
                                    size="large"
                                    disabled={loading}
                                    sx={{
                                        py: 1.5,
                                        fontSize: '1.1rem',
                                        fontWeight: 600,
                                        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                                        boxShadow: '0 4px 15px rgba(102, 126, 234, 0.4)',
                                        '&:hover': {
                                            background: 'linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%)',
                                            boxShadow: '0 6px 20px rgba(102, 126, 234, 0.6)',
                                            transform: 'translateY(-2px)',
                                        },
                                        '&:disabled': {
                                            background: 'rgba(0, 0, 0, 0.12)',
                                            boxShadow: 'none',
                                        },
                                    }}
                                >
                                    {loading ? 'Signing In...' : 'Sign In'}
                                </Button>
                            </Box>
                        </CardContent>
                    </Card>
                </Container>
            </Box>
        </ThemeProvider>
    );
};

export default Login;
