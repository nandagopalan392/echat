import React, { useState } from 'react';
import {
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    Typography,
    Button,
    Box,
    TextField,
    IconButton,
    Chip,
    Alert,
    AlertTitle,
    Card,
    CardContent,
    Stack
} from '@mui/material';
import {
    Close as CloseIcon,
    ContentCopy as CopyIcon,
    Check as CheckIcon,
    OpenInNew as OpenInNewIcon,
    Security as SecurityIcon,
    NoAccounts as NoAccountsIcon,
    Warning as WarningIcon
} from '@mui/icons-material';

const GatedModelDialog = ({ isOpen, onClose, gatedModelInfo }) => {
    const [copiedStep, setCopiedStep] = useState(null);

    if (!gatedModelInfo) return null;

    const copyToClipboard = async (text, stepIndex) => {
        try {
            await navigator.clipboard.writeText(text);
            setCopiedStep(stepIndex);
            setTimeout(() => setCopiedStep(null), 2000);
        } catch (err) {
            console.error('Failed to copy text: ', err);
        }
    };

    const getErrorIcon = () => {
        switch (gatedModelInfo.error_type) {
            case 'gated_no_token':
                return <SecurityIcon color="warning" fontSize="large" />;
            case 'gated_no_access':
                return <NoAccountsIcon color="error" fontSize="large" />;
            default:
                return <WarningIcon color="warning" fontSize="large" />;
        }
    };

    const getErrorTitle = () => {
        switch (gatedModelInfo.error_type) {
            case 'gated_no_token':
                return 'Authentication Required';
            case 'gated_no_access':
                return 'Access Not Granted';
            default:
                return 'Access Restricted';
        }
    };

    const getErrorSeverity = () => {
        switch (gatedModelInfo.error_type) {
            case 'gated_no_token':
                return 'warning';
            case 'gated_no_access':
                return 'error';
            default:
                return 'warning';
        }
    };

    return (
        <Dialog
            open={isOpen}
            onClose={onClose}
            maxWidth="md"
            fullWidth
            PaperProps={{
                sx: {
                    borderRadius: 2,
                    backgroundImage: 'none',
                    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.12)'
                }
            }}
        >
            <DialogTitle sx={{ pb: 1 }}>
                <Box display="flex" justifyContent="space-between" alignItems="flex-start">
                    <Box display="flex" gap={2} alignItems="center">
                        {getErrorIcon()}
                        <Box>
                            <Typography variant="h6" component="div">
                                {getErrorTitle()}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                                Model: <Chip label={gatedModelInfo.model_name} size="small" variant="outlined" />
                            </Typography>
                        </Box>
                    </Box>
                    <IconButton onClick={onClose} size="small">
                        <CloseIcon />
                    </IconButton>
                </Box>
            </DialogTitle>

            <DialogContent dividers>
                <Stack spacing={3}>
                    <Alert severity={getErrorSeverity()}>
                        <AlertTitle>{getErrorTitle()}</AlertTitle>
                        {gatedModelInfo.message}
                    </Alert>

                    <Card variant="outlined">
                        <CardContent>
                            <Typography variant="subtitle2" gutterBottom>
                                Model Page
                            </Typography>
                            <Box display="flex" gap={1} alignItems="center">
                                <TextField
                                    fullWidth
                                    size="small"
                                    value={gatedModelInfo.model_url}
                                    InputProps={{
                                        readOnly: true,
                                        sx: { fontFamily: 'monospace', fontSize: '0.875rem' }
                                    }}
                                />
                                <IconButton
                                    onClick={() => copyToClipboard(gatedModelInfo.model_url, 'url')}
                                    size="small"
                                    color={copiedStep === 'url' ? 'success' : 'default'}
                                >
                                    {copiedStep === 'url' ? <CheckIcon /> : <CopyIcon />}
                                </IconButton>
                                <IconButton
                                    component="a"
                                    href={gatedModelInfo.model_url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    size="small"
                                    color="primary"
                                >
                                    <OpenInNewIcon />
                                </IconButton>
                            </Box>
                        </CardContent>
                    </Card>

                    <Card variant="outlined">
                        <CardContent>
                            <Typography variant="subtitle2" gutterBottom>
                                Steps to get access:
                            </Typography>
                            
                            <Box sx={{ mt: 2 }}>
                                {/* Only show token creation step if no token is available */}
                                {gatedModelInfo.error_type === 'gated_no_token' && (
                                    <>
                                        <Typography variant="body2" sx={{ fontWeight: 'bold', mb: 1 }}>
                                            Step 1: Create HuggingFace Access Token
                                        </Typography>
                                        <Typography variant="body2" paragraph>
                                            Visit: https://huggingface.co/settings/tokens
                                        </Typography>
                                        <Box display="flex" gap={1} mb={2}>
                                            <Button
                                                size="small"
                                                variant="outlined"
                                                startIcon={copiedStep === 'token-url' ? <CheckIcon /> : <CopyIcon />}
                                                onClick={() => copyToClipboard('https://huggingface.co/settings/tokens', 'token-url')}
                                                color={copiedStep === 'token-url' ? 'success' : 'primary'}
                                            >
                                                {copiedStep === 'token-url' ? 'Copied!' : 'Copy URL'}
                                            </Button>
                                            <Button
                                                size="small"
                                                variant="contained"
                                                startIcon={<OpenInNewIcon />}
                                                component="a"
                                                href="https://huggingface.co/settings/tokens"
                                                target="_blank"
                                                rel="noopener noreferrer"
                                            >
                                                Open Page
                                            </Button>
                                        </Box>
                                    </>
                                )}
                                
                                {gatedModelInfo.steps && gatedModelInfo.steps.map((step, index) => {
                                    // For gated_no_token, skip token creation steps (handled above)
                                    if (gatedModelInfo.error_type === 'gated_no_token' && 
                                        (step.includes('token') && step.includes('settings/tokens'))) {
                                        return null;
                                    }
                                    
                                    // Calculate step number based on error type
                                    const baseStepNumber = gatedModelInfo.error_type === 'gated_no_token' ? 2 : 1;
                                    const stepNumber = index + baseStepNumber;
                                    
                                    const urlMatch = step.match(/(https?:\/\/[^\s]+)/);
                                    const hasUrl = urlMatch && urlMatch[1];
                                    
                                    return (
                                        <Box key={index} sx={{ mb: 2 }}>
                                            <Typography variant="body2" sx={{ fontWeight: 'bold', mb: 1 }}>
                                                Step {stepNumber}: {step}
                                            </Typography>
                                            {hasUrl && (
                                                <Box display="flex" gap={1}>
                                                    <Button
                                                        size="small"
                                                        variant="outlined"
                                                        startIcon={<CopyIcon />}
                                                        onClick={() => copyToClipboard(hasUrl, `step-${index}`)}
                                                    >
                                                        Copy URL
                                                    </Button>
                                                    <Button
                                                        size="small"
                                                        variant="contained"
                                                        startIcon={<OpenInNewIcon />}
                                                        component="a"
                                                        href={hasUrl}
                                                        target="_blank"
                                                        rel="noopener noreferrer"
                                                    >
                                                        Open Page
                                                    </Button>
                                                </Box>
                                            )}
                                        </Box>
                                    );
                                }).filter(Boolean)}
                            </Box>
                        </CardContent>
                    </Card>

                    <Alert severity="info">
                        <AlertTitle>💡 What happens next?</AlertTitle>
                        {gatedModelInfo.error_type === 'gated_no_token' ? (
                            "After creating your access token and requesting model access, come back and try selecting the model again. The system will automatically detect your access token and proceed with the download."
                        ) : (
                            "After getting approval for model access, come back and try selecting the model again. Your access token is already configured."
                        )}
                    </Alert>
                </Stack>
            </DialogContent>

            <DialogActions sx={{ p: 3 }}>
                <Button onClick={onClose} variant="outlined">
                    Close
                </Button>
                <Button
                    component="a"
                    href={gatedModelInfo.model_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    variant="contained"
                    startIcon={<OpenInNewIcon />}
                >
                    Visit Model Page
                </Button>
            </DialogActions>
        </Dialog>
    );
};

export default GatedModelDialog;
