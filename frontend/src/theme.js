import { createTheme } from '@mui/material/styles';

// Premium Modern Theme - Sophisticated Colors
export const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#2563eb', // Premium blue
      light: '#60a5fa',
      dark: '#1d4ed8',
      contrastText: '#ffffff',
    },
    secondary: {
      main: '#64748b', // Sophisticated slate
      light: '#94a3b8',
      dark: '#475569',
      contrastText: '#ffffff',
    },
    background: {
      default: '#f8fafc',
      paper: '#ffffff',
    },
    text: {
      primary: '#0f172a',
      secondary: '#64748b',
      disabled: '#94a3b8',
    },
    divider: 'rgba(148, 163, 184, 0.2)',
    success: {
      main: '#059669',
      light: '#10b981',
      dark: '#047857',
    },
    warning: {
      main: '#d97706',
      light: '#f59e0b',
      dark: '#b45309',
    },
    error: {
      main: '#dc2626',
      light: '#ef4444',
      dark: '#b91c1c',
    },
    info: {
      main: '#0891b2',
      light: '#06b6d4',
      dark: '#0e7490',
    },
  },
  typography: {
    fontFamily: '"Inter", "SF Pro Display", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", sans-serif',
    h1: {
      fontSize: '2.75rem',
      fontWeight: 700,
      lineHeight: 1.2,
      color: '#0f172a',
      letterSpacing: '-0.025em',
    },
    h2: {
      fontSize: '2.25rem',
      fontWeight: 600,
      lineHeight: 1.3,
      color: '#0f172a',
      letterSpacing: '-0.02em',
    },
    h3: {
      fontSize: '1.875rem',
      fontWeight: 600,
      lineHeight: 1.3,
      color: '#0f172a',
      letterSpacing: '-0.015em',
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 600,
      lineHeight: 1.4,
      color: '#0f172a',
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 600,
      lineHeight: 1.4,
      color: '#0f172a',
    },
    h6: {
      fontSize: '1.125rem',
      fontWeight: 600,
      lineHeight: 1.5,
      color: '#0f172a',
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.6,
      color: '#475569',
      fontWeight: 400,
    },
    body2: {
      fontSize: '0.875rem',
      lineHeight: 1.6,
      color: '#64748b',
      fontWeight: 400,
    },
    button: {
      textTransform: 'none',
      fontWeight: 600,
      fontSize: '0.875rem',
    },
  },
  shape: {
    borderRadius: 12,
  },
  shadows: [
    'none',
    '0px 1px 2px rgba(0, 0, 0, 0.05)',
    '0px 1px 3px rgba(0, 0, 0, 0.1), 0px 1px 2px rgba(0, 0, 0, 0.06)',
    '0px 4px 6px rgba(0, 0, 0, 0.07), 0px 2px 4px rgba(0, 0, 0, 0.06)',
    '0px 10px 15px rgba(0, 0, 0, 0.1), 0px 4px 6px rgba(0, 0, 0, 0.05)',
    '0px 20px 25px rgba(0, 0, 0, 0.1), 0px 10px 10px rgba(0, 0, 0, 0.04)',
    '0px 25px 50px rgba(0, 0, 0, 0.15), 0px 12px 20px rgba(0, 0, 0, 0.08)',
    '0px 2px 4px rgba(37, 99, 235, 0.1), 0px 8px 16px rgba(37, 99, 235, 0.1)',
    '0px 4px 6px rgba(37, 99, 235, 0.1), 0px 12px 20px rgba(37, 99, 235, 0.15)',
    '0px 8px 12px rgba(37, 99, 235, 0.15), 0px 16px 24px rgba(37, 99, 235, 0.2)',
    '0px 12px 16px rgba(37, 99, 235, 0.2), 0px 20px 28px rgba(37, 99, 235, 0.25)',
    '0px 16px 20px rgba(37, 99, 235, 0.25), 0px 24px 32px rgba(37, 99, 235, 0.3)',
    '0px 20px 24px rgba(37, 99, 235, 0.3), 0px 28px 36px rgba(37, 99, 235, 0.35)',
    '0px 24px 28px rgba(37, 99, 235, 0.35), 0px 32px 40px rgba(37, 99, 235, 0.4)',
    '0px 28px 32px rgba(37, 99, 235, 0.4), 0px 36px 44px rgba(37, 99, 235, 0.45)',
    '0px 32px 36px rgba(37, 99, 235, 0.45), 0px 40px 48px rgba(37, 99, 235, 0.5)',
    '0px 36px 40px rgba(37, 99, 235, 0.5), 0px 44px 52px rgba(37, 99, 235, 0.55)',
    '0px 40px 44px rgba(37, 99, 235, 0.55), 0px 48px 56px rgba(37, 99, 235, 0.6)',
    '0px 44px 48px rgba(37, 99, 235, 0.6), 0px 52px 60px rgba(37, 99, 235, 0.65)',
    '0px 48px 52px rgba(37, 99, 235, 0.65), 0px 56px 64px rgba(37, 99, 235, 0.7)',
    '0px 52px 56px rgba(37, 99, 235, 0.7), 0px 60px 68px rgba(37, 99, 235, 0.75)',
    '0px 56px 60px rgba(37, 99, 235, 0.75), 0px 64px 72px rgba(37, 99, 235, 0.8)',
    '0px 60px 64px rgba(37, 99, 235, 0.8), 0px 68px 76px rgba(37, 99, 235, 0.85)',
    '0px 64px 68px rgba(37, 99, 235, 0.85), 0px 72px 80px rgba(37, 99, 235, 0.9)',
    '0px 68px 72px rgba(37, 99, 235, 0.9), 0px 76px 84px rgba(37, 99, 235, 0.95)',
  ],
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          backgroundColor: '#f8fafc',
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          background: 'linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)',
          color: '#0f172a',
          boxShadow: '0px 1px 3px rgba(0, 0, 0, 0.1), 0px 1px 2px rgba(0, 0, 0, 0.06)',
          borderBottom: '1px solid rgba(148, 163, 184, 0.2)',
          backdropFilter: 'blur(8px)',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundColor: '#ffffff',
          boxShadow: '0px 4px 6px rgba(0, 0, 0, 0.07), 0px 2px 4px rgba(0, 0, 0, 0.06)',
          borderRadius: 16,
          border: '1px solid #f1f5f9',
          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
          '&:hover': {
            boxShadow: '0px 20px 25px rgba(0, 0, 0, 0.1), 0px 10px 10px rgba(0, 0, 0, 0.04)',
            transform: 'translateY(-4px)',
            borderColor: '#e2e8f0',
          },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          textTransform: 'none',
          fontWeight: 600,
          padding: '10px 20px',
          fontSize: '0.875rem',
          transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
          '&:hover': {
            transform: 'translateY(-1px)',
            boxShadow: '0px 10px 15px rgba(0, 0, 0, 0.1), 0px 4px 6px rgba(0, 0, 0, 0.05)',
          },
        },
        contained: {
          background: 'linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%)',
          color: '#ffffff',
          boxShadow: '0px 4px 6px rgba(37, 99, 235, 0.3)',
          '&:hover': {
            background: 'linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%)',
            boxShadow: '0px 10px 15px rgba(37, 99, 235, 0.4)',
          },
        },
        outlined: {
          borderColor: '#2563eb',
          color: '#2563eb',
          borderWidth: 1.5,
          '&:hover': {
            borderColor: '#1d4ed8',
            backgroundColor: 'rgba(37, 99, 235, 0.04)',
            borderWidth: 1.5,
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundColor: '#ffffff',
          borderRadius: 12,
          border: '1px solid #f1f5f9',
        },
        elevation1: {
          boxShadow: '0px 1px 2px rgba(0, 0, 0, 0.05)',
        },
        elevation2: {
          boxShadow: '0px 1px 3px rgba(0, 0, 0, 0.1), 0px 1px 2px rgba(0, 0, 0, 0.06)',
        },
        elevation3: {
          boxShadow: '0px 4px 6px rgba(0, 0, 0, 0.07), 0px 2px 4px rgba(0, 0, 0, 0.06)',
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 12,
            backgroundColor: '#ffffff',
            transition: 'all 0.2s ease-in-out',
            '& fieldset': {
              borderColor: '#e2e8f0',
              borderWidth: 1.5,
            },
            '&:hover fieldset': {
              borderColor: '#2563eb',
            },
            '&.Mui-focused fieldset': {
              borderColor: '#2563eb',
              borderWidth: 2,
            },
          },
        },
      },
    },
    MuiSelect: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          backgroundColor: '#ffffff',
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          fontSize: '0.875rem',
          color: '#64748b',
          '&.Mui-selected': {
            color: '#2563eb',
          },
        },
      },
    },
    MuiTabs: {
      styleOverrides: {
        indicator: {
          backgroundColor: '#2563eb',
          height: 3,
          borderRadius: '3px 3px 0 0',
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          fontWeight: 600,
          fontSize: '0.75rem',
        },
        filled: {
          '&.MuiChip-colorPrimary': {
            background: 'linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%)',
            color: '#ffffff',
          },
          '&.MuiChip-colorSecondary': {
            backgroundColor: '#f1f5f9',
            color: '#475569',
          },
        },
      },
    },
    MuiTableHead: {
      styleOverrides: {
        root: {
          backgroundColor: '#f8fafc',
          '& .MuiTableCell-head': {
            fontWeight: 600,
            color: '#0f172a',
            borderBottom: '2px solid #e2e8f0',
            fontSize: '0.875rem',
          },
        },
      },
    },
    MuiTableCell: {
      styleOverrides: {
        root: {
          borderBottom: '1px solid #f1f5f9',
          padding: '16px',
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundColor: '#ffffff',
          borderRight: '1px solid #f1f5f9',
          boxShadow: '0px 4px 6px rgba(0, 0, 0, 0.07), 0px 2px 4px rgba(0, 0, 0, 0.06)',
        },
      },
    },
    MuiListItemButton: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          margin: '4px 12px',
          padding: '12px 16px',
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            backgroundColor: 'rgba(37, 99, 235, 0.08)',
            transform: 'translateX(4px)',
          },
          '&.Mui-selected': {
            backgroundColor: 'rgba(37, 99, 235, 0.12)',
            color: '#2563eb',
            '&:hover': {
              backgroundColor: 'rgba(37, 99, 235, 0.16)',
            },
          },
        },
      },
    },
    MuiIconButton: {
      styleOverrides: {
        root: {
          color: '#64748b',
          borderRadius: 10,
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            backgroundColor: 'rgba(37, 99, 235, 0.08)',
            color: '#2563eb',
            transform: 'scale(1.05)',
          },
        },
      },
    },
  },
});