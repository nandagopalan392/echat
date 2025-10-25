/**
 * Enhanced Authentication Service
 * 
 * Features:
 * - Secure httpOnly cookie-based authentication
 * - Automatic token refresh
 * - Token rotation
 * - Secure logout with cookie clearing
 */

const API_BASE_URL = '';

class AuthService {
    constructor() {
        this.refreshTimer = null;
        this.tokenCheckInterval = null;
    }

    /**
     * Get token expiration time from cookie
     * @returns {number|null} Expiration timestamp in milliseconds
     */
    getTokenExpiration() {
        const cookies = document.cookie.split(';');
        for (let cookie of cookies) {
            const [name, value] = cookie.trim().split('=');
            if (name === 'token_expires_at') {
                return parseInt(value);
            }
        }
        return null;
    }

    /**
     * Check if token is about to expire (within 5 minutes)
     * @returns {boolean}
     */
    shouldRefreshToken() {
        const expiresAt = this.getTokenExpiration();
        if (!expiresAt) return false;

        const now = Date.now();
        const fiveMinutes = 5 * 60 * 1000;
        
        return (expiresAt - now) < fiveMinutes;
    }

    /**
     * Login user with credentials
     * Tokens are stored in httpOnly cookies automatically
     * 
     * @param {string} username 
     * @param {string} password 
     * @returns {Promise<Object>} User data
     */
    async login(username, password) {
        try {
            const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                credentials: 'include', // Important: include cookies
                body: JSON.stringify({ username, password }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Login failed');
            }

            const data = await response.json();
            
            // Store user info in localStorage (not sensitive)
            localStorage.setItem('username', data.username);
            localStorage.setItem('is_admin', data.is_admin);
            localStorage.setItem('role', data.role);

            // ✅ REMOVED: No longer storing tokens in localStorage
            // Tokens are now only in httpOnly cookies (secure)

            // Start automatic token refresh
            this.startTokenRefresh();

            console.log('✅ Login successful, tokens stored securely in httpOnly cookies');
            
            return data;
        } catch (error) {
            console.error('❌ Login error:', error);
            throw error;
        }
    }

    /**
     * Refresh access token using refresh token from cookie
     * @returns {Promise<Object>} New token data
     */
    async refreshToken() {
        try {
            const response = await fetch(`${API_BASE_URL}/api/auth/refresh`, {
                method: 'POST',
                credentials: 'include', // Send cookies with request
            });

            if (!response.ok) {
                // Refresh token invalid/expired - redirect to login
                console.error('❌ Token refresh failed, redirecting to login');
                this.logout();
                window.location.href = '/';
                throw new Error('Token refresh failed');
            }

            const data = await response.json();
            
            // ✅ REMOVED: No longer updating localStorage token
            // Token is automatically updated in httpOnly cookie by backend

            console.log('✅ Token refreshed successfully');
            
            return data;
        } catch (error) {
            console.error('❌ Token refresh error:', error);
            throw error;
        }
    }

    /**
     * Start automatic token refresh timer
     * Checks every minute if token needs refreshing
     */
    startTokenRefresh() {
        // Clear any existing timers
        this.stopTokenRefresh();

        // Check token expiration every minute
        this.tokenCheckInterval = setInterval(() => {
            if (this.shouldRefreshToken()) {
                console.log('🔄 Token expiring soon, refreshing...');
                this.refreshToken().catch(err => {
                    console.error('Failed to refresh token:', err);
                });
            }
        }, 60 * 1000); // Check every minute

        console.log('⏰ Token refresh timer started');
    }

    /**
     * Stop automatic token refresh timer
     */
    stopTokenRefresh() {
        if (this.tokenCheckInterval) {
            clearInterval(this.tokenCheckInterval);
            this.tokenCheckInterval = null;
            console.log('⏰ Token refresh timer stopped');
        }
    }

    /**
     * Logout user - clears cookies and localStorage
     * @returns {Promise<void>}
     */
    async logout() {
        try {
            // Stop token refresh
            this.stopTokenRefresh();

            // Call backend logout endpoint to clear cookies
            await fetch(`${API_BASE_URL}/api/auth/logout`, {
                method: 'POST',
                credentials: 'include',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            // Clear localStorage (user info only, no tokens)
            localStorage.removeItem('username');
            localStorage.removeItem('is_admin');
            localStorage.removeItem('role');

            console.log('✅ Logged out successfully');
        } catch (error) {
            console.error('❌ Logout error:', error);
            // Still clear local data even if API call fails
            localStorage.removeItem('username');
            localStorage.removeItem('is_admin');
            localStorage.removeItem('role');
        }
    }

    /**
     * Check if user is authenticated
     * @returns {boolean}
     */
    isAuthenticated() {
        // Check if we have a token expiration cookie
        const expiresAt = this.getTokenExpiration();
        if (!expiresAt) return false;

        // Check if token hasn't expired
        return Date.now() < expiresAt;
    }

    /**
     * Get user info from localStorage
     * @returns {Object|null}
     */
    getUserInfo() {
        const username = localStorage.getItem('username');
        if (!username) return null;

        return {
            username,
            is_admin: localStorage.getItem('is_admin') === 'true',
            role: localStorage.getItem('role') || 'Engineer'
        };
    }

    /**
     * Initialize auth service on app start
     * Starts token refresh if user is logged in
     */
    initialize() {
        if (this.isAuthenticated()) {
            this.startTokenRefresh();
            console.log('✅ Auth service initialized, user is authenticated');
        } else {
            console.log('ℹ️ Auth service initialized, user not authenticated');
        }
    }
}

// Export singleton instance
export const authService = new AuthService();
export default authService;
