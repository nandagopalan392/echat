import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../services/api';
import { 
    ThemeProvider, 
    createTheme, 
    Box, 
    Drawer, 
    List, 
    ListItemButton, 
    ListItemIcon, 
    ListItemText, 
    Typography, 
    Divider, 
    IconButton 
} from '@mui/material';
import { 
    ArrowBack, 
    People as PeopleIcon, 
    Timeline as TimelineIcon, 
    Assessment as AssessmentIcon,
    PersonAdd as PersonAddIcon 
} from '@mui/icons-material';

const ManageUserPage = () => {
    const navigate = useNavigate();
    const [activeTab, setActiveTab] = useState('users');
    const [loading, setLoading] = useState(true);
    const [users, setUsers] = useState([]);
    const [activities, setActivities] = useState([]);
    const [stats, setStats] = useState({
        totalUsers: 0,
        activeUsers: 0,
        totalSessions: 0,
        totalMessages: 0
    });
    const [showAddUserModal, setShowAddUserModal] = useState(false);
    const [newUser, setNewUser] = useState({
        username: '',
        email: '',
        password: '',
        role: 'Engineer'
    });

    useEffect(() => {
        loadUserData();
    }, []);

    const loadUserData = async () => {
        try {
            setLoading(true);
            await Promise.all([
                loadUsers(),
                loadActivities(),
                loadStats()
            ]);
        } catch (error) {
            console.error('Error loading user data:', error);
        } finally {
            setLoading(false);
        }
    };

    const loadUsers = async () => {
        try {
            const response = await api.getUsers();
            console.log('Users API response:', response);
            setUsers(response.users || []);
        } catch (error) {
            console.error('Error loading users:', error);
        }
    };

    const loadActivities = async () => {
        try {
            const response = await api.getUserActivities();
            console.log('Activities API response:', response);
            setActivities(response.activities || []);
        } catch (error) {
            console.error('Error loading activities:', error);
        }
    };

    const loadStats = async () => {
        try {
            const response = await api.getUserStatsGeneral();
            console.log('Stats API response:', response);
            setStats(response.stats || {
                totalUsers: 0,
                activeUsers: 0,
                totalSessions: 0,
                totalMessages: 0
            });
        } catch (error) {
            console.error('Error loading stats:', error);
        }
    };

    const handleDeleteUser = async (userId) => {
        if (window.confirm('Are you sure you want to delete this user?')) {
            try {
                await api.deleteUser(userId);
                await loadUsers();
            } catch (error) {
                console.error('Error deleting user:', error);
            }
        }
    };

    const handleUpdateUserRole = async (userId, newRole) => {
        try {
            await api.updateUserRole(userId, newRole);
            await loadUsers();
        } catch (error) {
            console.error('Error updating user role:', error);
        }
    };

    const handleAddUser = async () => {
        try {
            if (!newUser.username || !newUser.email || !newUser.password) {
                alert('Please fill in all required fields');
                return;
            }

            await api.createUser(newUser);
            setShowAddUserModal(false);
            setNewUser({ username: '', email: '', password: '', role: 'Engineer' });
            await loadUsers();
            alert('User created successfully!');
        } catch (error) {
            console.error('Error adding user:', error);
            alert('Error creating user. Please try again.');
        }
    };

    const formatDate = (dateString) => {
        if (!dateString) return 'N/A';
        
        try {
            const date = new Date(dateString);
            // Check if date is valid
            if (isNaN(date.getTime())) {
                return 'Invalid Date';
            }
            
            return date.toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });
        } catch (error) {
            console.error('Date formatting error:', error);
            return 'Invalid Date';
        }
    };

    const theme = createTheme({
        palette: {
            primary: {
                main: '#2563eb',
            },
        },
    });

    const tabs = [
        { id: 'users', name: 'All Users', icon: PeopleIcon },
        { id: 'activities', name: 'Activities', icon: TimelineIcon },
        { id: 'stats', name: 'Statistics', icon: AssessmentIcon }
    ];

    if (loading) {
        return (
            <div className="min-h-screen bg-gray-50 flex items-center justify-center">
                <div className="text-center">
                    <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
                    <p className="mt-2 text-gray-500">Loading user data...</p>
                </div>
            </div>
        );
    }

    return (
        <ThemeProvider theme={theme}>
            <Box sx={{ minHeight: '100vh', bgcolor: '#f8fafc', display: 'flex' }}>
                {/* Sidebar */}
                <Drawer
                    variant="permanent"
                    sx={{
                        width: 280,
                        flexShrink: 0,
                        '& .MuiDrawer-paper': {
                            width: 280,
                            boxSizing: 'border-box',
                            bgcolor: '#ffffff',
                            borderRight: '1px solid #f1f5f9',
                            boxShadow: '0px 4px 6px rgba(0, 0, 0, 0.07), 0px 2px 4px rgba(0, 0, 0, 0.06)',
                            background: 'linear-gradient(180deg, #ffffff 0%, #f8fafc 100%)',
                        },
                    }}
                >
                    <Box sx={{ p: 3 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                            <IconButton
                                edge="start"
                                onClick={() => navigate('/chat')}
                                sx={{ 
                                    mr: 2,
                                    color: '#64748b',
                                    borderRadius: '10px',
                                    transition: 'all 0.2s ease-in-out',
                                    '&:hover': {
                                        backgroundColor: 'rgba(37, 99, 235, 0.08)',
                                        color: '#2563eb',
                                        transform: 'scale(1.05)',
                                    },
                                }}
                            >
                                <ArrowBack />
                            </IconButton>
                            <PeopleIcon sx={{ 
                                mr: 1, 
                                color: '#2563eb',
                                fontSize: '1.5rem',
                            }} />
                            <Typography variant="h6" sx={{ 
                                fontWeight: 700,
                                color: '#0f172a',
                                fontSize: '1.125rem',
                            }}>
                                Manage Users
                            </Typography>
                        </Box>
                        <Divider sx={{ 
                            mb: 3,
                            borderColor: 'rgba(148, 163, 184, 0.2)',
                        }} />
                        
                        <List sx={{ p: 0 }}>
                            {tabs.map((tab) => {
                                const IconComponent = tab.icon;
                                return (
                                    <ListItemButton
                                        key={tab.id}
                                        selected={activeTab === tab.id}
                                        onClick={() => setActiveTab(tab.id)}
                                        sx={{ 
                                            borderRadius: '12px', 
                                            mb: 1,
                                            margin: '4px 0',
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
                                                '& .MuiListItemIcon-root': {
                                                    color: '#2563eb',
                                                },
                                                '& .MuiListItemText-primary': {
                                                    color: '#2563eb',
                                                    fontWeight: 600,
                                                },
                                            },
                                        }}
                                    >
                                        <ListItemIcon sx={{ 
                                            color: activeTab === tab.id ? '#2563eb' : '#64748b',
                                            minWidth: '40px',
                                        }}>
                                            <IconComponent />
                                        </ListItemIcon>
                                        <ListItemText 
                                            primary={tab.name} 
                                            primaryTypographyProps={{
                                                fontSize: '0.875rem',
                                                fontWeight: activeTab === tab.id ? 600 : 500,
                                                color: activeTab === tab.id ? '#2563eb' : '#475569',
                                            }}
                                        />
                                    </ListItemButton>
                                );
                            })}
                        </List>
                        
                        {activeTab === 'users' && (
                            <Box sx={{ mt: 2, pl: 2, borderLeft: '2px solid #f1f5f9' }}>
                                <Box
                                    onClick={() => setShowAddUserModal(true)}
                                    sx={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        padding: '12px 16px',
                                        color: '#475569',
                                        cursor: 'pointer',
                                        borderRadius: '12px',
                                        transition: 'all 0.2s ease-in-out',
                                        fontSize: '0.875rem',
                                        fontWeight: 500,
                                        '&:hover': {
                                            backgroundColor: 'rgba(37, 99, 235, 0.08)',
                                            transform: 'translateX(4px)',
                                        },
                                    }}
                                >
                                    <PersonAddIcon sx={{ width: 16, height: 16, mr: 1.5 }} />
                                    Add New User
                                </Box>
                            </Box>
                        )}
                        
                        <Box sx={{ 
                            mt: 3, 
                            p: 2.5, 
                            bgcolor: 'rgba(148, 163, 184, 0.05)', 
                            borderRadius: '12px',
                            border: '1px solid rgba(148, 163, 184, 0.1)',
                        }}>
                            <Typography variant="subtitle2" sx={{ 
                                fontWeight: 600,
                                color: '#0f172a',
                                mb: 1.5,
                                fontSize: '0.875rem',
                            }}>
                                Quick Stats
                            </Typography>
                            <Box sx={{ fontSize: '0.875rem', color: '#64748b' }}>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                                    <span>Total Users:</span>
                                    <Typography sx={{ fontWeight: 600, color: '#2563eb', fontSize: '0.875rem' }}>
                                        {stats.totalUsers}
                                    </Typography>
                                </Box>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                                    <span>Active Users:</span>
                                    <Typography sx={{ fontWeight: 600, color: '#2563eb', fontSize: '0.875rem' }}>
                                        {stats.activeUsers}
                                    </Typography>
                                </Box>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                                    <span>Total Sessions:</span>
                                    <Typography sx={{ fontWeight: 600, color: '#2563eb', fontSize: '0.875rem' }}>
                                        {stats.totalSessions}
                                    </Typography>
                                </Box>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                    <span>Messages:</span>
                                    <Typography sx={{ fontWeight: 600, color: '#2563eb', fontSize: '0.875rem' }}>
                                        {stats.totalMessages}
                                    </Typography>
                                </Box>
                            </Box>
                        </Box>
                    </Box>
                </Drawer>

                {/* Main Content */}
                <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
                    {/* Header */}
                    <Box sx={{ bgcolor: 'white', borderBottom: '1px solid #f1f5f9', boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)' }}>
                        <Box sx={{ px: 3, py: 2 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <Box>
                                    <Typography variant="h4" sx={{ 
                                        fontWeight: 700,
                                        color: '#0f172a',
                                        fontSize: '1.5rem',
                                        mb: 0.5,
                                    }}>
                                        {tabs.find(t => t.id === activeTab)?.name}
                                    </Typography>
                                    <Typography sx={{ 
                                        color: '#64748b',
                                        fontSize: '0.875rem',
                                    }}>
                                        {activeTab === 'users' && 'View and manage all system users'}
                                        {activeTab === 'activities' && 'Monitor user activities and system usage'}
                                        {activeTab === 'stats' && 'View detailed system statistics and analytics'}
                                    </Typography>
                                </Box>
                            </Box>
                        </Box>
                    </Box>

                    {/* Content */}
                    <Box sx={{ flexGrow: 1, p: 3 }}>
                        <div className="flex-1 p-6">
                {/* All Users Tab */}
                {activeTab === 'users' && (
                    <div className="bg-white rounded-lg shadow">
                        <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center">
                            <h2 className="text-lg font-medium text-gray-900">All Users</h2>
                            <button
                                onClick={() => setShowAddUserModal(true)}
                                className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
                            >
                                Add User
                            </button>
                        </div>
                        <div className="overflow-x-auto">
                            <table className="min-w-full divide-y divide-gray-200">
                                <thead className="bg-gray-50">
                                    <tr>
                                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                            User
                                        </th>
                                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                            Email
                                        </th>
                                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                            Role
                                        </th>
                                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                            Created
                                        </th>
                                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                            Actions
                                        </th>
                                    </tr>
                                </thead>
                                <tbody className="bg-white divide-y divide-gray-200">
                                    {users.map((user) => (
                                        <tr key={user.id} className="hover:bg-gray-50">
                                            <td className="px-6 py-4 whitespace-nowrap">
                                                <div className="flex items-center">
                                                    <div className="h-10 w-10 rounded-full bg-indigo-100 flex items-center justify-center">
                                                        <span className="text-indigo-600 font-medium">
                                                            {user.username.charAt(0).toUpperCase()}
                                                        </span>
                                                    </div>
                                                    <div className="ml-4">
                                                        <div className="text-sm font-medium text-gray-900">
                                                            {user.username}
                                                        </div>
                                                    </div>
                                                </div>
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                                {user.email}
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap">
                                                <select
                                                    value={user.role}
                                                    onChange={(e) => handleUpdateUserRole(user.id, e.target.value)}
                                                    className="text-sm border border-gray-300 rounded px-2 py-1"
                                                >
                                                    <option value="Engineer">Engineer</option>
                                                    <option value="Manager">Manager</option>
                                                    <option value="Business Development">Business Development</option>
                                                    <option value="Associate">Associate</option>
                                                    <option value="Admin">Admin</option>
                                                </select>
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                                {formatDate(user.created_at)}
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                                                <button
                                                    onClick={() => handleDeleteUser(user.id)}
                                                    className="text-red-600 hover:text-red-900 transition-colors"
                                                >
                                                    Delete
                                                </button>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}

                {/* Activities Tab */}
                {activeTab === 'activities' && (
                    <div className="bg-white rounded-lg shadow">
                        <div className="px-6 py-4 border-b border-gray-200">
                            <h2 className="text-lg font-medium text-gray-900">Recent Activities</h2>
                        </div>
                        <div className="p-6">
                            <div className="space-y-4">
                                {activities.map((activity, index) => (
                                    <div key={index} className="flex items-start space-x-4 p-4 bg-gray-50 rounded-lg">
                                        <div className="flex-shrink-0">
                                            <div className="h-8 w-8 rounded-full bg-indigo-100 flex items-center justify-center">
                                                <span className="text-indigo-600 text-sm font-medium">
                                                    {activity.user_name?.charAt(0).toUpperCase()}
                                                </span>
                                            </div>
                                        </div>
                                        <div className="flex-1">
                                            <p className="text-sm text-gray-900">
                                                <span className="font-medium">{activity.user_name}</span> {activity.action}
                                            </p>
                                            <p className="text-xs text-gray-500">{formatDate(activity.timestamp)}</p>
                                        </div>
                                    </div>
                                ))}
                                {activities.length === 0 && (
                                    <p className="text-center text-gray-500 py-8">No recent activities</p>
                                )}
                            </div>
                        </div>
                    </div>
                )}

                {/* Statistics Tab */}
                {activeTab === 'stats' && (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                        <div className="bg-white rounded-lg shadow p-6">
                            <div className="flex items-center">
                                <div className="flex-shrink-0">
                                    <div className="h-12 w-12 rounded-lg bg-blue-100 flex items-center justify-center">
                                        <span className="text-2xl">👥</span>
                                    </div>
                                </div>
                                <div className="ml-4">
                                    <h3 className="text-lg font-medium text-gray-900">Total Users</h3>
                                    <p className="text-3xl font-bold text-blue-600">{stats.totalUsers}</p>
                                </div>
                            </div>
                        </div>

                        <div className="bg-white rounded-lg shadow p-6">
                            <div className="flex items-center">
                                <div className="flex-shrink-0">
                                    <div className="h-12 w-12 rounded-lg bg-green-100 flex items-center justify-center">
                                        <span className="text-2xl">🟢</span>
                                    </div>
                                </div>
                                <div className="ml-4">
                                    <h3 className="text-lg font-medium text-gray-900">Active Users</h3>
                                    <p className="text-3xl font-bold text-green-600">{stats.activeUsers}</p>
                                </div>
                            </div>
                        </div>

                        <div className="bg-white rounded-lg shadow p-6">
                            <div className="flex items-center">
                                <div className="flex-shrink-0">
                                    <div className="h-12 w-12 rounded-lg bg-purple-100 flex items-center justify-center">
                                        <span className="text-2xl">💬</span>
                                    </div>
                                </div>
                                <div className="ml-4">
                                    <h3 className="text-lg font-medium text-gray-900">Total Sessions</h3>
                                    <p className="text-3xl font-bold text-purple-600">{stats.totalSessions}</p>
                                </div>
                            </div>
                        </div>

                        <div className="bg-white rounded-lg shadow p-6">
                            <div className="flex items-center">
                                <div className="flex-shrink-0">
                                    <div className="h-12 w-12 rounded-lg bg-orange-100 flex items-center justify-center">
                                        <span className="text-2xl">📝</span>
                                    </div>
                                </div>
                                <div className="ml-4">
                                    <h3 className="text-lg font-medium text-gray-900">Total Messages</h3>
                                    <p className="text-3xl font-bold text-orange-600">{stats.totalMessages}</p>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
                        </div>
                    </Box>
                </Box>

            {/* Add User Modal */}
            {showAddUserModal && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                    <div className="bg-white rounded-lg p-6 w-full max-w-md">
                        <h3 className="text-lg font-medium text-gray-900 mb-4">Add New User</h3>
                        
                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    Username *
                                </label>
                                <input
                                    type="text"
                                    value={newUser.username}
                                    onChange={(e) => setNewUser({ ...newUser, username: e.target.value })}
                                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
                                    placeholder="Enter username"
                                />
                            </div>
                            
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    Email *
                                </label>
                                <input
                                    type="email"
                                    value={newUser.email}
                                    onChange={(e) => setNewUser({ ...newUser, email: e.target.value })}
                                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
                                    placeholder="Enter email"
                                />
                            </div>
                            
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    Password *
                                </label>
                                <input
                                    type="password"
                                    value={newUser.password}
                                    onChange={(e) => setNewUser({ ...newUser, password: e.target.value })}
                                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
                                    placeholder="Enter password"
                                />
                            </div>
                            
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    Role
                                </label>
                                <select
                                    value={newUser.role}
                                    onChange={(e) => setNewUser({ ...newUser, role: e.target.value })}
                                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
                                >
                                    <option value="Engineer">Engineer</option>
                                    <option value="Manager">Manager</option>
                                    <option value="Business Development">Business Development</option>
                                    <option value="Associate">Associate</option>
                                    <option value="Admin">Admin</option>
                                </select>
                            </div>
                        </div>
                        
                        <div className="flex justify-end space-x-3 mt-6">
                            <button
                                onClick={() => {
                                    setShowAddUserModal(false);
                                    setNewUser({ username: '', email: '', password: '', role: 'Engineer' });
                                }}
                                className="px-4 py-2 text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={handleAddUser}
                                className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
                            >
                                Add User
                            </button>
                        </div>
                    </div>
                </div>
            )}
            </Box>
        </ThemeProvider>
    );
};

export default ManageUserPage;
