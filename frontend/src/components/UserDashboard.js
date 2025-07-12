import React, { useState, useEffect } from 'react';
import { api } from '../services/api';

const UserDashboard = ({ username, onClose }) => {
    const [userStats, setUserStats] = useState({
        totalMessages: 0,
        totalSessions: 0,
        lastActive: null,
        recentChats: []
    });
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        fetchUserStats();
    }, [username]);

    const fetchUserStats = async () => {
        try {
            setLoading(true);
            const response = await api.get(`/api/admin/user-stats/${username}`);
            
            if (response && response.data) {
                setUserStats(response.data);
                setError(null);
            } else {
                throw new Error('Invalid response format');
            }
        } catch (error) {
            console.error('Error fetching user stats:', error);
            setError('Failed to load user statistics');
        } finally {
            setLoading(false);
        }
    };

    const formatDate = (dateString) => {
        if (!dateString) return 'Never';
        try {
            return new Date(dateString).toLocaleString();
        } catch (error) {
            console.error('Date formatting error:', error);
            return 'Invalid Date';
        }
    };

    if (loading) {
        return (
            <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center">
                <div className="bg-white rounded-lg p-6">
                    <p>Loading user statistics...</p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center">
                <div className="bg-white rounded-lg p-6">
                    <h2 className="text-xl font-bold mb-4">Error</h2>
                    <p className="text-red-600">{error}</p>
                    <button 
                        onClick={onClose}
                        className="mt-4 px-4 py-2 bg-indigo-600 text-white rounded-md"
                    >
                        Close
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center">
            <div className="bg-white rounded-lg p-6 w-3/4 max-h-[80vh] overflow-y-auto">
                <div className="flex justify-between items-center mb-6">
                    <h2 className="text-2xl font-bold">User Dashboard: {username}</h2>
                    <button onClick={onClose} className="text-gray-500 hover:text-gray-700">×</button>
                </div>

                <div className="grid grid-cols-3 gap-4 mb-6">
                    <div className="bg-indigo-50 p-4 rounded-lg">
                        <h3 className="text-lg font-semibold mb-2">Total Messages</h3>
                        <p className="text-3xl font-bold text-indigo-600">{userStats?.totalMessages ?? 0}</p>
                    </div>
                    <div className="bg-indigo-50 p-4 rounded-lg">
                        <h3 className="text-lg font-semibold mb-2">Total Sessions</h3>
                        <p className="text-3xl font-bold text-indigo-600">{userStats?.totalSessions ?? 0}</p>
                    </div>
                    <div className="bg-indigo-50 p-4 rounded-lg">
                        <h3 className="text-lg font-semibold mb-2">Last Active</h3>
                        <p className="text-xl text-indigo-600">{formatDate(userStats?.lastActive)}</p>
                    </div>
                </div>

                <div className="mt-6">
                    <h3 className="text-xl font-semibold mb-4">Recent Chats</h3>
                    <div className="space-y-4">
                        {userStats?.recentChats?.map((chat, index) => (
                            <div key={index} className="bg-gray-50 p-4 rounded-lg">
                                <div className="flex justify-between items-center mb-2">
                                    <h4 className="font-medium">{chat?.topic || 'No Topic'}</h4>
                                    <span className="text-sm text-gray-500">{formatDate(chat?.date)}</span>
                                </div>
                                <p className="text-gray-600">{chat?.lastMessage || 'No messages'}</p>
                            </div>
                        ))}
                        {!userStats?.recentChats?.length && (
                            <p className="text-gray-500">No recent chats found</p>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default UserDashboard;
