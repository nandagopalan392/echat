import React, { useState, useEffect } from 'react';
import { api } from '../services/api';

const ActivityDashboard = ({ onClose }) => {
    const [stats, setStats] = useState({
        totalUsers: 0,
        activeUsers: 0,
        totalMessages: 0,
        recentActivities: []
    });
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        fetchActivityStats();
    }, []);

    const fetchActivityStats = async () => {
        try {
            setLoading(true);
            const response = await api.get('/api/admin/activity-stats');
            console.log('Activity stats response:', response); // Debug log
            
            if (response && response.data) {
                setStats(response.data);
                setError(null);
            } else {
                throw new Error('Invalid response format');
            }
        } catch (error) {
            console.error('Error fetching activity stats:', error);
            setError('Failed to load activity statistics');
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center">
                <div className="bg-white rounded-lg p-6">
                    <p>Loading activity statistics...</p>
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
                    <h2 className="text-2xl font-bold">Activity Dashboard</h2>
                    <button onClick={onClose} className="text-gray-500 hover:text-gray-700">×</button>
                </div>

                <div className="grid grid-cols-3 gap-4 mb-6">
                    <div className="bg-indigo-50 p-4 rounded-lg">
                        <h3 className="text-lg font-semibold mb-2">Total Users</h3>
                        <p className="text-3xl font-bold text-indigo-600">{stats?.totalUsers ?? 0}</p>
                    </div>
                    <div className="bg-indigo-50 p-4 rounded-lg">
                        <h3 className="text-lg font-semibold mb-2">Active Users Today</h3>
                        <p className="text-3xl font-bold text-indigo-600">{stats?.activeUsers ?? 0}</p>
                    </div>
                    <div className="bg-indigo-50 p-4 rounded-lg">
                        <h3 className="text-lg font-semibold mb-2">Total Messages</h3>
                        <p className="text-3xl font-bold text-indigo-600">{stats?.totalMessages ?? 0}</p>
                    </div>
                </div>

                <div className="mt-6">
                    <h3 className="text-xl font-semibold mb-4">Recent Activities</h3>
                    <div className="space-y-4">
                        {stats?.recentActivities?.map((activity, index) => (
                            <div key={index} className="bg-gray-50 p-4 rounded-lg">
                                <div className="flex justify-between items-center">
                                    <div>
                                        <span className="font-medium">{activity.username}</span>
                                        <span className="text-gray-600 ml-2">{activity.action}</span>
                                        <span className="text-gray-600 ml-2">{activity.content}</span>
                                    </div>
                                    <span className="text-sm text-gray-500">
                                        {new Date(activity.timestamp).toLocaleString()}
                                    </span>
                                </div>
                            </div>
                        ))}
                        {!stats?.recentActivities?.length && (
                            <p className="text-gray-500">No recent activities found</p>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ActivityDashboard;
