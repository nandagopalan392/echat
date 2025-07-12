import React, { useState, useEffect } from 'react';
import { api } from '../services/api';

const FileListDashboard = ({ onClose }) => {
    const [files, setFiles] = useState([]);
    const [stats, setStats] = useState({
        totalFiles: 0,
        totalSize: 0,
        formatStats: []
    });
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        fetchFiles();
    }, []);

    const fetchFiles = async () => {
        try {
            const response = await api.get('/api/admin/files');
            
            // The API returns data directly, not wrapped in a data field
            if (response && typeof response === 'object') {
                setFiles(response.files || []);
                setStats(response.stats || {
                    totalFiles: 0,
                    totalSize: 0,
                    formatStats: []
                });
                setError(null);
            } else {
                throw new Error('Invalid response format');
            }
        } catch (error) {
            console.error('Error fetching files:', error);
            setError('Failed to load files');
        } finally {
            setLoading(false);
        }
    };

    const formatSize = (bytes) => {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    };

    const formatDate = (dateString) => {
        return new Date(dateString).toLocaleString();
    };

    if (loading) return <div className="text-center p-4">Loading...</div>;
    if (error) return <div className="text-red-500 p-4">{error}</div>;

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center">
            <div className="bg-white rounded-lg p-6 w-3/4 max-h-[80vh] overflow-y-auto">
                <div className="flex justify-between items-center mb-6">
                    <h2 className="text-2xl font-bold">Uploaded Files</h2>
                    <button onClick={onClose} className="text-gray-500 hover:text-gray-700">×</button>
                </div>

                {/* File Statistics Section */}
                <div className="grid grid-cols-3 gap-4 mb-6">
                    <div className="bg-indigo-50 p-4 rounded-lg">
                        <h3 className="text-lg font-semibold mb-2">Total Files</h3>
                        <p className="text-3xl font-bold text-indigo-600">{stats.totalFiles}</p>
                    </div>
                    <div className="bg-indigo-50 p-4 rounded-lg">
                        <h3 className="text-lg font-semibold mb-2">Total Size</h3>
                        <p className="text-3xl font-bold text-indigo-600">{formatSize(stats.totalSize)}</p>
                    </div>
                    <div className="bg-indigo-50 p-4 rounded-lg">
                        <h3 className="text-lg font-semibold mb-2">File Types</h3>
                        <div className="space-y-1">
                            {stats.formatStats?.map((stat) => (
                                <div key={stat.format} className="flex justify-between">
                                    <span className="text-gray-600">.{stat.format}</span>
                                    <span className="font-semibold">{stat.count}</span>
                                </div>
                            ))}
                            {!stats.formatStats?.length && (
                                <p className="text-gray-500">No files uploaded</p>
                            )}
                        </div>
                    </div>
                </div>

                <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                        <tr>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                File Name
                            </th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                Format
                            </th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                Size
                            </th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                Upload Date
                            </th>
                        </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                        {files.map((file, index) => (
                            <tr key={index} className="hover:bg-gray-50">
                                <td className="px-6 py-4 whitespace-nowrap">{file.filename}</td>
                                <td className="px-6 py-4 whitespace-nowrap">{file.format}</td>
                                <td className="px-6 py-4 whitespace-nowrap">{formatSize(file.size)}</td>
                                <td className="px-6 py-4 whitespace-nowrap">{formatDate(file.upload_date)}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default FileListDashboard;
