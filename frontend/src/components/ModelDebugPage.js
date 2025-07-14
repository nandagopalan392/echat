import React, { useState, useEffect } from 'react';
import { api } from '../services/api';

const ModelDebugPage = () => {
    const [models, setModels] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        loadModels();
    }, []);

    const loadModels = async () => {
        try {
            setLoading(true);
            const response = await api.get('/api/models/available');
            console.log('Full API response:', response);
            
            if (response && response.models) {
                setModels(response.models);
            } else {
                setError('No models in response');
            }
        } catch (err) {
            console.error('Error loading models:', err);
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const groupedByCategory = models.reduce((acc, model) => {
        const category = model.category || 'unknown';
        if (!acc[category]) acc[category] = [];
        acc[category].push(model);
        return acc;
    }, {});

    if (loading) return <div>Loading models...</div>;
    if (error) return <div className="text-red-500">Error: {error}</div>;

    return (
        <div className="p-6">
            <h1 className="text-2xl font-bold mb-4">Model Debug Information</h1>
            
            <div className="mb-4">
                <h2 className="text-lg font-semibold">Total Models: {models.length}</h2>
            </div>

            {Object.entries(groupedByCategory).map(([category, categoryModels]) => (
                <div key={category} className="mb-6">
                    <h3 className="text-xl font-bold mb-2 capitalize">
                        {category} Models ({categoryModels.length})
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
                        {categoryModels.map((model, index) => (
                            <div key={index} className="p-3 border rounded bg-gray-50">
                                <div className="font-medium">{model.name}</div>
                                <div className="text-sm text-gray-600">
                                    Category: {model.category || 'Unknown'}
                                </div>
                                <div className="text-sm text-gray-600">
                                    Size: {model.size || 'Unknown'}
                                </div>
                                <div className="text-sm text-gray-600">
                                    Source: {model.source || 'Unknown'}
                                </div>
                                {/* Extract parameter info from name */}
                                {(() => {
                                    const paramMatch = model.name.match(/(\d+\.?\d*)[bB]/i) || 
                                                     model.name.match(/:(\d+\.?\d*)([bB])?/i);
                                    return paramMatch && (
                                        <div className="text-sm text-blue-600 font-medium">
                                            Parameters: {paramMatch[1]}B
                                        </div>
                                    );
                                })()}
                                {model.description && (
                                    <div className="text-xs text-gray-500 mt-1">
                                        {model.description}
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            ))}

            <div className="mt-8">
                <h3 className="text-lg font-bold mb-2">Raw JSON Data</h3>
                <pre className="bg-gray-100 p-4 rounded overflow-auto text-xs">
                    {JSON.stringify(models, null, 2)}
                </pre>
            </div>
        </div>
    );
};

export default ModelDebugPage;
