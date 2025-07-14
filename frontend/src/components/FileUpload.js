import React, { useState } from 'react';
import { api } from '../services/api';

const FileUpload = ({ token }) => {
    const [progress, setProgress] = useState(0);
    const [uploading, setUploading] = useState(false);

    const handleFileUpload = async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        setUploading(true);
        setProgress(0);

        try {
            await api.uploadFileWithProgress(file, false, "", (progress) => {
                setProgress(progress);
            });
            
            if (progress === 100) {
                alert('File uploaded and processed successfully!');
            }
        } catch (error) {
            console.error('Upload failed:', error);
            alert('Upload failed: ' + error.message);
        } finally {
            setUploading(false);
        }
    };

    return (
        <div>
            <input 
                type="file" 
                onChange={handleFileUpload}
                disabled={uploading}
            />
            {uploading && (
                <div className="progress-bar">
                    <div 
                        className="progress" 
                        style={{ width: `${progress}%` }}
                    >
                        {progress}%
                    </div>
                </div>
            )}
            <style jsx>{`
                .progress-bar {
                    width: 100%;
                    height: 20px;
                    background-color: #f0f0f0;
                    border-radius: 10px;
                    overflow: hidden;
                    margin-top: 10px;
                }
                .progress {
                    height: 100%;
                    background-color: #4CAF50;
                    transition: width 0.3s ease-in-out;
                    text-align: center;
                    color: white;
                }
            `}</style>
        </div>
    );
};

export default FileUpload;
