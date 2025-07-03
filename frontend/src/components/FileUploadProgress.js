import React from 'react';

const FileUploadProgress = ({ fileName, progress, currentFile, totalFiles }) => {
  return (
    <div className="fixed inset-0 flex items-center justify-center z-50 bg-black bg-opacity-50">
      <div className="bg-white p-6 rounded-lg shadow-xl w-96 max-w-[90%]">
        <div className="flex justify-between mb-4">
          <span className="text-sm font-medium text-gray-700 truncate flex-1" title={fileName}>
            {fileName}
            {totalFiles > 1 && ` (${currentFile}/${totalFiles})`}
          </span>
          <span className="text-sm font-semibold ml-2 text-indigo-600">
            {Math.round(progress)}%
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2.5 mb-2">
          <div
            className="bg-indigo-600 h-2.5 rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
        <div className="text-xs text-gray-500 text-center">
          {progress < 90 ? 'Uploading...' : 'Processing...'}
          {totalFiles > 1 && ` (File ${currentFile} of ${totalFiles})`}
        </div>
      </div>
    </div>
  );
};

export default FileUploadProgress;
