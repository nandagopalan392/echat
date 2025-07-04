import React from 'react';

const ResponseOptions = ({ options, onSelect }) => {
  // Helper function to safely convert any value to string
  const safeString = (value) => {
    if (value === null || value === undefined) return '';
    if (typeof value === 'string') return value;
    if (typeof value === 'object') {
      return JSON.stringify(value);
    }
    return String(value);
  };

  // Helper function to render option content
  const renderOptionContent = (option) => {
    // Debug logging
    console.log("ResponseOptions - rendering option:", option);
    console.log("Option type:", typeof option);
    if (typeof option === 'object') {
      console.log("Option thinking length:", option.thinking ? option.thinking.length : 0);
      console.log("Option content length:", option.content ? option.content.length : 0);
      console.log("Option thinking preview:", option.thinking ? option.thinking.substring(0, 100) + "..." : "none");
      console.log("Option content preview:", option.content ? option.content.substring(0, 100) + "..." : "none");
    }
    
    // Handle structured options (new format)
    if (typeof option === 'object' && option !== null) {
      const hasThinking = option.thinking && safeString(option.thinking).trim() !== '';
      const hasContent = option.content && safeString(option.content).trim() !== '';
      const hasStyle = option.style && safeString(option.style).trim() !== '';
      
      return (
        <div className="space-y-3">
          {hasThinking && (
            <div className="text-sm text-gray-600 bg-gray-50 p-3 rounded-lg border-l-4 border-blue-300">
              <div className="font-semibold text-blue-700 mb-1">Thinking Process:</div>
              <div className="whitespace-pre-wrap">{safeString(option.thinking)}</div>
            </div>
          )}
          {hasContent && (
            <div className="text-gray-800">
              <div className="whitespace-pre-wrap break-words">{safeString(option.content)}</div>
            </div>
          )}
          {hasStyle && (
            <div className="text-xs text-gray-500 mt-2">
              Style: <span className="capitalize font-medium">{safeString(option.style)}</span>
            </div>
          )}
        </div>
      );
    }
    
    // Handle legacy string options
    return (
      <div className="whitespace-pre-wrap break-words">
        {safeString(option)}
      </div>
    );
  };

  return (
    <div className="w-full space-y-4 my-4">
      <div className="text-center text-sm text-gray-600 bg-gray-100 p-2 rounded-lg">
        {options.length === 1 
          ? "Please provide feedback on this response to help improve AI quality"
          : "Please select the response you prefer to help improve AI responses"
        }
      </div>
      <div className="w-full flex flex-wrap gap-4">
        {Array.isArray(options) ? options.map((option, index) => {
          const optionContent = renderOptionContent(option);
          return (
            <div 
              key={index}
              className="flex-1 min-w-[300px] p-4 rounded-2xl bg-white shadow-md text-gray-800 cursor-pointer hover:shadow-lg transition-shadow duration-200 border-2 border-transparent hover:border-indigo-300"
              onClick={() => onSelect(index)}
            >
              {optionContent}
              <div className="mt-3 text-right">
                <span className="bg-indigo-100 text-indigo-600 px-2 py-1 rounded text-xs">
                  {options.length === 1 ? "Select to Approve" : `Option ${index + 1}`}
                </span>
              </div>
            </div>
          );
        }) : (
          <div className="text-red-500">Error: options is not an array</div>
        )}
      </div>
    </div>
  );
};

export default ResponseOptions;