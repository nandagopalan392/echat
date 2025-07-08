import React, { useState } from 'react';
import MarkdownRenderer from './MarkdownRenderer';

const ThinkBubble = ({ content, isStreaming = false }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  
  // Ensure content is a string
  const safeContent = typeof content === 'string' ? content : String(content || '');
  
  // Limit initial display to first 150 characters if longer than 200
  const shouldTruncate = safeContent.length > 200 && !isExpanded;
  const displayContent = shouldTruncate ? safeContent.substring(0, 150) + '...' : safeContent;
  
  return (
    <div className="border border-indigo-200 rounded-lg p-3 bg-indigo-50 my-2 text-sm">
      <div className="flex items-center mb-2">
        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-indigo-600 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
        <span className="font-medium text-indigo-600">Thinking Process</span>
      </div>
      
      <div className="text-gray-700">
        <MarkdownRenderer content={displayContent} className="think-bubble-content" />
        {isStreaming && <span className="ml-1 animate-pulse">▎</span>}
      </div>

      {safeContent.length > 200 && (
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="text-indigo-600 hover:text-indigo-800 text-xs mt-2 focus:outline-none"
        >
          {isExpanded ? "Show less" : "Show more"}
        </button>
      )}
    </div>
  );
};

export default ThinkBubble;
