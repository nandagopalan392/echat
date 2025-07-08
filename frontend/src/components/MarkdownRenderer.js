import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import '../styles/markdown.css'; // Make sure this path is correct

const MarkdownRenderer = ({ content, className = '' }) => {
  // Ensure content is a string and handle any undefined/null values
  const safeContent = typeof content === 'string' ? content : String(content || '');
  
  return (
    <div className={`markdown-content ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={{
          // Headings
          h1: ({ node, children, ...props }) => (
            <h1 className="text-2xl font-bold mt-6 mb-4" {...props}>{children}</h1>
          ),
          h2: ({ node, children, ...props }) => (
            <h2 className="text-xl font-bold mt-5 mb-3" {...props}>{children}</h2>
          ),
          h3: ({ node, children, ...props }) => (
            <h3 className="text-lg font-bold mt-4 mb-2" {...props}>{children}</h3>
          ),
          
          // Lists
          ul: ({ node, children, ...props }) => (
            <ul className="list-disc ml-5 my-4" {...props}>{children}</ul>
          ),
          ol: ({ node, children, ...props }) => (
            <ol className="list-decimal ml-5 my-4" {...props}>{children}</ol>
          ),
          
          // Links
          a: ({ node, href, children, ...props }) => (
            <a 
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 hover:underline"
              {...props}
            >
              {children}
            </a>
          ),
          
          // Code blocks
          code: ({ node, inline, className, children, ...props }) => {
            const match = /language-(\w+)/.exec(className || '');
            return !inline ? (
              <div className="my-4">
                <pre className={`bg-gray-800 text-white p-4 rounded overflow-x-auto ${className}`}>
                  <code {...props}>{children}</code>
                </pre>
              </div>
            ) : (
              <code className="bg-gray-100 px-1 py-0.5 rounded text-gray-800" {...props}>
                {children}
              </code>
            );
          },
          
          // Bold and italic
          strong: ({ node, children, ...props }) => (
            <strong className="font-bold" {...props}>{children}</strong>
          ),
          em: ({ node, children, ...props }) => (
            <em className="italic" {...props}>{children}</em>
          ),
          
          // Blockquotes
          blockquote: ({ node, children, ...props }) => (
            <blockquote className="border-l-4 border-gray-300 pl-4 italic my-4" {...props}>
              {children}
            </blockquote>
          ),
          
          // Images
          img: ({ node, src, alt, ...props }) => (
            <img 
              src={src} 
              alt={alt || ''} 
              className="max-w-full h-auto my-4 rounded"
              {...props} 
            />
          ),
        }}
      >
        {safeContent}
      </ReactMarkdown>
    </div>
  );
};

export default MarkdownRenderer;
