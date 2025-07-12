import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import UserMenu from './UserMenu';
import ChatSidebar from './ChatSidebar';
import UserDashboard from './UserDashboard';
import ActivityDashboard from './ActivityDashboard';  // Add this import
import FileListDashboard from './FileListDashboard';  // Add this import
import ModelSettings from './ModelSettings';  // Add ModelSettings import
import echatLogo from '../assets/echat_logo.svg';
import CircularProgress from '@mui/material/CircularProgress';
import LoadingSpinner from './LoadingSpinner';
import FileUploadProgress from './FileUploadProgress';
import ThinkBubble from './ThinkBubble';
import ResponseOptions from './ResponseOptions';
import MarkdownRenderer from './MarkdownRenderer';

// ========================================
// DEBUG CONFIGURATION
// ========================================
// Set DEBUG_MODE to true to enable console logs for debugging
// Set to false in production to reduce console noise
const DEBUG_MODE = false;

// ========================================

// Debug helper function
const debugLog = (message, ...args) => {
  if (DEBUG_MODE) {
    console.log(message, ...args);
  }
};

// Add a minimal version of the debug function that avoids all JSON parsing
const safeProcessSSE = (eventData) => {
  
  // Very simple processing - avoid JSON parsing entirely
  if (!eventData || typeof eventData !== 'string') {
    return { empty: true };
  }
  
  // Handle different message types without JSON parsing
  if (eventData.includes('type":"start')) {
    return { type: "start" };
  }
  if (eventData.includes('type":"end')) {
    return { type: "end" };
  }
  
  // Handle different content formats
  if (eventData.startsWith('plain:')) {
    return { content: eventData.substring(6) };
  }
  
  // Extract text content directly, bypassing JSON
  if (eventData.includes('content":"')) {
    const startPos = eventData.indexOf('content":"') + 10;
    let endPos = eventData.indexOf('"}', startPos);
    if (endPos === -1) {
      endPos = eventData.length;
    }
    const extractedContent = eventData.substring(startPos, endPos);
    return { content: extractedContent.replace(/\\"/g, '"') };
  }
  
  // Default fallback - just return the raw data
  return { content: eventData.replace(/^data: /, '') };
};

// Add this utility function for detailed data inspection
const inspectRawData = (data) => {
  if (!DEBUG_MODE) return;
  
  debugLog('RAW SSE DATA INSPECTION:');
  debugLog('Length:', data?.length || 0);
  
  if (!data || data.length === 0) {
    debugLog('EMPTY DATA!');
    return;
  }
  
  debugLog('First/Last chars:', data.substring(0, 50), '...', data.substring(Math.max(0, data.length - 50)));
};

// Update the debugging function at the top of the file
const debugSSE = (event) => {
  if (!DEBUG_MODE) return { empty: true };
  
  debugLog('SSE Event received');
  inspectRawData(event.data);
  
  if (!event.data || event.data.trim() === '') {
    debugLog('Empty SSE data received');
    return { empty: true };
  }
  
  // Try to parse as JSON first
  try {
    const trimmedData = event.data.trim();
    const parsed = JSON.parse(trimmedData);
    debugLog('Successfully parsed JSON:', parsed);
    return parsed;
  } catch (error) {
    debugLog('Parse error:', error.message);
    
    // Try more aggressive JSON extraction
    try {
      const firstBrace = event.data.indexOf('{');
      if (firstBrace >= 0) {
        let depth = 1;
        let lastBrace = -1;
        
        for (let i = firstBrace + 1; i < event.data.length; i++) {
          if (event.data[i] === '{') depth++;
          else if (event.data[i] === '}') {
            depth--;
            if (depth === 0) {
              lastBrace = i;
              break;
            }
          }
        }
        
        if (lastBrace > firstBrace) {
          const jsonCandidate = event.data.substring(firstBrace, lastBrace + 1);
          debugLog('Extracted JSON candidate:', jsonCandidate);
          const parsed = JSON.parse(jsonCandidate);
          debugLog('Successfully parsed extracted JSON:', parsed);
          return parsed;
        }
      }
    } catch (innerError) {
      debugLog('Failed to extract valid JSON:', innerError);
    }
    
    // Handle legacy plain: format if present
    if (event.data.startsWith('plain:')) {
      const plainText = event.data.substring(6);
      debugLog('Legacy plaintext format detected:', plainText);
      return { content: plainText };
    }
    
    // When all else fails, try to extract anything useful
    // Look for data: pattern and extract content after it
    if (event.data.startsWith('data:')) {
      try {
        const content = event.data.substring(5).trim();
        console.log('Extracted content after data: prefix:', content);
        
        // Try to parse this content as JSON
        try {
          const parsed = JSON.parse(content);
          console.log('Successfully parsed content after data: prefix:', parsed);
          return parsed;
        } catch (jsonError) {
          // If it's not valid JSON, just return it as raw content
          return { content: content, rawText: true };
        }
      } catch (extractError) {
        console.error('Error extracting content after data: prefix:', extractError);
      }
    }
    
    // Return the raw data as content as a last resort
    return { content: event.data, rawText: true };
  }
};

// Add a helper function to parse SSE formatted responses
const extractCleanJsonFromSSE = (data) => {
  // This function extracts clean JSON from SSE data
  try {
    // For SSE messages without data: prefix
    if (!data.startsWith('data:')) {
      return JSON.parse(data);
    }
    
    // For standard SSE messages with data: prefix
    const jsonString = data.substring(data.indexOf('{'), data.lastIndexOf('}') + 1);
    return JSON.parse(jsonString);
  } catch (e) {
    console.error('Failed to extract JSON from SSE data:', e);
    return { content: data };
  }
};

// Simplify parseSSEResponse to handle pure text with think tags
const parseSSEResponse = async (response, onChunk) => {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let messageData = [];
  
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      
      buffer += decoder.decode(value, { stream: true });
      
      // Process the buffer line by line for proper JSON handling
      let lines = buffer.split('\n');
      // Keep the last potentially incomplete line in the buffer
      buffer = lines.pop() || '';
      
      for (const line of lines) {
        if (!line.trim() || line.startsWith(': ping')) continue;
        
        // Extract data content
        if (line.startsWith('data:')) {
          try {
            // Try to extract and parse JSON
            const jsonContent = line.substring(5).trim();
            if (!jsonContent) continue;
            
            console.log("Processing chunk:", jsonContent.substring(0, 50) + "...");
            
            const data = JSON.parse(jsonContent);
            messageData.push(data);
            
            // Process the data based on its content
            if (data.thinking !== undefined) {
              // Format a combined message with thinking tags
              const combinedContent = data.thinking ? 
                `<think>${data.thinking}</think>${data.content || ''}` : 
                (data.content || '');
              
              onChunk(combinedContent, data.isComplete || false);
            } else if (data.content) {
              // Just forward the content
              onChunk(data.content, data.isComplete || false);
            } else if (data.full_response) {
              // Use full_response if available
              onChunk(data.full_response, true);
            }
          } catch (e) {
            console.warn("Failed to parse JSON:", e);
            
            // If JSON parsing fails, pass the raw content
            const rawContent = line.substring(5).trim();
            onChunk(rawContent, false);
          }
        }
      }
    }
    
    // Create a clean final result with all collected data
    return {
      messages: messageData,
      text: messageData.length > 0 ? 
        (messageData[messageData.length - 1].content || 
         messageData[messageData.length - 1].full_response || '') : '',
      isComplete: true
    };
  } catch (error) {
    console.error("Error parsing SSE response:", error);
    return { 
      text: "Error processing response", 
      error: true,
      message: error.message,
      messages: messageData
    };
  }
};

// New helper function to extract content from data lines without full JSON parsing
function extractContentFromDataLine(dataLine) {
  // Case 1: Simple content extraction using regex
  const contentMatch = dataLine.match(/"content":\s*"([^"]*)"/);
  if (contentMatch && contentMatch[1]) {
    return contentMatch[1];
  }
  
  // Case 2: Try safe JSON parsing
  try {
    const jsonData = JSON.parse(dataLine);
    if (jsonData.content) {
      return jsonData.content;
    } else if (jsonData.full_response) {
      return jsonData.full_response;
    }
  } catch (e) {
    // Failed to parse as JSON - continue to other methods
  }
  
  // Case 3: Extract content from chunks that look like {content: "X"}
  const chunks = dataLine.split(/\}\s*\{/).filter(Boolean);
  let extractedContent = '';
  
  for (const chunk of chunks) {
    // Wrap in braces if they were removed by the split
    const wrappedChunk = chunk.startsWith('{') ? chunk : '{' + chunk;
    const endingChunk = wrappedChunk.endsWith('}') ? wrappedChunk : wrappedChunk + '}';
    
    try {
      // Try to parse each chunk
      const fixedChunk = endingChunk
        .replace(/\\+"/g, '"') // Fix escaped quotes
        .replace(/\\+{/g, '{') // Fix escaped braces
        .replace(/\\+}/g, '}'); // Fix escaped braces
        
      const parsedChunk = JSON.parse(fixedChunk);
      if (parsedChunk.content) {
        extractedContent += parsedChunk.content;
      }
    } catch (e) {
      // Skip unparseable chunks
    }
  }
  
  // Return whatever content we were able to extract
  return extractedContent;
}

// Helper function to extract RLHF options from text content
function extractRLHFOptionsFromText(text) {
  // Function to help identify likely option boundaries
  const findOptionBoundaries = (text) => {
    const markers = [
      // Look for numbered options
      {regex: /\b(option\s*1|option\s*a)[\s\:]/i, isStart: true, optionNum: 0},
      {regex: /\b(option\s*2|option\s*b)[\s\:]/i, isStart: true, optionNum: 1},
      
      // Look for think tags as option markers (common in RLHF outputs)
      {regex: /<think>/i, isStart: true, optionNum: null},
      {regex: /<\/think>/i, isStart: false, optionNum: null}
    ];
    
    const boundaries = [];
    for (const marker of markers) {
      const match = text.match(marker.regex);
      if (match) {
        boundaries.push({
          pos: match.index,
          isStart: marker.isStart,
          optionNum: marker.optionNum
        });
      }
    }
    
    return boundaries.sort((a, b) => a.pos - b.pos);
  };
  
  // Main extraction logic
  try {
    // Try to find option boundaries
    const boundaries = findOptionBoundaries(text);
    
    // If we have clear option markers, extract based on those
    if (boundaries.length >= 2) {
      const options = [];
      let currentOption = "";
      let currentOptionNum = -1;
      
      for (const boundary of boundaries) {
        if (boundary.isStart && boundary.optionNum !== null) {
          // Start of a numbered option
          if (currentOptionNum >= 0 && currentOption) {
            options[currentOptionNum] = currentOption.trim();
          }
          currentOptionNum = boundary.optionNum;
          currentOption = "";
        } else if (!boundary.isStart && currentOptionNum >= 0) {
          // End of current option
          options[currentOptionNum] = currentOption.trim();
          currentOptionNum = -1;
        }
      }
      
      // Add any final option
      if (currentOptionNum >= 0 && currentOption) {
        options[currentOptionNum] = currentOption.trim();
      }
      
      // Return only if we found at least 2 options
      if (options.length >= 2) {
        return options;
      }
    }
    
    // Fallback: Split the text in half for two options
    const midpoint = Math.floor(text.length / 2);
    let splitPoint = text.indexOf('\n\n', midpoint);
    if (splitPoint === -1) splitPoint = midpoint;
    
    return [
      text.substring(0, splitPoint).trim(),
      text.substring(splitPoint).trim()
    ];
  } catch (e) {
    console.warn("Failed to extract RLHF options:", e);
    return [];
  }
}

// Completely rewrite the RLHF option processing function to handle complex responses
const processRLHFOptions = (options) => {
  if (!Array.isArray(options)) return [];
  
  return options.map(option => {
    // If not a string, convert to string and return
    if (typeof option !== 'string') {
      return String(option);
    }
    
    // Detect if this is a full object response with think tags
    if (option.includes('<think>') && option.includes('</think>')) {
      // Extract content after </think> tag as the actual answer
      const parts = option.split('</think>');
      if (parts.length > 1) {
        // Get everything after the </think> tag
        return parts[1]
          .replace(/\\n/g, '\n')
          .replace(/\\\\/g, '\\')
          .replace(/\\"/g, '"')
          .replace(/\{"content":\s*"([^"]*)"\}/g, '$1')
          .replace(/\{"type":[^}]*\}/g, '')
          .replace(/"$/g, '')
          .trim();
      }
    }
    
    // For options containing only content fields
    if (option.includes('"content":')) {
      // Extract all content values and join them
      const contentPattern = /"content":\s*"([^"]*)"/g;
      const contents = [];
      let match;
      
      while ((match = contentPattern.exec(option)) !== null) {
        if (match[1]) {
          contents.push(match[1]);
        }
      }
      
      if (contents.length > 0) {
        return contents.join('')
          .replace(/\\n/g, '\n')
          .replace(/\\\\/g, '\\')
          .replace(/\\"/g, '"')
          .trim();
      }
    }
    
    // For options that include the full_response field
    if (option.includes('"full_response":')) {
      const fullResponseMatch = /"full_response":\s*"([^"]+)"/g.exec(option);
      if (fullResponseMatch && fullResponseMatch[1]) {
        return fullResponseMatch[1]
          .replace(/\\n/g, '\n')
          .replace(/\\\\/g, '\\')
          .replace(/\\"/g, '"');
      }
    }
    
    // Remove all JSON/SSE formatting as a last resort
    return option
      .replace(/^{"type": "[^"]*"}/, '')
      .replace(/\{"type": "[^"]*"\}/, '')
      .replace(/\{"content": "([^"]*)"\}/g, '$1')
      .replace(/^"/, '')
      .replace(/"$/, '')
      .replace(/\\n/g, '\n')
      .replace(/\\\\/g, '\\')
      .replace(/\\"/g, '"')
      .trim();
  });
};

// Add a function to normalize whitespace in messages
const normalizeWhitespace = (text) => {
  if (typeof text !== 'string') return text;
  
  // Replace multiple consecutive whitespace with a single space
  return text
    .replace(/[\n\r]+/g, '\n')      // Normalize newlines
    .replace(/[ \t]+/g, ' ')        // Normalize spaces and tabs
    .replace(/\n[ \t]+/g, '\n')     // Remove leading space on lines
    .replace(/[ \t]+\n/g, '\n')     // Remove trailing space on lines
    .trim();                         // Remove leading/trailing whitespace
};

// Improved cleanContent function with better handling of malformed escape sequences
const cleanStreamingContent = (content) => {
  // Ensure content is a string
  if (typeof content !== 'string') {
    // If it's an object with a 'content' property, use that
    if (content && typeof content === 'object' && content.content) {
      content = content.content;
    } else if (content && typeof content === 'object' && content.type === 'thinking') {
      // If it's a thinking block object
      return `<think>${content.content || ''}</think>`;
    } else if (content && typeof content === 'object' && content.type === 'response') {
      // If it's a response object
      return content.content || '';
    } else {
      // Convert to string as a fallback
      try {
        content = String(content || '');
      } catch (e) {
        console.error("Failed to convert content to string:", e);
        return '';
      }
    }
  }

  // Only proceed with string operations if content is a string
  if (typeof content !== 'string') return '';
  
  // Extract full_response if present
  if (content.includes('"full_response":')) {
    try {
      const fullResponseMatch = /"full_response":\s*"((?:\\.|[^"\\])*)"/s.exec(content);
      if (fullResponseMatch && fullResponseMatch[1]) {
        return normalizeWhitespace(
          fullResponseMatch[1]
            .replace(/\\n/g, '\n')  // Convert escaped newlines
            .replace(/\\t/g, '\t')  // Convert escaped tabs
            .replace(/\\\\/g, '\\') // Convert escaped backslashes
            .replace(/\\"/g, '"')   // Convert escaped quotes
            .replace(/\n{2,}/g, '\n\n') // Normalize multiple newlines to max 2
        );
      }
    } catch (e) {
      console.warn("Failed to extract full_response:", e);
    }
  }
  
  // First replace all escaped JSON sequence issues
  let cleaned = content
    // Fix malformed JSON sequences first
    .replace(/\\+n/g, '\n')        // Convert any number of backslashes + n to newline
    .replace(/\\+"/g, '"')         // Fix escaped quotes
    .replace(/\\+{/g, '{')         // Fix escaped braces
    .replace(/\\+}/g, '}')         // Fix escaped braces
    .replace(/"{/g, '{')           // Remove quotes before object start
    .replace(/}"/g, '}')           // Remove quotes after object end
    .replace(/^"/, '')             // Remove start quote
    .replace(/"$/, '')             // Remove end quote
    // Then remove JSON structure artifacts
    .replace(/^{"type": "start"}\s*/, '')              // Remove start markers
    .replace(/{"type": "end"[^}]*}\s*$/, '')           // Remove end markers
    .replace(/\{"type":"[^"]*"\}/g, '')                // Remove type markers
    .replace(/\{"type":[^}]*\}/g, '')                  // Remove type markers
    .replace(/^data:\s*/, '')                          // Remove SSE data prefix
    .replace(/\{"content":\s*"([^"]*)"\}/g, '$1')      // Extract content values
    // Finally normalize whitespace and special characters
    .replace(/\n{3,}/g, '\n\n')                        // Normalize excessive newlines
    .replace(/\s{3,}/g, ' ')                           // Normalize excessive spaces
    .replace(/\bdata:\s+/g, '')                        // Remove embedded data prefixes
    .replace(/\\"type\\":[^,}]*,?/g, '');              // Remove embedded type fields
  
  return normalizeWhitespace(cleaned);
};

// Add helper function to clean string options for RLHF
function cleanStringOption(option) {
  if (typeof option !== 'string') {
    return String(option);
  }
  
  return option
    .replace(/\\n/g, '\n')
    .replace(/\\\\/g, '\\')
    .replace(/\\"/g, '"')
    .replace(/{"content":\s*"([^"]*)"\}/g, '$1')
    .replace(/^"/, '')
    .replace(/"$/, '')
    .trim();
}

// Add helper function to extract options from backend responses
function extractRLHFOptionsFromBackend(text) {
  // Look for option markers like "Option 1:" or numbered points
  const optionMarkers = [
    /\b(Option\s*1|Option\s*A)[\s\:]/i,
    /\b(Option\s*2|Option\s*B)[\s\:]/i,
    /\b(1\.)\s+/,
    /\b(2\.)\s+/,
    /<think>/i
  ];
  
  // Check if the text contains any option markers
  const hasMarkers = optionMarkers.some(marker => marker.test(text));
  
  if (hasMarkers) {
    try {
      // Try to split by think tags first (common RLHF format)
      if (text.includes('<think>') && text.includes('</think>')) {
        const parts = text.split('</think>');
        if (parts.length > 1) {
          // Get the parts after each think tag
          return parts.map(part => {
            const afterThinkTag = part.indexOf('<think>');
            return afterThinkTag > 0 ? 
              part.substring(0, afterThinkTag).trim() : 
              part.trim();
          }).filter(Boolean);
        }
      }
      
      // Try numbered options
      if (text.match(/\b(Option\s*1|Option\s*A|1\.)[\s\:]/i)) {
        // Split by option markers
        const option1Start = text.search(/\b(Option\s*1|Option\s*A|1\.)[\s\:]/i);
        const option2Start = text.search(/\b(Option\s*2|Option\s*B|2\.)[\s\:]/i);
        
        if (option1Start >= 0 && option2Start > option1Start) {
          return [
            text.substring(option1Start, option2Start).replace(/\b(Option\s*1|Option\s*A|1\.)[\s\:]/i, '').trim(),
            text.substring(option2Start).replace(/\b(Option\s*2|Option\s*B|2\.)[\s\:]/i, '').trim()
          ];
        }
      }
    } catch (e) {
      console.warn("Failed to extract RLHF options by markers:", e);
    }
  }
  
  // Simple fallback: just split in half
  const halfway = Math.floor(text.length / 2);
  const splitPos = text.indexOf('\n\n', halfway);
  
  return [
    text.substring(0, splitPos > 0 ? splitPos : halfway).trim(),
    text.substring(splitPos > 0 ? splitPos + 2 : halfway).trim()
  ];
}

// Add this function before the Chat component
const cleanSelectedResponse = (response) => {
    debugLog("cleanSelectedResponse called with:", response);
    
    if (!response) return '';
    
    try {
        // Handle structured response object (new format)
        if (typeof response === 'object' && response !== null) {
            debugLog("Processing structured response object");
            // For structured responses, preserve the full structure with thinking and content
            if (response.thinking && response.content) {
                console.log("Preserving structured response with thinking and content");
                // Format as expected by the frontend (with think tags)
                return `<think>${response.thinking}</think>${response.content}`;
            }
            // If only content field, return just content
            if (response.content) {
                debugLog("Extracting content from structured object:", response.content);
                return response.content;
            }
            // If no content field, stringify the whole object as fallback
            console.log("No content field found, using whole object");
            return JSON.stringify(response);
        }
        
        // Handle legacy string responses
        if (typeof response === 'string') {
            console.log("Processing string response");
            // Try parsing if it's a JSON string
            if (response.startsWith('{') || response.startsWith('[')) {
                try {
                    const parsed = JSON.parse(response);
                    if (parsed.thinking && parsed.content) {
                        console.log("Parsed structured response with thinking and content");
                        return `<think>${parsed.thinking}</think>${parsed.content}`;
                    }
                    if (parsed.content) {
                        console.log("Extracted content from JSON string:", parsed.content);
                        return parsed.content;
                    }
                    if (parsed.thinking) {
                        console.log("Only thinking found, using full parsed object");
                        return JSON.stringify(parsed);
                    }
                } catch (e) {
                    // Not valid JSON, continue with string cleaning
                    console.log("Not valid JSON, continuing with string cleaning");
                }
            }
            
            // Clean any remaining JSON/SSE formatting but preserve think tags
            const cleaned = response
                .replace(/\{"type":[^}]*\}/g, '')                // Remove type markers
                .replace(/\{"content":\s*"([^"]*)"\}/g, '$1')    // Extract content values
                .replace(/^data:\s*/, '')                        // Remove SSE prefix
                .replace(/\\n/g, '\n')                           // Fix newlines
                .replace(/\\\\/g, '\\')                          // Fix backslashes
                .replace(/\\"/g, '"')                            // Fix quotes
                .replace(/^"/, '')                               // Remove start quote
                .replace(/"$/, '')                               // Remove end quote
                .trim();
            console.log("Cleaned string result:", cleaned);
            return cleaned;
        }
        
        // Fallback for other types
        const fallback = String(response).trim();
        console.log("Fallback result:", fallback);
        return fallback;
    } catch (e) {
        console.error('Error cleaning response:', e);
        return String(response).trim();
    }
};

const Chat = () => {
    const [messages, setMessages] = useState([]);
    const [inputMessage, setInputMessage] = useState('');
    const [sessionId, setSessionId] = useState(null);
    const [isAdmin, setIsAdmin] = useState(false);
    const [users, setUsers] = useState([]);
    const [newUser, setNewUser] = useState({ username: '', password: '' });
    const [userError, setUserError] = useState('');
    const [showUserManagement, setShowUserManagement] = useState(false);
    const [selectedRole, setSelectedRole] = useState('Engineer');
    const [selectedUsername, setSelectedUsername] = useState(null);
    const [showActivityDashboard, setShowActivityDashboard] = useState(false);
    const [notification, setNotification] = useState(null);
    const [uploadStatus, setUploadStatus] = useState(null);
    const [showFileList, setShowFileList] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(null);
    const [isSessionSwitching, setIsSessionSwitching] = useState(false); // Add new state to track session switching
    const [streamingMessage, setStreamingMessage] = useState("");
    const [showResponseOptions, setShowResponseOptions] = useState(false);
    const [responseOptions, setResponseOptions] = useState([]);
    const [optionsMessageId, setOptionsMessageId] = useState(null);
    const [showKnowledgeHub, setShowKnowledgeHub] = useState(false);
    const [showModelSettings, setShowModelSettings] = useState(false);

    const roles = ['Engineer', 'Manager', 'Business Development', 'Associate'];


    useEffect(() => {
        checkAdminStatus();
        // Optionally, refresh admin status on token/user change
        // eslint-disable-next-line
    }, []);

    // Close Knowledge Hub dropdown when clicking outside
    useEffect(() => {
        const handleClickOutside = (event) => {
            if (showKnowledgeHub && !event.target.closest('.knowledge-hub-container')) {
                setShowKnowledgeHub(false);
            }
        };

        if (showKnowledgeHub) {
            document.addEventListener('mousedown', handleClickOutside);
            return () => {
                document.removeEventListener('mousedown', handleClickOutside);
            };
        }
    }, [showKnowledgeHub]);

    const checkAdminStatus = async () => {
        debugLog("Checking admin status...");
        try {
            const response = await api.get('/api/admin/users');
            debugLog("Admin check response:", response);
            if (response && response.users) {
                debugLog("Setting isAdmin to true");
                setIsAdmin(true);
                setUsers(response.users);
            } else {
                console.log("Setting isAdmin to false - no users in response");
                setIsAdmin(false);
            }
        } catch (error) {
            console.log("Setting isAdmin to false - error occurred:", error);
            setIsAdmin(false);
            // Optionally show a notification or log
            console.warn('Admin check failed:', error?.message || error);
        }
    };

    const showNotification = (message, isError = false) => {
        setNotification({
            message,
            isError
        });
        setTimeout(() => setNotification(null), 1000);
    };

    const showUploadStatus = (message, isError = false) => {
        setUploadStatus({
            message,
            isError
        });
        setTimeout(() => setUploadStatus(null), 1000);
    };

    const handleFileUpload = async (e, isFolder = false) => {
        const files = Array.from(e.target.files);
        const totalFiles = files.length;
        let completedFiles = 0;
        let totalProgress = 0;

        console.log(`Processing ${totalFiles} files`);
        
        try {
            for (const file of files) {
                const fileExt = file.name.split('.').pop().toLowerCase();
                if (!['xlsx', 'csv', 'docx', 'pdf'].includes(fileExt)) {
                    showUploadStatus('Unsupported file type', true);
                    continue;
                }
                
                // Calculate overall progress including completed files
                const baseProgress = (completedFiles / totalFiles) * 100;
                
                setUploadProgress({
                    fileName: isFolder ? `Folder Upload: ${files[0].webkitRelativePath.split('/')[0]}` : file.name,
                    progress: baseProgress,
                    currentFile: completedFiles + 1,
                    totalFiles: totalFiles
                });
                
                const folderPath = isFolder && file.webkitRelativePath 
                    ? file.webkitRelativePath 
                    : file.name;
                
                await api.uploadFileWithProgress(
                    file,
                    isFolder,
                    folderPath,
                    (progress) => {
                        const fileProgress = progress / totalFiles;
                        const currentTotalProgress = baseProgress + fileProgress;
                        
                        setUploadProgress(prev => ({
                            ...prev,
                            fileName: isFolder ? `Folder Upload: ${files[0].webkitRelativePath.split('/')[0]}` : file.name,
                            progress: currentTotalProgress,
                            currentFile: completedFiles + 1,
                            totalFiles: totalFiles,
                            currentFileName: file.name
                        }));
                    }
                );

                completedFiles++;
                totalProgress = (completedFiles / totalFiles) * 100;
            }
            
            showUploadStatus(isFolder ? 'Folder uploaded successfully' : 'File uploaded successfully');
        } catch (error) {
            console.error('Upload error:', error);
            showUploadStatus(`Failed to upload: ${error.message}`, true);
        } finally {
            // Clear progress after a delay
            setTimeout(() => {
                setUploadProgress(null);
            }, 1000);
        }
    };

    const handleSessionSelect = async (newSessionId) => {
        try {
            console.log('Switching to session:', newSessionId);
            
            // Prevent multiple session switches at once
            if (isSessionSwitching) return;
            setIsSessionSwitching(true);
            
            // Clear messages immediately
            setMessages([]);
            
            // If clicking same session, reset
            if (newSessionId === sessionId) {
                setSessionId(null);
                setIsSessionSwitching(false);
                return;
            }

            // Set loading state
            setIsProcessing(true);

            // Update session ID first
            setSessionId(newSessionId);

            if (newSessionId) {
                // Fetch messages for new session
                const response = await api.getChatMessages(newSessionId);
                console.log('Fetched messages:', response);

                if (response && response.messages) {
                    // Ensure we're still on the same session before updating messages
                    setMessages(response.messages.map((msg, index) => ({
                        id: `${newSessionId}-${index}`,
                        content: msg.content,
                        isUser: msg.isUser,
                        timestamp: msg.timestamp
                    })));
                }
            }
        } catch (error) {
            console.error('Error switching sessions:', error);
            showNotification('Error loading chat history', true);
            setMessages([]); // Clear messages on error
        } finally {
            setIsProcessing(false);
            setIsSessionSwitching(false);
        }
    };

    const handleAddUser = async (e) => {
        e.preventDefault();
        setUserError('');
        
        if (!newUser.username || !newUser.password) {
            setUserError('Username and password are required');
            return;
        }

        try {
            await api.post('/api/admin/add-user', {
                ...newUser,
                role: selectedRole
            });
            setNewUser({ username: '', password: '' });
            checkAdminStatus(); // Refresh user list
        } catch (error) {
            setUserError(error.response?.data?.detail || 'Error adding user');
            console.error('Error adding user:', error);
        }
    };

    const handleUserClick = (username) => {
        setSelectedUsername(username);
    };

    const handleResponseSelect = async (index) => {
        try {
            const selectedResponse = responseOptions[index];
            
            const optionMessage = messages.find(msg => msg.id === optionsMessageId);
            
            // Make isRLHF true by default since this function is only called for RLHF scenarios
            const isRLHF = true;
            
            debugLog("RLHF check - optionMessage:", optionMessage);
            console.log("RLHF check - isRLHF:", isRLHF);
            
            // Process the selected response using the updated cleaning function
            const finalContent = cleanSelectedResponse(selectedResponse);
            debugLog("Final content after cleaning:", finalContent);
            debugLog("Final content type:", typeof finalContent);
            debugLog("Original message content:", optionMessage?.content);
            
            // Keep the full structured content (including think tags) for proper rendering
            const safeContent = finalContent;
            debugLog("Safe content to store:", safeContent);
            
            // Update message with cleaned response
            setMessages(prevMessages => {
                const updatedMessages = [...prevMessages];
                const messageIndex = updatedMessages.findIndex(msg => msg.id === optionsMessageId);
                
                if (messageIndex !== -1) {
                    debugLog("Updating message at index:", messageIndex);
                    debugLog("Old message:", updatedMessages[messageIndex]);
                    
                    updatedMessages[messageIndex] = {
                        ...updatedMessages[messageIndex],
                        content: safeContent,
                        hasResponseOptions: false,
                        selectedOption: index,
                        isStreaming: false,
                        rlhf_enabled: false,  // Disable RLHF UI after selection
                        preserveThinking: true  // Flag to ensure thinking is preserved
                    };
                    
                    debugLog("Updated message:", updatedMessages[messageIndex]);
                    
                    debugLog("New message:", updatedMessages[messageIndex]);
                }
                
                return updatedMessages;
            });
            
            // Hide response options UI
            setShowResponseOptions(false);
            setResponseOptions([]);
            setOptionsMessageId(null);  // Clear the options message ID
            
            // Handle RLHF feedback and save selected response to backend
            if (isRLHF && sessionId) {
                debugLog('=== STARTING RLHF FEEDBACK PROCESS ===');
                debugLog('Session ID:', sessionId);
                debugLog('Selected Index:', index);
                debugLog('Is RLHF:', isRLHF);
                debugLog('Safe Content:', safeContent);
                
                setIsProcessing(true);
                try {
                    // Submit RLHF feedback
                    debugLog('About to call api.submitRLHFFeedback...');
                    const feedbackResponse = await api.submitRLHFFeedback(sessionId, index);
                    debugLog('RLHF feedback response received:', feedbackResponse);
                    
                    // Save the selected response content to the database
                    debugLog('About to call api.updateMessage...');
                    const updateResponse = await api.updateMessage(sessionId, safeContent);
                    debugLog('Message update response received:', updateResponse);
                    
                    showNotification('Thanks for your feedback!');
                } catch (error) {
                    console.error('Error submitting RLHF feedback or updating message:', error);
                    showNotification('Failed to submit feedback', true);
                }
            } else {
                debugLog('RLHF feedback skipped - isRLHF:', isRLHF, 'sessionId:', sessionId);
            }
        } catch (error) {
            console.error('Error handling response selection:', error);
            showNotification('Failed to process selection', true);
        } finally {
            setIsProcessing(false);
        }
    };

    const formatMessage = (content, isInThinkBlock = false, currentThinkContent = "") => {
        if (!content) return content;

        // Create a deep copy of content to avoid modifying the original string
        let processedContent = content;
        
        // First, handle complete think blocks with regex
        const parts = [];
        let lastIndex = 0;
        const regex = /<think>([\s\S]*?)<\/think>/g;
        let match;
        
        // Process all complete <think>...</think> tags
        while ((match = regex.exec(processedContent)) !== null) {
            // Add content before the <think> tag with markdown rendering
            if (match.index > lastIndex) {
                const textContent = processedContent.slice(lastIndex, match.index);
                parts.push(
                    <div key={`text-${lastIndex}-${match.index}`}>
                        <MarkdownRenderer content={textContent} />
                    </div>
                );
            }
            
            // Add the think bubble component with the content inside the tags
            parts.push(
                <ThinkBubble 
                    key={`think-${match.index}`} 
                    content={match[1].trim()} 
                    isStreaming={false}
                />
            );
            
            lastIndex = match.index + match[0].length;
        }
        
        // Add any remaining content after the last complete <think> tag
        if (lastIndex < processedContent.length) {
            // Check for incomplete <think> tag
            const incompleteThinkStart = processedContent.slice(lastIndex).indexOf("<think>");
            
            if (incompleteThinkStart !== -1 && isInThinkBlock) {
                // Add content before the incomplete <think> tag with markdown rendering
                if (incompleteThinkStart > 0) {
                    const textContent = processedContent.slice(lastIndex, lastIndex + incompleteThinkStart);
                    parts.push(
                        <div key={`text-end-incomplete`}>
                            <MarkdownRenderer content={textContent} />
                        </div>
                    );
                }
                
                // Extract the content inside the incomplete think tag
                const thinkContent = processedContent.slice(lastIndex + incompleteThinkStart + 7);
                
                // Add the streaming think bubble with the incomplete content
                parts.push(
                    <ThinkBubble 
                        key="think-incomplete" 
                        content={thinkContent} 
                        isStreaming={true}
                    />
                );
            } else {
                // No incomplete think tag, just add remaining content with markdown rendering
                const textContent = processedContent.slice(lastIndex);
                parts.push(
                    <div key="text-end">
                        <MarkdownRenderer content={textContent} />
                    </div>
                );
            }
        }
        
        return parts.length > 0 ? (
            <div className="flex flex-col gap-2">
                {parts}
            </div>
        ) : <MarkdownRenderer content={processedContent} />;
    };

    // Add a helper function to render messages with streaming effect
    const renderMessageContent = (message) => {
        // If this message has a selectedOption, it means the user has chosen a response
        // Handle structured content with proper formatting
        if (message.selectedOption !== undefined) {
            const content = message.content;
            
            // If content is a structured object, render it with the same formatting as ResponseOptions
            if (typeof content === 'object' && content !== null) {
                const hasThinking = content.thinking && String(content.thinking).trim() !== '';
                const hasContent = content.content && String(content.content).trim() !== '';
                const hasStyle = content.style && String(content.style).trim() !== '';
                
                return (
                    <div className="space-y-3">
                        {hasThinking && (
                            <div className="text-sm text-gray-600 bg-gray-50 p-3 rounded-lg border-l-4 border-blue-300">
                                <div className="font-semibold text-blue-700 mb-1">Thinking Process:</div>
                                <MarkdownRenderer content={String(content.thinking)} className="text-sm" />
                            </div>
                        )}
                        {hasContent && (
                            <div className="text-gray-800">
                                <MarkdownRenderer content={String(content.content)} />
                            </div>
                        )}
                        {hasStyle && (
                            <div className="text-xs text-gray-500 mt-2">
                                Style: <span className="capitalize font-medium">{String(content.style)}</span>
                            </div>
                        )}
                    </div>
                );
            }
            
            // For string content with think tags, use formatMessage to render properly
            if (typeof content === 'string' && content.includes('<think>')) {
                return formatMessage(content);
            }
            
            // For other string content, use MarkdownRenderer
            return <MarkdownRenderer content={String(content)} />;
        }
        
        // Ensure content is always a string before processing
        let displayContent = message.content;
        if (typeof displayContent !== 'string') {
            if (typeof displayContent === 'object' && displayContent !== null) {
                // Handle structured response object
                if (displayContent.content) {
                    displayContent = displayContent.content;
                } else {
                    displayContent = JSON.stringify(displayContent);
                }
            } else {
                // Convert any other type to string
                displayContent = String(displayContent);
            }
        }
        
        // Handle the case where content is a JSON string containing thinking and content fields
        // Only for messages that haven't been selected yet
        if (displayContent.includes('"thinking"') && displayContent.includes('"content"')) {
          try {
            const jsonData = JSON.parse(displayContent);
            if (jsonData.content !== undefined) {
              // For JSON content, just use the content field
              displayContent = jsonData.content;
            }
          } catch (e) {
            // If parsing fails, try to extract using regex
            const contentMatch = displayContent.match(/"content":\s*"((?:\\.|[^"\\])*)"/);
            
            if (contentMatch) {
              const content = contentMatch[1]
                .replace(/\\"/g, '"')
                .replace(/\\n/g, '\n');
              
              displayContent = content;
            }
          }
        }
        
        // Clean up any remaining JSON/SSE formatting
        const cleanedContent = displayContent
          .replace(/^data:\s*/, '')
          .replace(/\{"type":[^}]*\}/g, '')
          .replace(/\\n/g, '\n')
          .replace(/\\\\/g, '\\')
          .replace(/\\"/g, '"')
          .trim();
        
        // Apply formatMessage to handle think blocks - this works for all content with think tags
        const formattedContent = formatMessage(
          cleanedContent,
          message.isInThinkBlock,
          message.currentThinkContent
        );
        
        // Add streaming cursor if needed
        if (message.isStreaming && !message.isUser) {
          return (
            <div className="whitespace-pre-wrap break-words">
              {formattedContent}
              <span className="animate-pulse ml-1">▎</span>
            </div>
          );
        }
        
        return formattedContent;
    };

    // Improved updateMessage function to be used in handleSubmit
    const updateMessageContent = (id, content, isStreaming, setMessages, rlhfEnabled = false) => {
        debugLog("updateMessageContent called with:", { id, content, isStreaming, rlhfEnabled });
        
        // Handle different content types
        let processedContent = content;
        
        // For structured objects, preserve them as-is so they can be rendered properly
        if (typeof content === 'object' && content !== null) {
            console.log("Preserving structured object in message content");
            processedContent = content;
        }
        // If content is a stringified JSON object
        else if (typeof content === 'string' && 
            (content.startsWith('{') || content.includes('"thinking"'))) {
          try {
            // Try to parse JSON to extract its content
            const jsonData = JSON.parse(content);
            if (jsonData.content !== undefined || jsonData.thinking !== undefined) {
              // For structured JSON, preserve the object structure
              console.log("Preserving parsed structured object");
              processedContent = jsonData;
            }
          } catch (e) {
            // If parsing fails, use content as is
            console.warn("Failed to parse content as JSON:", e);
          }
        }
        // Ensure we always have a valid value for rendering
        if (processedContent === null || processedContent === undefined) {
            processedContent = '';
        }
      
        debugLog("Final processed content:", processedContent);
        
        // Use a functional update to ensure we're working with the latest state
        setMessages(prevMessages => 
          prevMessages.map(msg => 
            msg.id === id 
              ? { 
                  ...msg, 
                  content: processedContent,
                  isStreaming,
                  rlhfEnabled: rlhfEnabled  // Use camelCase for consistency
                } 
              : msg
          )
        );
    };

    // Improved handleSubmit function with better error handling
const handleSubmit = async (e) => {
  e.preventDefault();
  if (!inputMessage.trim()) return;
  
  // Add user message immediately 
  const userMsgObj = { 
    id: `user-${Date.now()}`,
    content: inputMessage, 
    isUser: true 
  };
  setMessages(prevMessages => [...prevMessages, userMsgObj]);
  
  // Create assistant message placeholder
  const aiMessageId = `ai-${Date.now()}`;
  setMessages(prevMessages => [
    ...prevMessages, 
    { id: aiMessageId, content: "", isUser: false, isStreaming: true }
  ]);
  
  setInputMessage('');
  setIsProcessing(true);
  
  // Set a timeout for the entire request
  const timeoutDuration = 300000; // 300 seconds (5 minutes)
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutDuration);
  
  try {
    debugLog("Making API request to /api/chat/send");
    
    const response = await fetch('/api/chat/send', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      },
      body: JSON.stringify({
        content: inputMessage,
        session_id: sessionId,
        rlhf: true
      }),
      signal: controller.signal // Add abort signal
    });
    
    // Clear the timeout since the request has completed
    clearTimeout(timeoutId);
    
    if (!response.ok) {
      if (response.status === 504) {
        // Special handling for gateway timeout
        throw new Error("The server took too long to respond. Please try a shorter question or try again later.");
      }
      throw new Error(`API error: ${response.status}`);
    }

    // Parse response as JSON only
    const responseData = await response.json();
    
    // Handle errors from the backend
    if (responseData.error) {
      throw new Error(responseData.error);
    }
    
    // Check if we have RLHF options to display - simplified condition
    const hasResponseOptions = responseData.response_options && Array.isArray(responseData.response_options) && responseData.response_options.length > 0;
    const isRlhfEnabled = responseData.rlhf_enabled === true;
    
    if (hasResponseOptions && isRlhfEnabled) {
      // Show the prompt message asking for user to select/review
      updateMessageContent(
        aiMessageId, 
        responseData.content || "Please review this response and provide feedback:", 
        false, 
        setMessages,
        true  // Enable RLHF for this message
      );
      
      // Show RLHF options UI
      setResponseOptions(responseData.response_options);
      setOptionsMessageId(aiMessageId);
      setShowResponseOptions(true);
    } else {
      // Regular response - just use content without think tags
      // The thinking information is typically not shown for regular responses
      const displayContent = responseData.content || '';
      
      // Update with complete content
      updateMessageContent(aiMessageId, displayContent, false, setMessages);
    }
    
    // Update session ID if needed
    if (responseData.session_id && !sessionId) {
      setSessionId(responseData.session_id);
    }
    
  } catch (e) {
    // Clear the timeout if there was an error
    clearTimeout(timeoutId);
    
    console.error("Error sending message:", e);
    
    // Specially handle abort errors from timeout
    if (e.name === 'AbortError') {
      updateMessageContent(
        aiMessageId, 
        "⏱️ **Request Timed Out**\n\nThe AI is taking longer than expected to respond (5+ minutes). This might happen with complex questions or when the AI is processing multiple requests.\n\n**Suggestions:**\n• Try asking a simpler or more specific question\n• Break complex questions into smaller parts\n• Try again in a few moments\n• Check if your question requires extensive research or calculation", 
        false, 
        setMessages
      );
    } else if (e.message.includes('504')) {
      updateMessageContent(
        aiMessageId, 
        "🔄 **Gateway Timeout**\n\nThe server took too long to process your request. This typically happens with very complex questions.\n\n**Please try:**\n• Asking a simpler version of your question\n• Breaking your question into smaller parts\n• Trying again in a moment", 
        false, 
        setMessages
      );
    } else {
      updateMessageContent(
        aiMessageId, 
        `Error: ${e.message || "Failed to connect to server"}`, 
        false, 
        setMessages
      );
    }
  } finally {
    setIsProcessing(false);
  }
};

    return (
        <div className="flex h-screen bg-gray-50">
            <ChatSidebar onSelectSession={handleSessionSelect} currentSessionId={sessionId} />
            
            <div className="flex-1 flex flex-col">
                {/* Header section */}
                <div className="bg-white shadow-sm">
                    <div className="max-w-7xl mx-auto px-4 py-4 flex justify-between items-center">
                        <div className="flex items-center">
                            <img src={echatLogo} alt="eChat" className="h-8 w-auto mr-2" />
                        </div>
                        <h1 className="text-xl font-semibold text-gray-900">eChat</h1>
                        <div className="flex items-center space-x-4">
                            {/* Debug admin status */}
                            {debugLog("Admin status in render:", isAdmin)}
                            
                            {/* Only show upload button for admin */}
                            {isAdmin && (
                                <div className="relative">
                                    {/* Knowledge Hub Button with Dropdown */}
                                    <div className="relative inline-block knowledge-hub-container">
                                        <button
                                            onClick={() => setShowKnowledgeHub(!showKnowledgeHub)}
                                            className="flex items-center px-4 py-2 bg-purple-50 text-purple-600 rounded-lg cursor-pointer hover:bg-purple-100 transition-colors duration-200"
                                        >
                                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                                            </svg>
                                            Knowledge Hub
                                            <svg className={`ml-2 h-4 w-4 transition-transform duration-200 ${showKnowledgeHub ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                                            </svg>
                                        </button>
                                        
                                        {/* Dropdown Menu */}
                                        {showKnowledgeHub && (
                                            <div className="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg z-50 border border-gray-200">
                                                <div className="py-1">
                                                    {/* Upload Files Option */}
                                                    <label className="flex items-center px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer">
                                                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                                        </svg>
                                                        Upload Files
                                                        <input
                                                            type="file"
                                                            accept=".xlsx,.csv,.docx,.pdf"
                                                            multiple
                                                            onChange={(e) => {
                                                                handleFileUpload(e, false);
                                                                setShowKnowledgeHub(false);
                                                            }}
                                                            className="hidden"
                                                        />
                                                    </label>
                                                    
                                                    {/* Upload Folder Option */}
                                                    <label className="flex items-center px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer">
                                                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                                                        </svg>
                                                        Upload Folder
                                                        <input
                                                            type="file"
                                                            webkitdirectory="true"
                                                            directory="true"
                                                            multiple
                                                            onChange={(e) => {
                                                                handleFileUpload(e, true);
                                                                setShowKnowledgeHub(false);
                                                            }}
                                                            className="hidden"
                                                        />
                                                    </label>
                                                    
                                                    {/* View Files Option */}
                                                    <button
                                                        onClick={() => {
                                                            setShowFileList(true);
                                                            setShowKnowledgeHub(false);
                                                        }}
                                                        className="flex items-center w-full px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer"
                                                    >
                                                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                                                        </svg>
                                                        View Knowledge Base
                                                    </button>
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                    
                                    {uploadStatus && (
                                        <div className={`absolute top-full left-0 right-0 mt-1 px-2 py-1 text-sm text-center rounded ${
                                            uploadStatus.isError ? 'bg-red-500' : 'bg-green-500'
                                        } text-white z-40`}>
                                            {uploadStatus.message}
                                        </div>
                                    )}
                                </div>
                            )}
                            
                            {/* Add User Management Button for Admin */}
                            {isAdmin ? (
                                <>
                                    <button
                                        onClick={() => setShowActivityDashboard(true)}
                                        className="flex items-center px-4 py-2 bg-indigo-50 text-indigo-600 rounded-lg hover:bg-indigo-100"
                                    >
                                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                                        </svg>
                                        All Users
                                    </button>
                                    <button
                                        onClick={() => {
                                            setShowUserManagement(!showUserManagement);
                                        }}
                                        className="flex items-center px-4 py-2 bg-indigo-50 text-indigo-600 rounded-lg hover:bg-indigo-100 transition-colors duration-200"
                                    >
                                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
                                        </svg>
                                        Manage Users
                                    </button>
                                    
                                    {/* Model Settings Button */}
                                    <button
                                        onClick={() => setShowModelSettings(true)}
                                        className="flex items-center px-4 py-2 bg-blue-50 text-blue-600 rounded-lg hover:bg-blue-100 transition-colors duration-200"
                                    >
                                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                                        </svg>
                                        Model Settings
                                    </button>
                                </>
                            ) : (
                                <div className="text-red-500 text-sm font-semibold ml-4">Admin access required for these features.</div>
                            )}
                            
                            <UserMenu />
                        </div>
                    </div>
                </div>

                {/* Main content area */}
                <div className="flex-1 flex overflow-hidden">
                    {/* Chat section */}
                    <div className={`flex-1 flex flex-col ${showUserManagement ? 'mr-80' : ''}`}>
                        {/* Messages area */}
                        <div className="flex-1 overflow-y-auto p-6">
                            <div className="space-y-4">
                                {messages.map((message) => (
                                    <React.Fragment key={message.id}>
                                        <div 
                                            className={`flex ${message.isUser ? 'justify-end' : 'justify-start'}`}
                                        >
                                            <div className={`max-w-[70%] p-4 rounded-2xl ${
                                                message.isUser 
                                                    ? 'bg-indigo-600 text-white' 
                                                    : message.error 
                                                        ? 'bg-red-100 text-red-800'
                                                        : 'bg-white shadow-md text-gray-800'
                                            }`}>
                                                <div className="break-words">
                                                    {message.isUser ? (
                                                        // For user messages, keep simple text rendering
                                                        <div className="whitespace-pre-wrap">
                                                            {String(message.content || '')}
                                                        </div>
                                                    ) : (
                                                        // For AI messages, use markdown rendering
                                                        typeof message.content === 'string' 
                                                            ? renderMessageContent(message)
                                                            : <MarkdownRenderer content={String(message.content || '')} />
                                                    )}
                                                </div>
                                                {message.processingTime && (
                                                    <div className="text-xs mt-2 opacity-75">
                                                        Response time: {message.processingTime}s
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                        
                                        {/* Display response options after AI messages if available */}
                                        {!message.isUser && message.id === optionsMessageId && showResponseOptions && responseOptions.length > 0 && (
                                            <ResponseOptions 
                                                options={responseOptions} 
                                                onSelect={handleResponseSelect}
                                            />
                                        )}
                                    </React.Fragment>
                                ))}
                            </div>
                            {isProcessing && (
                                <div className="flex justify-center mt-4">
                                    <LoadingSpinner />
                                </div>
                            )}
                        </div>

                        {/* Input form */}
                        <div className="bg-white border-t p-4">
                            <form onSubmit={handleSubmit} className="max-w-4xl mx-auto flex gap-4">
                                <input
                                    type="text"
                                    value={inputMessage}
                                    onChange={(e) => setInputMessage(e.target.value)}
                                    placeholder="Type your message..."
                                    className={`flex-1 p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent ${
                                        isProcessing ? 'opacity-50' : ''
                                    }`}
                                    disabled={isProcessing}
                                />
                                <button 
                                    type="submit"
                                    disabled={isProcessing}
                                    className={`px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 ${
                                        isProcessing ? 'opacity-50 cursor-not-allowed' : ''
                                    }`}
                                >
                                    {isProcessing ? <LoadingSpinner /> : 'Send'}
                                </button>
                            </form>
                        </div>
                    </div>

                    {/* User Management Panel */}
                    {/* User Management Panel (Admin: Create/List users) */}
                    {showUserManagement && isAdmin && (
                        <div className="fixed inset-y-0 right-0 w-96 bg-white shadow-lg p-6 overflow-y-auto z-40">
                            <div className="flex justify-between items-center mb-6">
                                <h2 className="text-xl font-bold">User Management</h2>
                                <button onClick={() => setShowUserManagement(false)} className="text-gray-500 hover:text-gray-700 text-2xl">×</button>
                            </div>
                            {/* Create New User Form */}
                            <form onSubmit={handleAddUser} className="mb-6">
                                <div className="mb-4">
                                    <label className="block text-sm font-medium mb-1">Username</label>
                                    <input
                                        type="text"
                                        value={newUser.username}
                                        onChange={e => setNewUser({ ...newUser, username: e.target.value })}
                                        className="w-full border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                                        autoComplete="off"
                                    />
                                </div>
                                <div className="mb-4">
                                    <label className="block text-sm font-medium mb-1">Password</label>
                                    <input
                                        type="password"
                                        value={newUser.password}
                                        onChange={e => setNewUser({ ...newUser, password: e.target.value })}
                                        className="w-full border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                                    />
                                </div>
                                <div className="mb-4">
                                    <label className="block text-sm font-medium mb-1">Role</label>
                                    <select
                                        value={selectedRole}
                                        onChange={e => setSelectedRole(e.target.value)}
                                        className="w-full border border-gray-300 rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                                    >
                                        {roles.map(role => (
                                            <option key={role} value={role}>{role}</option>
                                        ))}
                                    </select>
                                </div>
                                {userError && <div className="text-red-500 text-sm mb-2">{userError}</div>}
                                <button type="submit" className="w-full bg-indigo-600 text-white py-2 rounded hover:bg-indigo-700 font-semibold">Create User</button>
                            </form>
                            {/* List of Users */}
                            <div>
                                <h3 className="text-lg font-semibold mb-2">All Users</h3>
                                <ul className="divide-y divide-gray-200">
                                    {users.map(user => (
                                        <li key={user.username} className="py-2 flex items-center justify-between cursor-pointer hover:bg-indigo-50 px-2 rounded"
                                            onClick={() => {
                                                setShowUserManagement(false);
                                                setSelectedUsername(user.username);
                                            }}
                                        >
                                            <span className="font-medium">{user.username}</span>
                                            <span className="text-xs bg-gray-200 text-gray-700 rounded px-2 py-0.5 ml-2">{user.role || 'User'}</span>
                                        </li>
                                    ))}
                                    {users.length === 0 && <li className="text-gray-500 py-2">No users found</li>}
                                </ul>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Modals and Dashboards */}
            {showFileList && (
                <FileListDashboard onClose={() => setShowFileList(false)} />
            )}
            {showActivityDashboard && (
                <ActivityDashboard onClose={() => setShowActivityDashboard(false)} />
            )}
            {/* Show UserDashboard as modal only when a user is selected from the list */}
            {selectedUsername && (
                <UserDashboard username={selectedUsername} onClose={() => setSelectedUsername(null)} />
            )}
            {/* Model Settings Modal */}
            {showModelSettings && (
                <ModelSettings 
                    isOpen={showModelSettings}
                    onClose={() => setShowModelSettings(false)}
                    onSave={(settings) => {
                        console.log('Model settings saved:', settings);
                        // Optionally show a success notification
                        setNotification({
                            message: 'Model settings saved successfully!',
                            type: 'success'
                        });
                        setTimeout(() => setNotification(null), 3000);
                    }}
                />
            )}
        </div>
    );
};

export default Chat;
