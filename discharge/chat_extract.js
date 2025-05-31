// AI Chat Extractor
// A bookmarklet to extract chat content from AI assistants and save as JSON
// Usage: paste the code below into a bookmark URL field.

javascript:(function() {
  // Create a style for our UI elements
  const style = document.createElement('style');
  style.textContent = `
    .ai-chat-extractor-overlay {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.7);
      z-index: 10000;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      color: white;
    }
    .ai-chat-extractor-card {
      background: #1a1a1a;
      border-radius: 10px;
      padding: 20px;
      width: 80%;
      max-width: 600px;
      box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
    }
    .ai-chat-extractor-header {
      font-size: 18px;
      font-weight: bold;
      margin-bottom: 15px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .ai-chat-extractor-close {
      cursor: pointer;
      font-size: 20px;
    }
    .ai-chat-extractor-button {
      background: #5436DA;
      color: white;
      border: none;
      padding: 10px 15px;
      border-radius: 5px;
      font-size: 14px;
      cursor: pointer;
      margin-top: 10px;
      margin-right: 10px;
    }
    .ai-chat-extractor-button:hover {
      background: #4520c9;
    }
    .ai-chat-extractor-textarea {
      width: 100%;
      height: 200px;
      background: #2d2d2d;
      color: #eee;
      border: 1px solid #444;
      border-radius: 5px;
      padding: 10px;
      font-family: monospace;
      margin-top: 10px;
    }
    .ai-chat-extractor-status {
      margin-top: 15px;
      color: #aaa;
      font-size: 14px;
    }
  `;
  document.head.appendChild(style);

  // Extract messages
  function extractMessages() {
    const messages = [];
    let role = 'user'; // Start with user as default

    // Try to find message blocks based on various AI chat UI patterns
    const messageBlocks = document.querySelectorAll('.message-content, .chat-message, .chat-turn');

    if (messageBlocks.length > 0) {
      // Found structured message blocks
      messageBlocks.forEach(block => {
        // Determine role
        const isHuman = block.classList.contains('human') ||
                       block.textContent.includes('Human:') ||
                       block.classList.contains('user-message') ||
                       block.classList.contains('user');

        role = isHuman ? 'user' : 'assistant';

        // Extract content
        let content = block.textContent.trim();
        // Remove role prefixes if present
        content = content.replace(/^(Human:|User:|Assistant:|AI:|ChatGPT:|Claude:|Bard:)\s*/i, '');

        messages.push({ role, content });
      });
    } else {
      // Fallback: look for elements that might contain messages
      const elements = document.querySelectorAll('p, div');
      let currentContent = '';

      elements.forEach(element => {
        const text = element.textContent.trim();

        // Check for role indicators
        if (text.startsWith('Human:') || text.startsWith('You:') || text.startsWith('User:')) {
          // Save previous message if it exists
          if (currentContent) {
            messages.push({ role, content: currentContent.trim() });
            currentContent = '';
          }
          role = 'user';
          currentContent = text.replace(/^(Human:|You:|User:)\s*/i, '');
        } else if (text.startsWith('Assistant:') || text.startsWith('AI:') || 
                  text.startsWith('Claude:') || text.startsWith('ChatGPT:') || 
                  text.startsWith('Bard:') || text.startsWith('Gemini:')) {
          // Save previous message if it exists
          if (currentContent) {
            messages.push({ role, content: currentContent.trim() });
            currentContent = '';
          }
          role = 'assistant';
          currentContent = text.replace(/^(Assistant:|AI:|Claude:|ChatGPT:|Bard:|Gemini:)\s*/i, '');
        } else if (text.length > 0) {
          // Continue current message
          currentContent += '\n' + text;
        }
      });

      // Add final message
      if (currentContent) {
        messages.push({ role, content: currentContent.trim() });
      }
    }

    return messages;
  }

  // Create JSON from messages
  function createJSON(messages) {
    return JSON.stringify({
      conversation_id: "extracted_" + Date.now(),
      title: document.title || "Extracted AI Conversation",
      messages: messages
    }, null, 2);
  }

  // Create and show UI
  function showExtractorUI() {
    // Create overlay
    const overlay = document.createElement('div');
    overlay.className = 'ai-chat-extractor-overlay';

    // Create card
    const card = document.createElement('div');
    card.className = 'ai-chat-extractor-card';

    // Create header
    const header = document.createElement('div');
    header.className = 'ai-chat-extractor-header';
    header.innerHTML = '<span>AI Chat Extractor</span><span class="ai-chat-extractor-close">×</span>';

    // Create description
    const description = document.createElement('div');
    description.textContent = 'Extract this conversation as JSON for use with context analysis tools';

    // Create extract button
    const extractButton = document.createElement('button');
    extractButton.className = 'ai-chat-extractor-button';
    extractButton.textContent = 'Extract Conversation';

    // Create copy button
    const copyButton = document.createElement('button');
    copyButton.className = 'ai-chat-extractor-button';
    copyButton.textContent = 'Copy to Clipboard';
    copyButton.style.display = 'none'; // Hide initially

    // Create download button
    const downloadButton = document.createElement('button');
    downloadButton.className = 'ai-chat-extractor-button';
    downloadButton.textContent = 'Download JSON';
    downloadButton.style.display = 'none'; // Hide initially

    // Create textarea
    const textarea = document.createElement('textarea');
    textarea.className = 'ai-chat-extractor-textarea';
    textarea.placeholder = 'Extracted JSON will appear here...';
    textarea.readOnly = true;

    // Create status
    const status = document.createElement('div');
    status.className = 'ai-chat-extractor-status';

    // Add event handlers
    header.querySelector('.ai-chat-extractor-close').addEventListener('click', () => {
      document.body.removeChild(overlay);
    });

    extractButton.addEventListener('click', () => {
      try {
        status.textContent = 'Extracting conversation...';
        const messages = extractMessages();
        const json = createJSON(messages);

        textarea.value = json;

        status.textContent = `Successfully extracted ${messages.length} messages!`;
        copyButton.style.display = 'inline-block';
        downloadButton.style.display = 'inline-block';
      } catch (error) {
        status.textContent = `Error: ${error.message}`;
        console.error(error);
      }
    });

    copyButton.addEventListener('click', () => {
      textarea.select();
      document.execCommand('copy');
      status.textContent = 'Copied to clipboard!';
    });

    downloadButton.addEventListener('click', () => {
      try {
        const blob = new Blob([textarea.value], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'ai_conversation.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        status.textContent = 'Downloaded JSON file!';
      } catch (error) {
        status.textContent = `Error downloading: ${error.message}`;
      }
    });

    // Assemble UI
    card.appendChild(header);
    card.appendChild(description);
    card.appendChild(extractButton);
    card.appendChild(copyButton);
    card.appendChild(downloadButton);
    card.appendChild(textarea);
    card.appendChild(status);
    overlay.appendChild(card);

    document.body.appendChild(overlay);
  }

  // Run the extractor
  showExtractorUI();
})();
