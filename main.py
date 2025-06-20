from flask import Flask, render_template, request, jsonify, session
import openai
import os
from datetime import datetime
import uuid

import json
# This is a test comment

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # Change this in production

# Configure Gemini (using OpenAI-compatible client)
api_key = os.getenv('GEMINI_API_KEY') or os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY or OPENAI_API_KEY environment variable must be set")

chatbot = ChatBot()

class ChatBot:
    def __init__(self):
        self.model = "gemini-2.5-flash"  # Gemini model
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        
    def get_response(self, message, conversation_history=None, found_recipes=None):
        """Get response from Gemini API"""
        try:
            # Prepare messages for the API
            system_prompt = ""
            if found_recipes:
                # Format recipes into a string
                recipe_details_list = []
                for recipe in found_recipes:
                    ingredients = recipe.get('ingredients', [])
                    if isinstance(ingredients, (list, tuple)):
                        ingredients_str = ", ".join(ingredients)
                    elif isinstance(ingredients, str):
                        # Attempt to parse if it's a JSON string list/dict, then join
                        try:
                            import json # Make sure json is imported if not already at top level
                            parsed_ingredients = json.loads(ingredients)
                            if isinstance(parsed_ingredients, list):
                                ingredients_str = ", ".join(parsed_ingredients)
                            elif isinstance(parsed_ingredients, dict):
                                ingredients_str = ", ".join([f"{k}: {v}" for k,v in parsed_ingredients.items()])
                            else:
                                ingredients_str = str(parsed_ingredients) # Fallback
                        except (ImportError, NameError, json.JSONDecodeError): # Added ImportError/NameError for safety
                            ingredients_str = ingredients # Keep as is if not valid JSON or json not available
                    else:
                        ingredients_str = str(ingredients)

                    recipe_text = (
                        f"Name: {recipe.get('name', 'N/A')}\n"
                        f"Description: {recipe.get('description', 'N/A')}\n"
                        f"Ingredients: {ingredients_str}\n"
                        f"Instructions: {recipe.get('instructions', 'N/A')}"
                    )
                    recipe_details_list.append(recipe_text)

                recipes_string = "\n\n---\n\n".join(recipe_details_list)

                system_prompt = (
                    "You are Chef Remy, a passionate and knowledgeable French chef. "
                    "The system has found the following recipes based on the user's input. "
                    "Your task is to present these recipes to the user in your charming and helpful style. "
                    "You can summarize them, comment on them, highlight interesting parts, and then list the full details. "
                    "Be enthusiastic and maintain your French persona (using words like 'mon ami', 'magnifique', 'voilà').\n\n"
                    "Here are the recipes:\n"
                    f"{recipes_string}"
                )
            else:
                system_prompt = (
                    "You are Chef Remy, a passionate and knowledgeable French chef who loves helping people discover amazing recipes. "
                    "You're enthusiastic, friendly, and speak with a slight French accent in your writing (using words like 'mon ami', 'magnifique', 'oui'). "
                    "You help users decide on recipes by asking about their preferences, dietary restrictions, available ingredients, cooking skill level, and time constraints. "
                    "You provide detailed recipe suggestions with cooking tips and encourage culinary creativity. "
                    "Always stay in character as the helpful chef Remy!"
                )

            messages = [{"role": "system", "content": system_prompt}]
            
            # Add conversation history if provided
            if conversation_history:
                messages.extend(conversation_history)
            
            # Add the current user message
            messages.append({"role": "user", "content": message})
            
            # Call Gemini API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500,
                temperature=0.7,
                stream=False
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error calling Gemini API: {str(e)}")
            return "Sorry, I'm having trouble processing your request right now."

# Initialize chatbot

@app.route('/')
def home():
    """Main chat interface"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        session['conversation'] = []
    
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Get conversation history from session
        conversation_history = session.get('conversation', [])
        
        # Get bot response
        conversation_history.append({"role": "user", "content": user_message})
        all_user_input = " ".join([msg['content'] for msg in conversation_history if msg['role'] == 'user'])
        recipes = find_similar_recipes(query=all_user_input, top_n=3)
        
        # Update conversation history
        # bot_response = format_recipes_response(recipes) # Comment out the old line
        bot_response = chatbot.get_response(message=user_message, conversation_history=conversation_history, found_recipes=recipes)
        conversation_history.append({"role": "assistant", "content": bot_response})
        
        # Keep only last 10 exchanges to manage token usage
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]
        
        session['conversation'] = conversation_history
        
        return jsonify({
            'response': bot_response,
            'timestamp': datetime.now().strftime('%H:%M')
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': 'An error occurred processing your message'}), 500

@app.route('/clear', methods=['POST'])
def clear_conversation():
    """Clear conversation history"""
    session['conversation'] = []
    return jsonify({'status': 'cleared'})

@app.route('/history')
def get_history():
    """Get conversation history"""
    conversation = session.get('conversation', [])
    return jsonify({'conversation': conversation})

# HTML Template (save this as templates/chat.html)
chat_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Chatbot</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #ff6b6b 0%, #ffa500 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .chat-container {
            width: 800px;
            height: 600px;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .chat-header {
            background: linear-gradient(135deg, #ff6b6b 0%, #ffa500 100%);
            color: white;
            padding: 20px;
            text-align: center;
            position: relative;
        }

        .chat-header h1 {
            font-size: 24px;
            font-weight: 300;
        }

        .clear-btn {
            position: absolute;
            right: 20px;
            top: 50%;
            transform: translateY(-50%);
            background: rgba(255,255,255,0.2);
            border: none;
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            cursor: pointer;
            transition: background 0.3s;
        }

        .clear-btn:hover {
            background: rgba(255,255,255,0.3);
        }

        .chat-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: #f8f9fa;
        }

        .message {
            margin-bottom: 15px;
            display: flex;
            align-items: flex-start;
        }

        .message.user {
            justify-content: flex-end;
        }

        .message-content {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 20px;
            position: relative;
        }

        .message.user .message-content {
            background: linear-gradient(135deg, #ff6b6b 0%, #ffa500 100%);
            color: white;
            border-bottom-right-radius: 5px;
        }

        .message.bot .message-content {
            background: white;
            color: #333;
            border: 1px solid #e1e8ed;
            border-bottom-left-radius: 5px;
        }

        .message-time {
            font-size: 12px;
            opacity: 0.7;
            margin-top: 5px;
        }

        .chat-input {
            padding: 20px;
            border-top: 1px solid #e1e8ed;
            background: white;
        }

        .input-group {
            display: flex;
            gap: 10px;
        }

        .message-input {
            flex: 1;
            padding: 12px 16px;
            border: 1px solid #e1e8ed;
            border-radius: 25px;
            outline: none;
            font-size: 14px;
        }

        .message-input:focus {
            border-color: #ff6b6b;
        }

        .send-btn {
            background: linear-gradient(135deg, #ff6b6b 0%, #ffa500 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 14px;
            transition: transform 0.2s;
        }

        .send-btn:hover {
            transform: translateY(-2px);
        }

        .send-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .typing-indicator {
            display: none;
            align-items: center;
            margin-bottom: 15px;
        }

        .typing-indicator .message-content {
            background: white;
            border: 1px solid #e1e8ed;
            padding: 12px 16px;
        }

        .typing-dots {
            display: flex;
            gap: 4px;
        }

        .typing-dots span {
            width: 8px;
            height: 8px;
            background: #ff6b6b;
            border-radius: 50%;
            animation: typing 1.4s infinite;
        }

        .typing-dots span:nth-child(2) {
            animation-delay: 0.2s;
        }

        .typing-dots span:nth-child(3) {
            animation-delay: 0.4s;
        }

        @keyframes typing {
            0%, 60%, 100% {
                transform: translateY(0);
            }
            30% {
                transform: translateY(-10px);
            }
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>👨‍🍳 Chef Remy's Kitchen</h1>
            <button class="clear-btn" onclick="clearConversation()">New Recipe</button>
        </div>
        
        <div class="chat-messages" id="chatMessages">
            <div class="message bot">
                <div class="message-content">
                    Bonjour, mon ami! 👨‍🍳 I am Chef Remy, and I'm absolutely delighted to help you discover the perfect recipe today! Whether you're looking for something quick and simple or a magnificent culinary adventure, I'm here to guide you. 
                    <br><br>
                    Tell me, what are you in the mood for? Perhaps something savory or sweet? And do you have any ingredients you'd like to use, or any dietary preferences I should know about?
                    <div class="message-time" id="currentTime"></div>
                </div>
            </div>
        </div>
        
        <div class="typing-indicator" id="typingIndicator">
            <div class="message-content">
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        </div>
        
        <div class="chat-input">
            <div class="input-group">
                <input type="text" id="messageInput" class="message-input" 
                       placeholder="Tell Chef Remy what you'd like to cook..." onkeypress="handleKeyPress(event)">
                <button id="sendBtn" class="send-btn" onclick="sendMessage()">Send</button>
            </div>
        </div>
    </div>

    <script>
        // Set current time for initial message
        document.getElementById('currentTime').textContent = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }

        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            // Add user message to chat
            addMessage(message, 'user');
            input.value = '';
            
            // Disable send button and show typing indicator
            const sendBtn = document.getElementById('sendBtn');
            sendBtn.disabled = true;
            showTypingIndicator();
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    addMessage(data.response, 'bot', data.timestamp);
                } else {
                    addMessage('Sorry, there was an error processing your message.', 'bot');
                }
            } catch (error) {
                addMessage('Sorry, there was a connection error.', 'bot');
            }
            
            // Hide typing indicator and re-enable send button
            hideTypingIndicator();
            sendBtn.disabled = false;
        }

        function addMessage(content, sender, timestamp = null) {
            const messagesContainer = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;
            
            const time = timestamp || new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
            
            messageDiv.innerHTML = `
                <div class="message-content">
                    ${content}
                    <div class="message-time">${time}</div>
                </div>
            `;
            
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function showTypingIndicator() {
            document.getElementById('typingIndicator').style.display = 'flex';
            const messagesContainer = document.getElementById('chatMessages');
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function hideTypingIndicator() {
            document.getElementById('typingIndicator').style.display = 'none';
        }

        async function clearConversation() {
            try {
                await fetch('/clear', { method: 'POST' });
                const messagesContainer = document.getElementById('chatMessages');
                messagesContainer.innerHTML = `
                    <div class="message bot">
                        <div class="message-content">
                            Bonjour, mon ami! 👨‍🍳 I am Chef Remy, and I'm absolutely delighted to help you discover the perfect recipe today! Whether you're looking for something quick and simple or a magnificent culinary adventure, I'm here to guide you. 
                            <br><br>
                            Tell me, what are you in the mood for? Perhaps something savory or sweet? And do you have any ingredients you'd like to use, or any dietary preferences I should know about?
                            <div class="message-time">${new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</div>
                        </div>
                    </div>
                `;
            } catch (error) {
                console.error('Error clearing conversation:', error);
            }
        }
    </script>
</body>
</html>
'''

# Create templates directory and save HTML file
import os
if not os.path.exists('templates'):
    os.makedirs('templates')

with open('templates/chat.html', 'w') as f:
    f.write(chat_html)

if __name__ == '__main__':   
    app.run(debug=True)