# Groupme-Chatbot

This is chatbot that can respond to messages in your groupchat!

Uses gpt-neo-1.3B from https://huggingface.co/EleutherAI/gpt-neo-1.3B

To use, create a .env with your groupme access token (From dev.groupme.com), your user ID (for listening to messages),
the group ID to connect to, and the bot ID to post with.

In bot.py, you can specify if you want to use GPU "cuda" or CPU to run the model. The model needs at least 10 gigs of VRAM, with a 2080TI it takes about 3 seconds for a message.
CPU takes much longer.

