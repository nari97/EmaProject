def return_rephrase_prompt():
    prompt = """
    [INST] 
    Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
    If the question is the first question in the chat history, you must return it as is. If the question is irrelevant to the chat history, you must return it as is.
    Let me share a couple examples.
    ```
    Chat History:
    
    Question: How is Lawrence doing?
    
    Standalone Question:
    How is Lawrence doing?
    ```
    Another example:
    ```
    Chat History:
    
    Question: Can you tell me about breakthroughs in science?
    
    Standalone Question:
    Can you tell me about breakthroughs in science?
    ```
    If this is the second question onwards, you should properly rephrase the question like this:
    ```
    Chat History:
    Human: How is Lawrence doing?
    AI: 
    Lawrence is injured and out for the season.
    
    Question: What was his injury?
    
    Standalone Question:
    What was Lawrence's injury?
    ```
    If this is the second question onwards, and it is irrelevant to the chat history, you should return it as is like this:
    ```
    Chat History:
    Human: How is Lawrence doing?
    AI: 
    Lawrence is injured and out for the season.
    
    Question: What is tesla?
    
    Standalone Question:
    What is tesla?
    ```
    Now, with those examples, here is the actual chat history and input question.
    Chat History:
    {chat_history}
    
    Question: {question}
    
    Standalone question:
    [/INST]
    """
    return prompt
