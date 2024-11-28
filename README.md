### RAG based application to extract specified transactions from bank statements.  
Models used 
1. GPT-4o mini
2. OpenAI o1-mini

***An OpenAI API Key is required to use this application***

---

### **Challenges faced**  
Depending on the way the bank statement is structured, different models may face different problems parsing the information on them.
While testing, 
- GPT-4o mini was found to be better in general, but struggled if there were multiple transactions having the same date, description, and transaction amount.  
- OpenAI o1-mini on the other hand was better at handling these kinds of multiple identical records.

In some rare cases, where each model tends to miss records that the other model is able to capture and vice versa, it may be beneficial to combine the outputs of both models and combine them, dropping the rows that got double counted.

Hence, users can select either of the models available, or choose to use all of them.

---

### **Future improvements**
- Find a way to improve the accuracy of the LLM
- Add other models that may be more accurate / local LLMs that won't require an API key to run (e.g. ollama)
