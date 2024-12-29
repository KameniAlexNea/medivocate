OPEN_QUESTION_PROMPT = """
You are a helpful assistant tasked with generating three open-ended, concise, and thoughtful questions based on a provided context. The questions should be clear enough to evaluate the understanding of the context and should not require the context to answer directly. Instead, they should encourage critical thinking, synthesis, or general knowledge about the topic. Provide accurate and complete answers for each question.  

Format your output in XML, where:  
- Each question is enclosed in a `<question>` tag.  
- The answer for each question is enclosed in an `<answer>` tag.  
- Each question-answer pair is wrapped in a `<qa>` tag.  
- The XML should be readable and indented properly for clarity.  

**Guidelines for Questions**:  
1. Avoid highly specific or overly detailed questions that require the exact text of the context to answer.  
2. Focus on broader themes, implications, or general knowledge derived from the context.  
3. Ensure questions are meaningful and can be answered independently of the exact wording of the context.  

**Input:**  
Context: {context}  

**Expected Output:**  
Generate three `<qa>` elements with corresponding `<question>` and `<answer>` tags. Structure the XML like this:  
```xml
<qas>
    <qa>
        <question>[First open-ended question based on the context]</question>
        <answer>[Answer to the first question]</answer>
    </qa>
    <qa>
        <question>[Second open-ended question based on the context]</question>
        <answer>[Answer to the second question]</answer>
    </qa>
    <qa>
        <question>[Third open-ended question based on the context]</question>
        <answer>[Answer to the third question]</answer>
    </qa>
</qas>
```

**Example Input:**  
Context:  
"The empires of Mali and Songhai were among the most powerful in West Africa during the Middle Ages. These empires thrived on trade, particularly in gold and salt, and were centers of culture and learning, as exemplified by the city of Timbuktu. Notable rulers included Mansa Musa, whose pilgrimage to Mecca in the 14th century displayed the immense wealth of Mali, and Askia Muhammad of Songhai, who reformed governance and strengthened Islam in the region."  

**Example Output:**  
```xml
<qas>
    <qa>
        <question>What were the primary factors that contributed to the success of medieval West African empires like Mali and Songhai?</question>
        <answer>The success of these empires was driven by control of trade routes, particularly for gold and salt, cultural and educational advancements, and strong leadership such as Mansa Musa and Askia Muhammad.</answer>
    </qa>
    <qa>
        <question>How did rulers like Mansa Musa and Askia Muhammad influence the cultural and religious landscape of their empires?</question>
        <answer>Mansa Musa promoted Islam through his famous pilgrimage to Mecca, demonstrating Mali's wealth, while Askia Muhammad reformed governance and strengthened Islamic practices in Songhai.</answer>
    </qa>
    <qa>
        <question>What role did cities like Timbuktu play in the development of medieval West African empires?</question>
        <answer>Timbuktu was a center of trade, education, and Islamic culture, housing renowned universities and attracting scholars and merchants from across the world.</answer>
    </qa>
</qas>
```
"""

QUIZZ_QUESTION_PROMPT = """

"""
