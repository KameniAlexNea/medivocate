OPEN_QUESTION_PROMPT = """
Vous êtes un assistant utile chargé de générer trois questions ouvertes, concises et réfléchies basées sur un contexte fourni. Les questions doivent être suffisamment claires pour évaluer la compréhension du contexte, sans nécessiter directement le contexte pour y répondre. Elles doivent encourager la pensée critique, la synthèse ou les connaissances générales sur le sujet. Fournissez des réponses précises et complètes pour chaque question.  

Formatez votre sortie en XML, où :  
- Chaque question est entourée d'une balise `<question>`.  
- La réponse à chaque question est entourée d'une balise `<answer>`.  
- Chaque paire question-réponse est encapsulée dans une balise `<qa>`.  
- Le XML doit être lisible et correctement indenté pour plus de clarté.  

**Directives pour les questions** :  
1. Évitez les questions très spécifiques ou trop détaillées qui nécessitent le texte exact du contexte pour y répondre.  
2. Concentrez-vous sur des thèmes plus larges, des implications ou des connaissances générales dérivées du contexte.  
3. Assurez-vous que les questions sont significatives et peuvent être répondues indépendamment du libellé exact du contexte.  

**Entrée :**
Contexte :
{context}  

**Sortie attendue :**  
Générez trois éléments `<qa>` avec des balises `<question>` et `<answer>` correspondantes. Structurez le XML comme suit :  
```xml
<qas>
    <qa>
        <question>[Première question ouverte basée sur le contexte]</question>
        <answer>[Réponse à la première question]</answer>
    </qa>
    <qa>
        <question>[Deuxième question ouverte basée sur le contexte]</question>
        <answer>[Réponse à la deuxième question]</answer>
    </qa>
    <qa>
        <question>[Troisième question ouverte basée sur le contexte]</question>
        <answer>[Réponse à la troisième question]</answer>
    </qa>
</qas>
```  

**Exemple d'entrée :**  
Contexte :  
"Les empires du Mali et du Songhaï étaient parmi les plus puissants d'Afrique de l'Ouest au Moyen Âge. Ces empires prospéraient grâce au commerce, en particulier de l'or et du sel, et étaient des centres de culture et de savoir, comme l'illustre la ville de Tombouctou. Parmi leurs dirigeants notables figuraient Mansa Musa, dont le pèlerinage à La Mecque au XIVe siècle a démontré l'immense richesse du Mali, et Askia Muhammad du Songhaï, qui a réformé la gouvernance et renforcé l'Islam dans la région."  

**Exemple de sortie :**  
```xml
<qas>
    <qa>
        <question>Quels étaient les principaux facteurs qui ont contribué au succès des empires médiévaux d'Afrique de l'Ouest comme le Mali et le Songhaï ?</question>
        <answer>Le succès de ces empires reposait sur le contrôle des routes commerciales, en particulier pour l'or et le sel, les avancées culturelles et éducatives, et le leadership fort de dirigeants comme Mansa Musa et Askia Muhammad.</answer>
    </qa>
    <qa>
        <question>Comment des dirigeants comme Mansa Musa et Askia Muhammad ont-ils influencé le paysage culturel et religieux de leurs empires ?</question>
        <answer>Mansa Musa a promu l'Islam à travers son célèbre pèlerinage à La Mecque, montrant la richesse du Mali, tandis qu'Askia Muhammad a réformé la gouvernance et renforcé les pratiques islamiques dans le Songhaï.</answer>
    </qa>
    <qa>
        <question>Quel rôle des villes comme Tombouctou ont-elles joué dans le développement des empires médiévaux d'Afrique de l'Ouest ?</question>
        <answer>Tombouctou était un centre de commerce, d'éducation et de culture islamique, abritant des universités renommées et attirant des érudits et des marchands du monde entier.</answer>
    </qa>
</qas>
``` 
"""

OPEN_QUESTION_PROMPT_EN = """
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
