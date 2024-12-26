## Implémenter un OCR pour convertir les documents non lisibles en format texte

**[Niveau 1] Implémenter un module OCR pour les documents non lisibles**

**Description :**

- Créer un module qui analyse des documents PDF ou images non machine-readable.
- Convertir ces documents en fichiers texte séparés (un fichier `.txt` par document).
- Explorer et sélectionner une bibliothèque OCR open-source (Tesseract, EasyOCR, etc.).
- Tester la solution sur un ensemble d'exemples variés pour garantir la qualité des sorties OCR.
- Démonstration attendue : Une interface ou un script qui prend des documents en entrée et génère les fichiers `.txt`.



## Structurer les documents pour un accès optimal aux informations

**[Niveau 3] Proposer une structuration des documents en chunks utiles pour le RAG**

**Description :**

- Concevoir une méthode pour diviser les documents en parties exploitables : pages, paragraphes, ou chunks basés sur des informations pertinentes.
- Explorer l’utilisation de LLMs pour identifier et extraire des informations utiles (contexte, maladies, traitements).
- Stocker chaque chunk avec des métadonnées claires pour faciliter les recherches (titre, source, lien avec une maladie, etc.).
- Démonstration attendue : Un prototype qui structure un document et montre comment ces chunks peuvent être recherchés efficacement.



## Développer un environnement de prompt engineering

**[Niveau 2] Développer un environnement pour améliorer les requêtes utilisateur**

**Description :**

- Concevoir une interface ou un pipeline qui reformule les requêtes utilisateur pour optimiser les recherches dans la base de données.
- Mettre en œuvre un calcul d’embedding et une mesure de similarité pour améliorer la précision des réponses.
- Explorer les outils comme LangChain pour faciliter cette tâche.
- Démonstration attendue : Une interface où l'utilisateur pose une question, et le système affiche la requête améliorée avec une réponse plus précise.

## Initialiser le projet et proposer les outils nécessaires

**[Niveau 2] Initialiser le projet et choisir les outils requis**

**Description :**

- Configurer l’environnement de développement avec les outils essentiels : frameworks (LangChain, Ollama, HuggingChat), bibliothèques NLP, etc.
- Identifier des solutions gratuites ou open-source pour faciliter le développement.
- Créer une documentation initiale listant les dépendances, outils, et un guide de démarrage rapide pour les contributeurs.
- Démonstration attendue : Une structure de projet fonctionnelle avec une première documentation décrivant comment exécuter un exemple de base.

## Concevoir un pipeline de recherche pour le RAG

**[Niveau 2] Développer un pipeline de recherche pour retrouver les informations pertinentes**

**Description :**

- Créer un pipeline qui interroge la base de données ou le vecteur store pour retrouver les chunks pertinents en fonction des requêtes utilisateur.
- Implémenter un mécanisme de recherche basée sur les embeddings (ex. Sentence Transformers, OpenAI embeddings).
- Tester avec différents algorithmes de similarité (cosine similarity, FAISS, etc.).
- Démonstration attendue : Une requête utilisateur renvoie les chunks les plus pertinents avec un score de similarité.

## Implémenter une base de données vectorielle pour le stockage des embeddings

**[Niveau 2] Mettre en place une base de données vectorielle pour le RAG**

**Description :**

- Explorer et configurer une base de données vectorielle adaptée, comme Pinecone, Weaviate, Milvus ou FAISS.
- Créer un pipeline pour indexer les chunks extraits des documents avec leurs embeddings.
- Vérifier que les performances de recherche restent optimales même avec un volume important de données.
- Démonstration attendue : Une base indexée et interrogeable avec des résultats pertinents en temps réel.

## Intégrer un modèle LLM pour la génération de réponses

**[Niveau 2] Intégrer un LLM pour répondre aux questions utilisateur**

**Description :**

- Intégrer un modèle LLM capable de synthétiser les informations des chunks retournés (ex. GPT-4, LLaMA, ou autre modèle open-source).
- Configurer le modèle pour qu’il inclue des références explicites aux sources des réponses.
- Tester la capacité du système à générer des réponses compréhensibles et contextuellement correctes.
- Démonstration attendue : Une question utilisateur renvoie une réponse générée par le LLM, appuyée par des sources fiables.

## Concevoir un pipeline de prétraitement des documents

**[Niveau 1] Développer un pipeline de prétraitement pour standardiser les documents**

**Description :**

- Mettre en place un pipeline pour nettoyer et normaliser les documents (suppression des caractères spéciaux, unification des formats).
- Gérer des cas spécifiques, comme les tableaux, les listes, ou les figures présents dans les documents.
- Fournir un mécanisme pour gérer les langues si les documents ne sont pas uniquement en français.
- Démonstration attendue : Un fichier brut est transformé en texte propre et structuré prêt à être ingéré dans le système RAG.

## Évaluer les outils de visualisation pour le suivi des performances

**[Niveau 1] Mettre en place un tableau de bord pour suivre les performances du système**

**Description :**

- Identifier un outil pour visualiser les métriques clés : latence de réponse, précision des réponses, taux de similarité, etc.
- Intégrer des graphiques pour le suivi des performances en temps réel.
- Fournir une interface permettant d’explorer les logs ou de tester les requêtes directement depuis le tableau de bord.
- Démonstration attendue : Un tableau de bord accessible avec des visualisations pertinentes des performances du RAG.