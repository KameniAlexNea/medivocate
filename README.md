# medivocate

An AI-driven platform empowering patients with trustworthy, personalized medical guidance to combat misinformation and promote equitable healthcare.

📦 ./
├── 📁 docs/
├── 📁 src/
│   ├── 📁 ocr/
│   ├── 📁 preprocessing/
│   ├── 📁 chunking/
│   ├── 📁 vector_store/
│   ├── 📁 rag_pipeline/
│   ├── 📁 llm_integration/
│   └── 📁 prompt_engineering/
├── 📁 tests/
│   ├── 📁 unit/
│   └── 📁 integration/
├── 📁 examples/
├── 📁 notebooks/
├── 📁 config/
├── 📄 README.md
├── 📄 CONTRIBUTING.md
├── 📄 requirements.txt
├── 📄 .gitignore
└── 📄 LICENSE

## Description des Dossiers et Fichiers

1. **`docs/`**

   * **Contient la documentation générale du projet.**
   * **Exemple : Guide de démarrage rapide, architecture du projet, et spécifications techniques.**
2. **`src/`**

   * **Dossier principal contenant le code source organisé par modules.**
   * **Sous-dossiers :**
     * `ocr/` : Module pour l'extraction de texte à partir de documents.
     * `preprocessing/` : Pipelines de nettoyage et de standardisation des documents.
     * `chunking/` : Méthodes pour diviser les documents en chunks exploitables.
     * `vector_store/` : Intégration de bases de données vectorielles.
     * `rag_pipeline/` : Implémentation du pipeline RAG (Retrieval-Augmented Generation).
     * `llm_integration/` : Gestion des modèles LLM pour la génération de réponses.
     * `prompt_engineering/` : Modules pour reformuler et optimiser les requêtes.
3. **`tests/`**

   * `unit/` : Tests unitaires pour chaque module.
   * `integration/` : Tests d’intégration entre plusieurs modules.
4. **`examples/`**

   * **Contient des exemples fonctionnels démontrant l'utilisation des principaux modules.**
5. **`notebooks/`**

   * **Jupyter notebooks pour des expérimentations ou des démonstrations rapides.**
6. **`config/`**

   * **Fichiers de configuration pour les bibliothèques, les pipelines, ou les environnements.**
