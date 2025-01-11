# [medivocate](https://huggingface.co/spaces/alexneakameni/medivocate)

An AI-driven platform empowering users with trustworthy, personalized history guidance to combat misinformation and promote equitable history.

Deployed on [HF Space](https://huggingface.co/spaces/alexneakameni/medivocate)

```
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
```

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

## Some recommendations

### Groq

While Groq's request limit is noted, a brief explanation of what Groq offers in terms of LLM integration would help. For instance:
*“Groq provides access to high-performance LLM APIs with free-tier support for RAG applications. Ideal for quick prototyping and testing.”*

### LangChain

**LangChain** is a framework for developing applications powered by large language models (LLMs).

For these applications, LangChain simplifies the entire application lifecycle:

* **Open-source libraries** : Build your applications using LangChain's open-source [components](https://python.langchain.com/docs/concepts/) and [third-party integrations](https://python.langchain.com/docs/integrations/providers/). Use [LangGraph](https://langchain-ai.github.io/langgraph/) to build stateful agents with first-class streaming and human-in-the-loop support.
* **Productionization** : Inspect, monitor, and evaluate your apps with [LangSmith](https://docs.smith.langchain.com/) so that you can constantly optimize and deploy with confidence.
* **Deployment** : Turn your LangGraph applications into production-ready APIs and Assistants with [LangGraph Platform](https://langchain-ai.github.io/langgraph/cloud/).

### LangSmith

LangSmith helps your team debug, evaluate, and monitor your language models and intelligent agents. It works with any LLM Application, including a native integration with the [LangChain Python](https://github.com/langchain-ai/langchain) and [LangChain JS](https://github.com/langchain-ai/langchainjs) open source libraries.

LangSmith is developed and maintained by [LangChain](https://langchain.com/), the company behind the LangChain framework.

### Dataset

🤗 Datasets is a lightweight library providing **two** main features:

* **one-line dataloaders for many public datasets** : one-liners to download and pre-process any of the [![number of datasets](https://camo.githubusercontent.com/f72d44747c1f113f645de3943872ae3f58dc604b216d6a6e75c62cd4435c3456/68747470733a2f2f696d672e736869656c64732e696f2f656e64706f696e743f75726c3d68747470733a2f2f68756767696e67666163652e636f2f6170692f736869656c64732f646174617365747326636f6c6f723d627269676874677265656e)](https://camo.githubusercontent.com/f72d44747c1f113f645de3943872ae3f58dc604b216d6a6e75c62cd4435c3456/68747470733a2f2f696d672e736869656c64732e696f2f656e64706f696e743f75726c3d68747470733a2f2f68756767696e67666163652e636f2f6170692f736869656c64732f646174617365747326636f6c6f723d627269676874677265656e) major public datasets (image datasets, audio datasets, text datasets in 467 languages and dialects, etc.) provided on the [HuggingFace Datasets Hub](https://huggingface.co/datasets). With a simple command like `squad_dataset = load_dataset("squad")`, get any of these datasets ready to use in a dataloader for training/evaluating a ML model (Numpy/Pandas/PyTorch/TensorFlow/JAX),
* **efficient data pre-processing** : simple, fast and reproducible data pre-processing for the public datasets as well as your own local datasets in CSV, JSON, text, PNG, JPEG, WAV, MP3, Parquet, etc. With simple commands like `processed_dataset = dataset.map(process_example)`, efficiently prepare the dataset for inspection and ML model evaluation and training.

### Search Index : Hugging Face + Chroma, BM25
