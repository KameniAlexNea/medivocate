


# Examples

This guide provides functional examples demonstrating the use of the main modules.

---

## Read and Export Documents

### **Option 1: Use Pre-Embedded Files (Recommended)**

If you have access to the pre-embedded documents stored in ChromaDB, you can skip all the steps below. Simply download the ready-to-use data:

* **Download Pre-Embedded Documents:**

  Access the pre-embedded documents from [this link](https://drive.google.com/drive/folders/1MJlkbD_omZ_nUt8EGtOD9fI17BZ8MNmR).

No further processing is needed if you use this option.

---

### **Option 2: Process PDF Documents Manually**

If the pre-embedded documents are unavailable, follow these steps:

1. **Download PDF Documents**

   Access the collection of documents from our [Google Drive](https://drive.google.com/drive/folders/1zZ741_LWxZwkCnMp-sO0UL6wEf8KFMln).
2. **Extract and Organize Files**

   Extract the downloaded documents into the `data/Books` directory. The directory should resemble the following structure:

   ```plaintext
   data/Books/
   ├── CAD Antériorité des Civilisations Nègres.pdf
   ├── COMMENT_ELABORER_UNE_THEORIE_AFRICAINE_Les_Fondements_de_la_science.pdf
   ├── Cultes de crane - Ouest Cameroun.pdf
   ├── Encyclopédie des Plantes Medicinales.pdf
   ├── ...
   └── Volume VIII - L’Afrique depuis 1935.pdf
   ```

---

### **Step-by-Step Processing**

1. **Read All Readable Documents**

   Use the reader engine to process all readable documents in the specified folder:

   ```bash
   python -m src.ocr.reader.reader_engine --pdf_path data/Books
   ```
2. **Process Non-Readable Documents Using OCR**

   Handle any non-readable documents with the OCR engine:

   ```bash
   python -m src.ocr.main --pdf_path data/Books
   ```
3. **Clean Document Pages**

   Clean and process all pages in the PDF documents. At this stage, approximately 12,954 pages will be processed:

   ```bash
   python -m src.chunking.text_cleaner --pdf_text_path data/Books
   ```
4. **Categorize and Filter Pages**

   Categorize pages and remove unnecessary content. Save the filtered chunks in the `data/chunks` folder:

   ```bash
   python -m src.chunking.chunk --input_folder data/Books/ --save_folder data/chunks
   ```
5. **Create the Chroma Database**

   Compute embeddings for all chunked documents (approximately 48,000 chunks) and store them in ChromaDB:

   ```bash
   python -m src.chunking.create_vector_store --docs_dir data/chunks
   ```

---

### Final Notes

* **Pre-Embedded vs. Manual Processing:**

  If you have the pre-embedded documents, skip all manual steps and directly use the provided files.
* **Dependencies:**

  Ensure the required Python modules are installed before executing the scripts.
* **Support:**

  For additional help, consult the relevant module documentation or contact the support team.

## All avalaible documents in this project

* ```
  data/Books/
  ├── CAD Antériorité des Civilisations Nègres.pdf
  ├── COMMENT_ELABORER_UNE_THEORIE_AFRICAINE_Les_Fondements_de_la_science.pdf
  ├── Cultes de crane - Ouest Cameroun.pdf
  ├── Encyclopédie des Plantes Medicinales.pdf
  ├── la bible sécrète des noirs.pdf
  ├── LA CRYPTO-COMMUNICATION AFRICAINE.pdf
  ├── LAfrique noire pré-coloniale (Cheikh Anta Diop) .pdf
  ├── La_naissance_du_panafricanisme_Les_racines_caraibes,_americaines(1).pdf
  ├── la-pharmacopee-des-plantes-medicinales-de-lafrique-de-louestok.pdf
  ├── La_Philosophie_Africaine_de_la_Période_Pharaonique_by_Théophile(1).pdf
  ├── La santé par les plantes.pdf
  ├── la tradition kemite.pdf
  ├── LE_DISCOURS_AFRICAIN_DE_LA_METHODE_L’algorithme_de_l’esprit,_la.pdf
  ├── Le Livre Des Morts Des Anciens egyptiens.pdf
  ├── Les_fondements_éco_Cheikh Anta Diop (1).pdf
  ├── Lorigine_négro_africaine_du_savoir_grec_Jean_Philippe_Omotunde@lechat.pdf
  ├── NATIONS NEGRES ET CULTURE.pdf
  ├── Philippe_Laburthe_Tolra_Initiations_et_Sociétés_secrètes_au_Cameroun.pdf
  ├── SPIRITUALITE KEMET II.pdf
  ├── THEORIE_DE_LA_CROISSANCE_ECONOMIQUE_Cosmos,_Complexité,_Valeur_et.pdf
  ├── THEORIE_GENERALE_DE_L_ETAT_EN_AFRIQUE_POUVOIR,_FEDERALISME_ET_DEMOCRATIE.pdf
  ├── Une_contre_histoire_de_la_colonisation_française_Driss_Ghali.pdf
  ├── Volume II - Afrique ancienne.pdf
  ├── Volume III - L’Afrique du VIIe au XIe siècle.pdf
  ├── Volume I - Méthodologie et préhistoire africaine.pdf
  ├── Volume IV - L’Afrique du XIIe au XVIe siècle.pdf
  ├── Volume VIII - L’Afrique depuis 1935.pdf
  ├── Volume VII - L’Afrique sous domination coloniale, 1880-1935.pdf
  ├── Volume VI - Le XIXe siècle jusque vers les années 1880.pdf
  └── Volume V - L’Afrique du XVIe au XVIIIe siècle.pdf

  1 directory, 30 files
  ```
