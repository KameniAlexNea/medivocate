# Examples

Contient des exemples fonctionnels démontrant l'utilisation des principaux modules.

## Read and Export Documents

* Download PDF document from our [drive](https://drive.google.com/drive/folders/1zZ741_LWxZwkCnMp-sO0UL6wEf8KFMln) :
* Extract and add it to data/Books

  ```
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
* Read all document readable from the corresponded folder
  `python -m src.ocr.reader.reader_engine --pdf_path data/Books`
* Add non readable documents with our OCR engine
  `python -m src.ocr.main --pdf_path data/Books`
