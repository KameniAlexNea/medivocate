gdown --folder https://drive.google.com/drive/folders/1zZ741_LWxZwkCnMp-sO0UL6wEf8KFMln?usp=sharing
mv Books/ data/
python -m src.ocr.reader.reader_engine --pdf_path data/Books
python -m src.ocr.main --pdf_path data/Books
python -m src.chunking.text_cleaner --pdf_text_path data/Books
python -m src.chunking.chunk --input_folder data/Books/ --save_folder data/chunks --chunk_size 1500
python -m src.chunking.create_vector_store --docs_dir data/chunks
