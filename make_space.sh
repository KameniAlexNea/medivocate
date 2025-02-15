rm -rf data/space
rm -rf data/medivocate.zip
mkdir data/space
cp app.py data/space
cp load_data.py data/space

mkdir data/space/data
cp -r data/chroma_db data/space/data

mkdir data/space/src

mkdir data/space/src/rag_pipeline
cp -r src/rag_pipeline data/space/src/

mkdir data/space/src/utilities
cp -r src/utilities data/space/src/

mkdir data/space/src/vector_store
cp -r src/vector_store data/space/src/

cp requirements.txt data/space

find data/space/src -type d -name "__pycache__" -exec rm -rf {} +

zip -r data/medivocate.zip data/space