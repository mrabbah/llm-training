from langchain.document_loaders.hugging_face_dataset import HuggingFaceDatasetLoader
from langchain.indexes import VectorstoreIndexCreator
dataset_name = "imdb"
page_content_column = "text"

loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)

data = loader.load()

print(data[:15])

# In this example, we use data from a dataset to answer a question
 
dataset_name = "tweet_eval"
page_content_column = "text"
name = "stance_climate"

loader = HuggingFaceDatasetLoader(dataset_name, page_content_column, name)

index = VectorstoreIndexCreator().from_loaders([loader])
query = "What are the most used hashtag?"
result = index.query(query)
print(result)