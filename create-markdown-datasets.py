import os
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# function that get all markdown files in path_to_dir
def get_markdown_files(path_to_dir, list_folders_to_skip=[".git", ".github", "node_modules", "public", "src", "test", "tests", "vendor"]):
    markdown_files = []
    for root, dirs, files in os.walk(path_to_dir):
        # Exclude folders in list_folders_to_skip
        dirs[:] = [d for d in dirs if d not in list_folders_to_skip]
        for file in files:
            if file.endswith(".md") or file.endswith(".adoc"):
                markdown_files.append(os.path.join(root, file))
    return markdown_files


# function that takes as argument a list of markdown files and returns a list of docs using UnstructuredMarkdownLoader
def get_docs(markdown_files):
    docs = []
    for markdown_file in markdown_files:
        loader = UnstructuredMarkdownLoader(markdown_file)
        doc = loader.load()[0]
        docs.append(doc)
    return docs

# function that return chunks of text from a list of docs, takes as parameters a list of docs and chunk_size (default=1000) and chunk_overlap (default=100)
# using RecursiveCharacterTextSplitter
def get_chunks(docs, chunk_size=1000, chunk_overlap=100):
    chunks = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split_documents(docs)
    return chunks

path_to_dir = "C:\\Users\\mrabb\\Documents\\GitHub\\corteza-docs"
markdown_files = get_markdown_files(path_to_dir, list_folders_to_skip=[".git", ".github", "node_modules", "test", "tests", "vendor"])
path_to_dir = "C:\\Users\\mrabb\\Documents\\GitHub\\corteza"
markdown_files += get_markdown_files(path_to_dir, list_folders_to_skip=[".git", ".github", "node_modules", "test", "tests", "vendor"])
print(len(markdown_files))

docs = get_docs(markdown_files)
print(docs[0])
print("------------------------------------------------------------------------")
print(docs[1])
print("------------------------------------------------------------------------")   
print(docs[-1])
print("------------------------------------------------------------------------")
print(docs[-2])