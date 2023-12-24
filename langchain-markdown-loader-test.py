from langchain.document_loaders import UnstructuredMarkdownLoader
markdown_path = "./md1.md"
loader = UnstructuredMarkdownLoader(markdown_path)
data = loader.load()
print(data)
print("------------------------------------------------------------------------")
loader = UnstructuredMarkdownLoader(markdown_path, mode="elements")
data = loader.load()
print(data[0])