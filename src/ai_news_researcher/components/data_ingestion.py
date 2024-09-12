from langchain.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import json

# Custom class to manage summaries
class SummaryManager:
    def __init__(self):
        self.summaries = []

    def add_summary(self, summary):
        # Create a LangChain Document object and add it to the summaries list
        doc = Document(page_content=summary)  # Use 'page_content' instead of 'content'
        self.summaries.append(doc)

    def get_summaries(self):
        # Return all stored summaries
        return self.summaries
    
    def save_to_json(self, filename):
        # Save summaries to a JSON file
        with open(filename, 'w') as f:
            json.dump([{"summary": doc.page_content} for doc in self.summaries], f)

# Your existing code
url = "https://news.google.com/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US%3Aen"
loader = AsyncChromiumLoader([url])
tt = Html2TextTransformer()
docs = tt.transform_documents(loader.load())
ts = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
fd = ts.split_documents(docs)

print(len(fd))

# Create an instance of the summary manager
summary_manager = SummaryManager()

for xx in fd:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "you are a helpful intelligent assistant"},
            {"role": "user", "content": f"summarize the following into bullet points, only consider meaningful sentences, also ignore all headings and words:\n\n{xx}"}
        ]
    )
    summary = response["choices"][0]["message"]["content"]
    summary_manager.add_summary(summary)  # Add summary to the manager

# Retrieve all summaries
saved_summaries = summary_manager.get_summaries()

# Print the saved summaries
print(f"Total summaries saved: {len(saved_summaries)}")
for i, doc in enumerate(saved_summaries):
    print(f"Summary {i + 1}:\n{doc.page_content}\n")  # Use 'page_content' to access the text

# Save summaries to a JSON file
summary_manager.save_to_json('summaries.json')
print("Summaries saved to summaries.json")