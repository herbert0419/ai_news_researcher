import pandas as pd
from langchain.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter

url = "https://news.google.com/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US%3Aen"
loader = AsyncChromiumLoader([url])
tt = Html2TextTransformer()
docs = tt.transform_documents(loader.load())
ts = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
fd = ts.split_documents(docs)

print(len(fd))
summaries = []
for xx in fd:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "you are a helpful intelligent assistant"},
            {"role": "user", "content": f"summarize the following into bullet points, only consider meaningful sentences, also ignore all headings and words:\n\n{xx}"}
        ]
    )
    summaries.append(response["choices"][0]["message"]["content"])

# Save summaries to an Excel file
df = pd.DataFrame(summaries, columns=["Summary"])
df.to_excel("summaries.xlsx", index=False)

print("Summaries saved to summaries.xlsx")
