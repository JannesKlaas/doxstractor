# Doxstractor 📄➡️📊 

Doxstractor is a modular library to extract structured data from documents using LLMs.

There are many situations where you want to extract data such as numbers, text or categories from a bunch of documents. Doxstractor was created with M&A due dilligence in mind. When a company is sold, the prospective buyer will recieve a data room with everything from key employment contracts to real estate leases. 

People will then need to go through all these documents and extract key information, such as "How many stock options have been granted?" or "Does this lease contain a break clause?". This data is then first compiled into spreadsheets, and finally written up in the due dilligence report. It is tedious for the people doing it and expensive to the people buying the report.

## Tutorial
A full tutorial can be found [here](https://github.com/JannesKlaas/doxstractor/blob/main/tutorial_notebooks/tutorial.ipynb).
You can open it in Colab under [this link](https://colab.research.google.com/github/JannesKlaas/doxstractor/blob/main/tutorial_notebooks/tutorial.ipynb)


### Installation
Install using pip:
`pip install doxstractor`

### Concepts

Doxstractor has three key components: 

1. `models` wrap various NLP models in a convenient way.
2. `extractors` pass data and perform prompt engineering for `models` and postprocess model outputs.
3. `nodes` chain multiple `extractors`, so that multiple attributes can be extracted from a set of documents.

Say you are working on a transaction of a retailer. You have to review both leases for real estate as well as employment contracts for key personnel. Ugh! That can be a lot of annoying work! But don't worry, `doxstractor` is here to help.

### Setting up models

To keep things simple, let's say you want to extract the addresses of the leased buildings and the base salaries of the employees. Both should be clearly spelled out in the documents, so a model trained on [SQUAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) should do a good job. These models will extract actual sections from the documents, so the risk of hallucinations is not given (they might still extract a wrong section).


This is how you define a model:
```python
import doxstractor as dxc
model = dxc.TransformersQAModel(model="deepset/roberta-base-squad2", na_threshold = 0.2)
```

The `TransformersQAModel` uses Huggingface `transformers` under the hood. So any huggingface QA model will work. The `na_threshold` sets the minimum confidence the model needs to have. If it is below it will return "NA".

There are multiple models you can use. Some models are better at some tasks than others.
This is how you create a model using the Anthropic API.
```python
anthropic_model = dxc.AnthropicAPIModel(model="claude-3-haiku-20240307")
```
For this model to work you need to have an Anthropic API key set under the `ANTHROPIC_API_KEY` environment variable.

### Setting up extractors.
There are three types of extractors: `dxc.TextExtractor`, `dxc.NumericExtractor`, `dxc.CategoryExtractor` for text, numbers and categories respectively.

To we define our address extractor as follows:
```python
address_extractor = dxc.TextExtractor(
    name="address", # Identifies the extractor
    query="What is the address of the leased building?", # Passed to the model
    model=model, # The model we defined earlier.
)
```

From here, we can extract an address from a given text:
```python
address = address_extractor.extract(text)
```
```
[Out]: 6335 1St – Avenue South, Seattle, Washington.
```

To get our salary extractor for the employment contracts we do almost the same:
```python
salary_extractor = dxc.NumericExtractor(
    name="salary", 
    query="What is the base salary?", 
    model=model, 
)
```
```
[Out]: 575,000
```

We can use the same model as for the addresses, the extractor will ensure that the response is numeric.

The category extractor takes one extra argument which is the categories. We will set one up with the anthropic model we defined earlier.
```python
doc_classifier = dxc.CategoryExtractor(
    name="doctype", 
    query="What type of agreement is this?", 
    categories=["employment", "lease", "other"],
    model=anthropic_model
                                      )
```


### Setting up nodes
Nodes allow us to chain multiple extractors together. This is useful when we want to extract multiple attributes.

In our case, we want to extract the address attribute when the document is a lease, and the salary when the document is an employment contract. `Node` allows for this conditionality. We place the classifier into the node itself. We then define two child nodes, for the salary and address. The keys of the children correspond to the categories of the classifier, so when the classifier picks a category, the node will then call the children of that category.

```python
chain = dxc.Node(extractor=doc_classifier, 
                 children={
                     "lease":[dxc.Node(address_extractor)], 
                     "employment": [dxc.Node(salary_extractor)]
                     }
                )
```

The nesting can go arbitarily deep. Only categorical extractors can be used for conditional child nodes.

Extraction with a chain works just the same as it does with a single extractor.
```python
chain.extract(text)
```

```
[Out]: {'doctype': 'employment', 'salary': '575,000'}
```


### Extracting data from folders

To extract data from folders, you simply loop over the files and apply your chain to all documents.
```python
path = "tutorial_data/"
file_paths = glob.glob(os.path.join(path, '*'))

collector = []
for fp in file_paths:
    with open(fp, "r") as f:
        html = f.read()
    soup = BeautifulSoup(html, features="html.parser")
    text = soup.get_text()
    data = chain.extract(text)
    data.update({"file_path":fp})
    collector.append(data)

pd.DataFrame(collector)[["file_path", "doctype", "salary", "address"]].sort_values("doctype")
```

|index|file\_path|doctype|salary|address|
|---|---|---|---|---|
|2|tutorial\_data/EDGAR\_employment\_agreement\_3\.html|employment||NaN|
|3|tutorial\_data/EDGAR\_employment\_agreement\_2\.html|employment|350,000|NaN|
|4|tutorial\_data/EDGAR\_employment\_agreement\_1\.html|employment|575,000|NaN|
|0|tutorial\_data/EDGAR\_lease\_agreement\_2\.html|lease|NaN| 3850 Annapolis Lane,|
|1|tutorial\_data/EDGAR\_lease\_agreement\_1\.html|lease|NaN| 6335 1St – Avenue South, Seattle, Washington\.|

The table above is correct (I checked the documents), except that it omitted one salary which is actually specified in the document. A better model can fix this.