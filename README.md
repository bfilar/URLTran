# URLTran
PyTorch implementation of Improving Phishing URL Detection via Transformers [Paper](https://arxiv.org/pdf/2106.05256.pdf)

## Data
The paper used ~1.8M URLs (90/10 split on benign vs. malicious). There are few places to gather malicious URLs. My recommendation is to do the following:

### Malicious URLs
__OpenPhish__ will provide 500 malicious URLs for free in TXT form. You can access that data [here](https://openphish.com/phishing_database.html).

Likewise, __PhishTank__ is an excellent resource that provides a daily feed of malicious URLs in CSV or JSON format. You can gather ~5K through the following [link](https://www.phishtank.com/developer_info.php).

Finally, there is an excellent OpenSource project, [Phishing.Database](https://github.com/mitchellkrogza/Phishing.Database), run by Mitchell Krog. There is a ton of data available here to plus up your dataset.

### Benign Data
I gathered benign URL data via two methods. The first was to use the top 50K domains from [Alexa](http://s3.amazonaws.com/alexa-static/top-1m.csv.zip).

Next I used my own Chrome browser history to get an additional 60K. It was pretty easy to do on my Macbook. First, make sure your browser is closed. Then in your terminal run the following command:

```bash
/usr/bin/sqlite3 -csv -header ~/Library/Application\ Support/Google/Chrome/Default/History "SELECT urls.id, urls.url FROM urls JOIN visits ON urls.id = visits.url LEFT JOIN visit_source ON visits.id = visit_source.id order by last_visit_time asc;" > history.csv
```

## Tasks
Parameters were all gathered from the URLTran paper.
### Masked Language Modeling
Masked Language Modeling (MLM) is a commonly used pre-training task for transformers. The task consists of randomly selecting a subset of tokens to be replaced by a special ‘[MASK]’ token. Then we seek to minimize cross-entropy loss corresponding to the prediction of correct tokens at masked positions. The original BERT paper uses the following methodology for `[MASK]` selection:
- 15% of the tokens were uniformly selected for masking
- Of those
  - 80% are replaced
  - 10% were left unchanged
  - 10% were replaced by a random vocabulary token at each iteration

```python
# Input from mlm.py:
    url = "huggingface.co/docs/transformers/task_summary"
    input_ids, output_ids = predict_mask(url, tokenizer, model)

# Output:
    Masked Input: [CLS]huggingface.co[MASK]docs[MASK]transformers/task_summary[SEP]
    Predicted Output: [CLS]huggingface.co/docs/transformers/task_summary[SEP]
```

### Fine-Tuning
Access the fine-tuning step in `classifier.py`

## ToDo
There are a few different variations I need to complete:
1. Vary the number of layers between {3, 6, 12} for `URLTran_CustVoc`.
2. Vary the number of tokens per input URL sequence between `{128, 256}`.
3. Use both a byte-level and character-level BPE tokenizer w/ 1K- and 10K-sized vocabs.